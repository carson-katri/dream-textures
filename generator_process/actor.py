from multiprocessing import Queue, Lock, current_process, get_context
import multiprocessing.synchronize
import enum
import traceback
import threading
from typing import Type, TypeVar, Generator
import site
import sys
import os
from ..absolute_path import absolute_path
from .future import Future

def _load_dependencies():
    site.addsitedir(absolute_path(".python_dependencies"))
    deps = sys.path.pop(-1)
    sys.path.insert(0, deps)
    if sys.platform == 'win32':
        # fix for ImportError: DLL load failed while importing cv2: The specified module could not be found.
        # cv2 needs python3.dll, which is stored in Blender's root directory instead of its python directory.
        python3_path = os.path.abspath(os.path.join(sys.executable, "..\\..\\..\\..\\python3.dll"))
        if os.path.exists(python3_path):
            os.add_dll_directory(os.path.dirname(python3_path))
if current_process().name == "__actor__":
    _load_dependencies()

class ActorContext(enum.IntEnum):
    """
    The context of an `Actor` object.
    
    One `Actor` instance is the `FRONTEND`, while the other instance is the backend, which runs in a separate process.
    The `FRONTEND` sends messages to the `BACKEND`, which does work and returns a result.
    """
    FRONTEND = 0
    BACKEND = 1

class Message:
    """
    Represents a function signature with a method name, positonal arguments, and keyword arguments.

    Note: All arguments must be picklable.
    """

    def __init__(self, method_name, args, kwargs):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
    
    CANCEL = "__cancel__"
    END = "__end__"

def _start_backend(cls, message_queue, response_queue):
    cls(
        ActorContext.BACKEND,
        message_queue=message_queue,
        response_queue=response_queue
    ).start()

class TracedError(BaseException):
    def __init__(self, base: BaseException, trace: str):
        self.base = base
        self.trace = trace

T = TypeVar('T', bound='Actor')

class Actor:
    """
    Base class for specialized actors.
    
    Uses queues to send actions to a background process and receive a response.
    Calls to any method declared by the frontend are automatically dispatched to the backend.

    All function arguments must be picklable.
    """

    _message_queue: Queue
    _response_queue: Queue
    _lock: multiprocessing.synchronize.Lock

    _shared_instance = None

    # Methods that are not used for message passing, and should not be overridden in `_setup`.
    _protected_methods = {
        "start",
        "close",
        "is_alive",
        "can_use",
        "shared"
    }

    def __init__(self, context: ActorContext, message_queue: Queue = None, response_queue: Queue = None):
        self.context = context
        self._message_queue = message_queue if message_queue is not None else get_context('spawn').Queue(maxsize=1)
        self._response_queue = response_queue if response_queue is not None else get_context('spawn').Queue(maxsize=1)
        self._setup()
        self.__class__._shared_instance = self
    
    def _setup(self):
        """
        Setup the Actor after initialization.
        """
        match self.context:
            case ActorContext.FRONTEND:
                self._lock = Lock()
                for name in filter(lambda name: callable(getattr(self, name)) and not name.startswith("_") and name not in self._protected_methods, dir(self)):
                    setattr(self, name, self._send(name))
            case ActorContext.BACKEND:
                pass

    @classmethod
    def shared(cls: Type[T]) -> T:
        return cls._shared_instance or cls(ActorContext.FRONTEND).start()

    def start(self: T) -> T:
        """
        Start the actor process.
        """
        match self.context:
            case ActorContext.FRONTEND:
                self.process = get_context('spawn').Process(target=_start_backend, args=(self.__class__, self._message_queue, self._response_queue), name="__actor__", daemon=True)
                self.process.start()
            case ActorContext.BACKEND:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                self._backend_loop()
        return self
    
    def close(self):
        """
        Stop the actor process.
        """
        match self.context:
            case ActorContext.FRONTEND:
                self.process.terminate()
                self._message_queue.close()
                self._response_queue.close()
            case ActorContext.BACKEND:
                pass
    
    @classmethod
    def shared_close(cls: Type[T]):
        if cls._shared_instance is None:
            return
        cls._shared_instance.close()
        cls._shared_instance = None
    
    def is_alive(self):
        match self.context:
            case ActorContext.FRONTEND:
                return self.process.is_alive()
            case ActorContext.BACKEND:
                return True

    def can_use(self):
        if result := self._lock.acquire(block=False):
            self._lock.release()
        return result

    def _backend_loop(self):
        while True:
            self._receive(self._message_queue.get())

    def _receive(self, message: Message):
        try:
            response = getattr(self, message.method_name)(*message.args, **message.kwargs)
            if isinstance(response, Generator):
                for res in iter(response):
                    extra_message = None
                    try:
                        extra_message = self._message_queue.get(block=False)
                    except:
                        pass
                    if extra_message == Message.CANCEL:
                        break
                    if isinstance(res, Future):
                        def check_cancelled():
                            try:
                                return self._message_queue.get(block=False) == Message.CANCEL
                            except:
                                return False
                        res.check_cancelled = check_cancelled
                        res.add_response_callback(lambda _, res: self._response_queue.put(res))
                        res.add_exception_callback(lambda _, e: self._response_queue.put(RuntimeError(repr(e))))
                        res.add_done_callback(lambda _: None)
                    else:
                        self._response_queue.put(res)
            else:
                self._response_queue.put(response)
        except Exception as e:
            trace = traceback.format_exc()
            try:
                if sys.modules[e.__module__].__file__.startswith(absolute_path(".python_dependencies")):
                    e = RuntimeError(repr(e))
                    # might be more suitable to have specific substitute exceptions for cases
                    # like torch.cuda.OutOfMemoryError for frontend handling in the future
            except (AttributeError, KeyError):
                pass
            self._response_queue.put(TracedError(e, trace))
        self._response_queue.put(Message.END)

    def _send(self, name):
        def _send(*args, _block=False, **kwargs):
            future = Future()
            def _send_thread(future: Future):
                self._lock.acquire()
                self._message_queue.put(Message(name, args, kwargs))

                while not future.done:
                    if future.cancelled:
                        self._message_queue.put(Message.CANCEL)
                    response = self._response_queue.get()
                    if response == Message.END:
                        future.set_done()
                    elif isinstance(response, TracedError):
                        response.base.__cause__ = Exception(response.trace)
                        future.set_exception(response.base)
                    elif isinstance(response, Exception):
                        future.set_exception(response)
                    else:
                        future.add_response(response)
                
                self._lock.release()
            if _block:
                _send_thread(future)
            else:
                thread = threading.Thread(target=_send_thread, args=(future,), daemon=True)
                thread.start()
            return future
        return _send
    
    def __del__(self):
        self.close()