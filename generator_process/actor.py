from multiprocessing import Queue, Process
import queue
import enum
import traceback
import threading
from typing import Type, TypeVar
from concurrent.futures import Future
import site

class ActorContext(enum.IntEnum):
    """
    The context of an `Actor` object.
    
    One `Actor` instance is the `FRONTEND`, while the other instance is the backend, which runs in a separate process.
    The `FRONTEND` sends messages to the `BACKEND`, which does work and returns a result.
    """
    FRONTEND = 0
    BACKEND = 1

class Message():
    """
    Represents a function signature with a method name, positonal arguments, and keyword arguments.

    Note: All arguments must be picklable.
    """

    def __init__(self, method_name, args, kwargs):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

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

class Actor():
    """
    Base class for specialized actors.
    
    Uses queues to serialize actions from different threads, and automatically dispatches methods to a separate process.
    """

    _message_queue: Queue
    _response_queue: Queue

    _shared_instance = None

    # Methods that are not used for message passing, and should not be overridden in `_setup`.
    _protected_methods = {
        "start",
        "close",
        "is_alive",
        "can_use",
        "shared"
    }

    def __init__(self, context: ActorContext, message_queue: Queue = Queue(), response_queue: Queue = Queue()):
        self.context = context
        self._message_queue = message_queue
        self._response_queue = response_queue
        self._setup()
        self.__class__._shared_instance = self
    
    def _setup(self):
        """
        Setup the Actor after initialization.
        """
        match self.context:
            case ActorContext.FRONTEND:
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
                self.process = Process(target=_start_backend, args=(self.__class__, self._message_queue, self._response_queue), name="__actor__")
                self.process.start()
            case ActorContext.BACKEND:
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
        cls._shared_instance.close()
        cls._shared_instance = None
    
    def is_alive(self):
        match self.context:
            case ActorContext.FRONTEND:
                return self.process.is_alive()
            case ActorContext.BACKEND:
                return True

    def can_use(self):
        return self._message_queue.empty() and self._response_queue.empty()
    
    def _load_dependencies(self):
        from ..absolute_path import absolute_path
        site.addsitedir(absolute_path(".python_dependencies"))

    def _backend_loop(self):
        self._load_dependencies()
        while True:
            self._receive(self._message_queue.get())

    def _receive(self, message: Message):
        try:
            response = getattr(self, message.method_name)(*message.args, **message.kwargs)
        except Exception as e:
            trace = traceback.format_exc()
            response = TracedError(e, trace)
        self._response_queue.put(response)

    def _send(self, name):
        def _send(*args, **kwargs):
            future = Future()
            def wait_for_response(future: Future):
                response = None
                while response is None:
                    try:
                        response = self._response_queue.get(block=False)
                    except queue.Empty:
                        continue
                if isinstance(response, TracedError):
                    response.base.__cause__ = Exception(response.trace)
                    future.set_exception(response.base)
                elif isinstance(response, Exception):
                    future.set_exception(response)
                else:
                    future.set_result(response)
            thread = threading.Thread(target=wait_for_response, args=(future,), daemon=True)
            thread.start()
            self._message_queue.put(Message(name, args, kwargs))
            return future
        return _send
    
    def __del__(self):
        self.close()