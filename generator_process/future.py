import functools
import threading
from typing import Callable, Any, MutableSet

class Future:
    """
    Object that represents a value that has not completed processing, but will in the future.

    Add callbacks to be notified when values become available, or use `.result()` and `.exception()` to wait for the value.
    """
    _response_callbacks: MutableSet[Callable[['Future', Any], None]] = set()
    _exception_callbacks: MutableSet[Callable[['Future', BaseException], None]] = set()
    _done_callbacks: MutableSet[Callable[['Future'], None]] = set()
    _responses: list = []
    _exception: BaseException | None = None
    _done_event: threading.Event
    done: bool = False
    cancelled: bool = False
    check_cancelled: Callable[[], bool] = lambda: False
    call_done_on_exception: bool = True

    def __init__(self):
        self._response_callbacks = set()
        self._exception_callbacks = set()
        self._done_callbacks = set()
        self._responses = []
        self._exception = None
        self._done_event = threading.Event()
        self.done = False
        self.cancelled = False
        self.call_done_on_exception = True

    def result(self, last_only=False):
        """
        Get the result value (blocking).
        """
        def _response():
            match len(self._responses):
                case 0:
                    return None
                case 1:
                    return self._responses[0]
                case _:
                    return self._responses[-1] if last_only else self._responses
        if self._exception is not None:
            raise self._exception
        if self.done:
            return _response()
        else:
            self._done_event.wait()
            if self._exception is not None:
                raise self._exception
            return _response()
    
    def exception(self):
        if self.done:
            return self._exception
        else:
            self._done_event.wait()
            return self._exception
    
    def cancel(self):
        self.cancelled = True

    def _run_on_main_thread(self, func):
        if threading.current_thread() == threading.main_thread():
            func()
            return
        try:
            import bpy
            bpy.app.timers.register(func, persistent=True)
        except:
            func()

    def add_response(self, response):
        """
        Add a response value and notify all consumers.
        """
        self._responses.append(response)
        def run_callbacks():
            for response_callback in self._response_callbacks:
                response_callback(self, response)
        self._run_on_main_thread(run_callbacks)

    def set_exception(self, exception: BaseException):
        """
        Set the exception.
        """
        self._exception = exception
        def run_callbacks():
            for exception_callback in self._exception_callbacks:
                exception_callback(self, exception)
        self._run_on_main_thread(run_callbacks)

    def set_done(self):
        """
        Mark the future as done.
        """
        assert not self.done
        self.done = True
        self._done_event.set()
        if self._exception is None or self.call_done_on_exception:
            def run_callbacks():
                for done_callback in self._done_callbacks:
                    done_callback(self)
            self._run_on_main_thread(run_callbacks)

    def add_response_callback(self, callback: Callable[['Future', Any], None]):
        """
        Add a callback to run whenever a response is received.
        Will be called multiple times by generator functions.
        """
        self._response_callbacks.add(callback)
    
    def add_exception_callback(self, callback: Callable[['Future', BaseException], None]):
        """
        Add a callback to run when the future errors.
        Will only be called once at the first exception.
        """
        self._exception_callbacks.add(callback)
        if self._exception is not None:
            self._run_on_main_thread(functools.partial(callback, self, self._exception))

    def add_done_callback(self, callback: Callable[['Future'], None]):
        """
        Add a callback to run when the future is marked as done.
        Will only be called once.
        """
        self._done_callbacks.add(callback)
        if self.done:
            self._run_on_main_thread(functools.partial(callback, self))
