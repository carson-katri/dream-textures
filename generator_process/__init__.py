import json
import subprocess
import sys
import os
import threading
import site
import traceback
import numpy as np
from multiprocessing.shared_memory import SharedMemory

from .action import ACTION_BYTE_LENGTH, Action
from .intent import INTENT_BYTE_LENGTH, Intent

from .registrar import registrar
from .intents.apply_ocio_transforms import *
from .intents.prompt_to_image import *
from .intents.send_stop import *
from .intents.upscale import *

MISSING_DEPENDENCIES_ERROR = "Python dependencies are missing. Click Download Latest Release to fix."

_shared_instance = None
class GeneratorProcess():
    def __init__(self):
        import bpy
        env = os.environ.copy()
        env.pop('PYTHONPATH', None) # in case if --python-use-system-env
        self.process = subprocess.Popen(
            [sys.executable,'-s','generator_process',bpy.app.binary_path],
            cwd=os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env
        )
        self.reader = self.process.stdout
        self.queue = []
        self.in_use = False
        self.killed = False
        self.thread = threading.Thread(target=self._run,daemon=True,name="BackgroundReader")
        self.thread.start()

        for intent in registrar._generator_intents:
            # Bind self with __get__
            setattr(self, intent.name, intent.func.__get__(self, GeneratorProcess))
    
    @classmethod
    def shared(self, create=True):
        global _shared_instance
        if _shared_instance is None and create:
            _shared_instance = GeneratorProcess()
        return _shared_instance
    
    @classmethod
    def kill_shared(self):
        global _shared_instance
        if _shared_instance is None:
            return
        _shared_instance.kill()
        _shared_instance = None
    
    @classmethod
    def can_use(self):
        self = self.shared(False)
        return not (self and self.in_use)
    
    def kill(self):
        self.killed = True
        self.process.kill()
    
    def send_intent(self, intent, *, payload = None, **kwargs):
        """Sends intent messages to backend.

        Arguments:
        * intent -- Intent enum or int
        * payload -- Bytes-like value that is not suitable for json, it is recommended to use a shared memory approach instead if the payload is large
        * **kwargs -- json serializable key-value pairs used for subprocess function arguments
        """
        if Intent(intent) == Intent.UNKNOWN:
            raise ValueError(f"Internal error, invalid Intent: {intent}")
        kwargs_len = payload_len = b'\x00'*8
        if kwargs:
            kwargs = bytes(json.dumps(kwargs), encoding='utf-8')
            kwargs_len = len(kwargs).to_bytes(len(kwargs_len), sys.byteorder, signed=False)
        if payload is not None:
            payload = bytes(payload)
            payload_len = len(payload).to_bytes(len(payload_len), sys.byteorder, signed=False)
        # keep all checks before writing so ipc doesn't get broken intents

        stdin = self.process.stdin
        stdin.write(intent.to_bytes(INTENT_BYTE_LENGTH, sys.byteorder, signed=False))
        stdin.write(kwargs_len)
        if kwargs:
            stdin.write(kwargs)
        stdin.write(payload_len)
        if payload:
            stdin.write(payload)
        stdin.flush()

    def _run(self):
        reader = self.reader
        def readUInt(length):
            return int.from_bytes(reader.read(length),sys.byteorder,signed=False)

        queue = self.queue
        def queue_exception_msg(msg):
            queue.append((Action.EXCEPTION, {'fatal': True, 'msg': msg, 'trace': None}))

        while not self.killed:
            action = readUInt(ACTION_BYTE_LENGTH)
            if action == Action.CLOSED:
                if not self.killed:
                    queue_exception_msg("Process closed unexpectedly")
                return
            kwargs_len = readUInt(8)
            kwargs = {} if kwargs_len == 0 else json.loads(reader.read(kwargs_len))
            payload_len = readUInt(8)
            if payload_len > 0:
                kwargs['payload'] = reader.read(payload_len)

            if action in [Action.INFO, Action.STEP_NO_SHOW, Action.IMAGE, Action.STEP_IMAGE, Action.STOPPED]:
                queue.append((action, kwargs))
            elif action == Action.EXCEPTION:
                queue.append((action, kwargs))
                if kwargs['fatal']:
                    return
            else:
                queue_exception_msg(f"Internal error, unexpected action id: {action}")
                return

BYTE_TO_NORMALIZED = 1.0 / 255.0
class Backend():
    def __init__(self):
        self.intent_backends = {}
        self.stdin = sys.stdin.buffer
        self.stdout = sys.stdout.buffer
        sys.stdout = open(os.devnull, 'w') # prevent stable diffusion logs from breaking ipc
        self.stderr = sys.stderr
        self.shared_memory = None
        self.stop_requested = False
        self.intent = Intent.UNKNOWN
        self.stopped_was_sent = False
        self.queue = []
        self.queue_appended = threading.Event()
        self.thread = threading.Thread(target=self._run,daemon=True,name="BackgroundReader")

    def check_stop(self):
        if self.stop_requested:
            self.stop_requested = False
            raise KeyboardInterrupt

    def send_action(self, action, *, payload = None, **kwargs):
        """Sends action messages to frontend.

        Arguments:
        * action -- Action enum or int
        * payload -- Bytes-like value that is not suitable for json, it is recommended to use a shared memory approach instead if the payload is large
        * **kwargs -- json serializable key-value pairs used for callback function arguments
        """
        if Action(action) == Action.UNKNOWN:
            raise ValueError(f"Internal error, invalid Action: {action}")
        kwargs_len = payload_len = b'\x00'*8
        if kwargs:
            kwargs = bytes(json.dumps(kwargs), encoding='utf-8')
            kwargs_len = len(kwargs).to_bytes(len(kwargs_len), sys.byteorder, signed=False)
        if payload is not None:
            payload = bytes(payload)
            payload_len = len(payload).to_bytes(len(payload_len), sys.byteorder, signed=False)
        # keep all checks before writing so ipc doesn't get broken actions

        self.stdout.write(action.to_bytes(ACTION_BYTE_LENGTH, sys.byteorder, signed=False))
        self.stdout.write(kwargs_len)
        if kwargs:
            self.stdout.write(kwargs)
        self.stdout.write(payload_len)
        if payload:
            self.stdout.write(payload)
        self.stdout.flush()
        if action in [Action.EXCEPTION, Action.STOPPED]:
            self.stopped_was_sent = True

    def send_info(self, msg):
        """Sends information to be shown to the user before generation begins."""
        self.send_action(Action.INFO, msg=msg)

    def send_exception(self, fatal = True, msg: str = None, trace: str = None):
        """Send exception information to frontend. When called within an except block arguments can be inferred.

        Arguments:
        * fatal -- whether the subprocess should be killed
        * msg -- user notified prompt
        * trace -- traceback string
        """
        exc = sys.exc_info()
        if msg is None:
            msg = repr(exc[1]) if exc[1] is not None else "Internal error, see system console for details"
        if trace is None and exc[2] is not None:
            trace = traceback.format_exc()
        if msg is None and trace is None:
            raise TypeError("msg and trace cannot be None outside of an except block")
        self.send_action(Action.EXCEPTION, fatal=fatal, msg=msg, trace=trace)
        if fatal:
            sys.exit(1)

    def share_image_memory(self, image):
        from PIL import ImageOps
        image_bytes = (np.asarray(ImageOps.flip(image).convert('RGBA'),dtype=np.float32) * BYTE_TO_NORMALIZED).tobytes()
        image_bytes_len = len(image_bytes)
        shared_memory = self.shared_memory
        if shared_memory is None or shared_memory.size != image_bytes_len:
            if shared_memory is not None:
                shared_memory.close()
            self.shared_memory = shared_memory = SharedMemory(create=True, size=image_bytes_len)
        shared_memory.buf[:] = image_bytes
        return shared_memory.name
    
    """ intent generator function format
    def intent(self):
        args = yield
        # imports and prior setup
        while True:
            try: # try...except is only needed if you call self.check_stop() within
                ... # execute intent
            except KeyboardInterrupt:
                pass
            args = yield
    """

    def _run(self):
        reader = self.stdin
        def readUInt(length):
            return int.from_bytes(reader.read(length),sys.byteorder,signed=False)

        while True:
            intent = readUInt(INTENT_BYTE_LENGTH)
            json_len = readUInt(8)
            if json_len == 0:
                return # stdin closed
            args = json.loads(reader.read(json_len))
            payload_len = readUInt(8)
            if payload_len > 0:
                args['payload'] = reader.read(payload_len)
            if intent == Intent.STOP:
                if 'stop_intent' in args and self.intent == args['stop_intent']:
                    self.stop_requested = True
            else:
                self.queue.append((intent, args))
                self.queue_appended.set()
    
    def main_loop(self):
        intents = {}
        for intent in registrar._intent_backends:
            intents[intent.intent] = intent.func(self)
        for fn in intents.values():
            next(fn)

        while True:
            if len(self.queue) == 0:
                self.queue_appended.clear()
                self.queue_appended.wait()
            (intent, args) = self.queue.pop(0)
            if intent in intents:
                self.intent = intent
                self.stopped_was_sent = False
                intents[intent].send(args)
                self.stop_requested = False
                self.intent = Intent.UNKNOWN
                if not self.stopped_was_sent:
                    self.send_action(Action.STOPPED)
            else:
                self.send_exception(True, f"Unknown intent {intent} sent to process. Expected one of {Intent._member_names_}.")

def main():
    back = Backend()
    try:
        if sys.platform == 'win32':
            from ctypes import WinDLL
            WinDLL(os.path.join(os.path.dirname(sys.argv[1]),"python3.dll")) # fix for ImportError: DLL load failed while importing cv2: The specified module could not be found.

        from absolute_path import absolute_path, CLIPSEG_WEIGHTS_PATH
        # Support Apple Silicon GPUs as much as possible.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Move Python runtime paths to end. (prioritize addon modules)
        paths = sys.path[1:]
        sys.path[:] = sys.path[0:1]

        sys.path.append(absolute_path("stable_diffusion/"))
        sys.path.append(absolute_path("stable_diffusion/src/clip"))
        sys.path.append(absolute_path("stable_diffusion/src/k-diffusion"))
        sys.path.append(absolute_path("stable_diffusion/src/taming-transformers"))
        sys.path.append(absolute_path("stable_diffusion/src/clipseg"))
        site.addsitedir(absolute_path(".python_dependencies"))
        sys.path.extend(paths)

        import pkg_resources
        pkg_resources._initialize_master_working_set()

        if sys.platform == 'win32':
            import scipy # This will hang when loading libbanded5x.Q3V52YHHGVBP5BKVHJ5RHQVFWHHSLVWO.gfortran-win_amd64.dll
                         # if imported while background reading thread is waiting on next intent.
                         # Hurray for more strange .dll bugs

        from ldm.invoke import txt2mask
        txt2mask.CLIPSEG_WEIGHTS = CLIPSEG_WEIGHTS_PATH

        back.thread.start()
        back.main_loop()
    except SystemExit:
        pass
    except:
        back.send_exception()