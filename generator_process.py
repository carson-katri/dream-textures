import json
from math import ceil, log
import subprocess
import sys
import os
import threading
import site
import traceback
import numpy as np
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory

MISSING_DEPENDENCIES_ERROR = "Python dependencies are missing. Click Download Latest Release to fix."

class Action(IntEnum):
    """IPC message types sent from backend to frontend"""
    UNKNOWN = -1 # placeholder so you can do Action(int).name or Action(int) == Action.UNKNOWN when int is invalid
                 # don't add anymore negative actions
    CLOSED = 0 # is not sent during normal operation, just allows for a simple way of detecting when the subprocess is closed
    INFO = 1
    IMAGE = 2
    STEP_IMAGE = 3
    STEP_NO_SHOW = 4
    EXCEPTION = 5
    STOPPED = 6

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN
ACTION_BYTE_LENGTH = ceil(log(max(Action)+1,256))

class Intent(IntEnum):
    """IPC messages types sent from frontend to backend"""
    UNKNOWN = -1

    PROMPT_TO_IMAGE = 0
    UPSCALE = 1
    STOP = 2

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN
INTENT_BYTE_LENGTH = ceil(log(max(Intent)+1,256))



def block_in_use(func):
    def block(self, *args, **kwargs):
        if self.in_use:
            raise RuntimeError(f"Can't call {func.__qualname__} while process is in use")
        try:
            self.in_use = True
            yield from func(self, *args, **kwargs)
        finally:
            self.in_use = False
    return block

_shared_instance = None
class GeneratorProcess():
    def __init__(self):
        import bpy
        self.process = subprocess.Popen([sys.executable,'generator_process.py',bpy.app.binary_path],cwd=os.path.dirname(os.path.realpath(__file__)),stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        self.reader = self.process.stdout
        self.queue = []
        self.in_use = False
        self.killed = False
        self.thread = threading.Thread(target=self._run,daemon=True,name="BackgroundReader")
        self.thread.start()
    
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

    def send_stop(self, stop_intent):
        self.send_intent(Intent.STOP, stop_intent=stop_intent)
    
    @block_in_use
    def prompt2image(self, args, step_callback, image_callback, info_callback, exception_callback):
        self.send_intent(Intent.PROMPT_TO_IMAGE, **args)

        queue = self.queue
        callbacks = {
            Action.INFO: info_callback,
            Action.IMAGE: image_callback,
            Action.STEP_IMAGE: step_callback,
            Action.STEP_NO_SHOW: step_callback,
            Action.EXCEPTION: exception_callback,
            Action.STOPPED: lambda: None
        }

        while True:
            while len(queue) == 0:
                yield # nothing in queue, let blender resume
            tup = queue.pop(0)
            action = tup[0]
            callbacks[action](**tup[1])
            if action in [Action.STOPPED, Action.EXCEPTION]:
                return
    
    @block_in_use
    def upscale(self, args, image_callback, info_callback, exception_callback):
        self.send_intent(Intent.UPSCALE, **args)

        queue = self.queue
        callbacks = {
            Action.INFO: info_callback,
            Action.IMAGE: image_callback,
            Action.EXCEPTION: exception_callback,
            Action.STOPPED: lambda: None
        }

        while True:
            while len(queue) == 0:
                yield
            tup = queue.pop(0)
            action = tup[0]
            callbacks[action](**tup[1])
            if action in [Action.STOPPED, Action.EXCEPTION]:
                return

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

    def prompt_to_image(self):
        args = yield
        self.send_info("Importing Dependencies")
        from absolute_path import absolute_path
        from stable_diffusion.ldm.generate import Generate
        from stable_diffusion.ldm.dream.devices import choose_precision
        from omegaconf import OmegaConf
        from io import StringIO
        models_config  = absolute_path('stable_diffusion/configs/models.yaml')
        model   = 'stable-diffusion-1.4'

        models  = OmegaConf.load(models_config)
        config  = absolute_path('stable_diffusion/' + models[model].config)
        weights = absolute_path('stable_diffusion/' + models[model].weights)
        generator: Generate = None

        def image_writer(image, seed, upscaled=False, first_seed=None):
            # Only use the non-upscaled texture, as upscaling is a separate step in this addon.
            if not upscaled:
                self.send_action(Action.IMAGE, shared_memory_name=self.share_image_memory(image), seed=seed, width=image.width, height=image.height)

        step = 0
        def view_step(samples, i):
            self.check_stop()
            nonlocal step
            step = i
            if args['show_steps']:
                image = generator.sample_to_image(samples)
                self.send_action(Action.STEP_IMAGE, shared_memory_name=self.share_image_memory(image), step=step, width=image.width, height=image.height)
            else:
                self.send_action(Action.STEP_NO_SHOW, step=step)

        def preload_models():
            tqdm = None
            try:
                from huggingface_hub.utils.tqdm import tqdm as hfh_tqdm
                tqdm = hfh_tqdm
            except:
                try:
                    from tqdm.auto import tqdm as auto_tqdm
                    tqdm = auto_tqdm
                except:
                    return

            current_model_name = ""
            def start_preloading(model_name):
                nonlocal current_model_name
                current_model_name = model_name
                self.send_info(f"Loading {model_name}")

            def update_decorator(original):
                def update(self, n=1):
                    result = original(self, n)
                    nonlocal current_model_name
                    frac = self.n / self.total
                    percentage = int(frac * 100)
                    if self.n - self.last_print_n >= self.miniters:
                        self.send_info(f"Downloading {current_model_name} ({percentage}%)")
                    return result
                return update
            old_update = tqdm.update
            tqdm.update = update_decorator(tqdm.update)

            import warnings
            import transformers
            transformers.logging.set_verbosity_error()

            start_preloading("BERT tokenizer")
            transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

            self.send_info("Preloading `kornia` requirements")
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                import kornia

            start_preloading("CLIP")
            clip_version = 'openai/clip-vit-large-patch14'
            transformers.CLIPTokenizer.from_pretrained(clip_version)
            transformers.CLIPTextModel.from_pretrained(clip_version)

            tqdm.update = old_update
        
        from transformers.utils.hub import TRANSFORMERS_CACHE
        model_paths = {'bert-base-uncased', 'openai--clip-vit-large-patch14'}
        if any(not os.path.isdir(os.path.join(TRANSFORMERS_CACHE, f'models--{path}')) for path in model_paths):
            preload_models()

        while True:
            try:
                self.check_stop()
                # Reset the step count
                step = 0

                if generator is None or generator.precision != (choose_precision(generator.device) if args['precision'] == 'auto' else args['precision']):
                    self.send_info("Loading Model")
                    generator = Generate(
                        conf=models_config,
                        model=model,
                        # These args are deprecated, but we need them to specify an absolute path to the weights.
                        weights=weights,
                        config=config,
                        precision=args['precision']
                    )
                    generator.free_gpu_mem = False # Not sure what this is for, and why it isn't a flag but read from Args()?
                    generator.load_model()
                    self.check_stop()
                self.send_info("Starting")
                
                tmp_stderr = sys.stderr = StringIO() # prompt2image writes exceptions straight to stderr, intercepting
                generator.prompt2image(
                    # a function or method that will be called each step
                    step_callback=view_step,
                    # a function or method that will be called each time an image is generated
                    image_callback=image_writer,
                    **args
                )
                if tmp_stderr.tell() > 0:
                    tmp_stderr.seek(0)
                    s = tmp_stderr.read()
                    i = s.find("Traceback") # progress also gets printed to stderr so check for an actual exception
                    if i != -1:
                        s = s[i:]
                        import re
                        low_ram = re.search(r"(Not enough memory, use lower resolution)( \(max approx. \d+x\d+\))",s,re.IGNORECASE)
                        if low_ram:
                            self.send_exception(False, f"{low_ram[1]}{' or disable full precision' if args['precision'] == 'float32' else ''}{low_ram[2]}", s)
                        elif s.find("CUDA out of memory. Tried to allocate") != -1:
                            self.send_exception(False, f"Not enough memory, use lower resolution{' or disable full precision' if args['precision'] == 'float32' else ''}", s)
                        else:
                            self.send_exception(True, msg=None, trace=s) # consider all unknown exceptions to be fatal so the generator process is fully restarted next time
            except KeyboardInterrupt:
                pass
            finally:
                sys.stderr = self.stderr
            args = yield

    def upscale(self):
        args = yield
        self.send_info("Starting")
        from absolute_path import REAL_ESRGAN_WEIGHTS_PATH
        import cv2
        from PIL import Image
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        while True:
            image = cv2.imread(args['input'], cv2.IMREAD_UNCHANGED)
            padding = 32
            if args['seamless']:
                image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'wrap')

            real_esrgan_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            self.send_info("Loading Upsampler")
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=REAL_ESRGAN_WEIGHTS_PATH,
                model=real_esrgan_model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=not args['full_precision']
            )
            self.send_info("Enhancing Input")
            output, _ = upsampler.enhance(image, outscale=args['outscale'])
            if args['seamless']:
                padding *= args['outscale']
                output = output[padding:-padding, padding:-padding]
            self.send_info("Converting Result")
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output = Image.fromarray(output)
            self.send_action(Action.IMAGE, shared_memory_name=self.share_image_memory(output), seed=args['name'], width=output.width, height=output.height)
            args = yield
    
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
        intents = {
            Intent.PROMPT_TO_IMAGE: self.prompt_to_image(),
            Intent.UPSCALE: self.upscale(),
        }
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
    try:
        back = Backend()

        if sys.platform == 'win32':
            from ctypes import WinDLL
            WinDLL(os.path.join(os.path.dirname(sys.argv[1]),"python3.dll")) # fix for ImportError: DLL load failed while importing cv2: The specified module could not be found.

        from absolute_path import absolute_path
        # Support Apple Silicon GPUs as much as possible.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        sys.path.append(absolute_path("stable_diffusion/"))
        sys.path.append(absolute_path("stable_diffusion/src/clip"))
        sys.path.append(absolute_path("stable_diffusion/src/k-diffusion"))
        sys.path.append(absolute_path("stable_diffusion/src/taming-transformers"))

        site.addsitedir(absolute_path(".python_dependencies"))
        import pkg_resources
        pkg_resources._initialize_master_working_set()

        if sys.platform == 'win32':
            import scipy # This will hang when loading libbanded5x.Q3V52YHHGVBP5BKVHJ5RHQVFWHHSLVWO.gfortran-win_amd64.dll
                         # if imported while background reading thread is waiting on next intent.
                         # Hurray for more strange .dll bugs

        back.thread.start()
        back.main_loop()
    except SystemExit:
        pass
    except:
        back.send_exception()

if __name__ == "__main__":
    main()