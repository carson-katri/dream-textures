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
import tempfile

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
    IMAGE_PATH = 6 # An image result that returns a path to a saved image instead of bytes

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

ACTION_BYTE_LENGTH = ceil(log(max(Action)+1,256)) # doubt there will ever be more than 255 actions, but just in case

class Intent(IntEnum):
    """IPC messages types sent from frontend to backend"""
    UNKNOWN = -1

    PROMPT_TO_IMAGE = 0
    UPSCALE = 1

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

_shared_instance = None
class GeneratorProcess():
    def __init__(self):
        self.process = subprocess.Popen([sys.executable,'generator_process.py'],cwd=os.path.dirname(os.path.realpath(__file__)),stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        self.reader = self.process.stdout
        self.queue = []
        self.args = None
        self.killed = False
        self.thread = threading.Thread(target=self._run,daemon=True,name="BackgroundReader")
        self.thread.start()
    
    @classmethod
    def shared(self):
        global _shared_instance
        if _shared_instance is None:
            _shared_instance = GeneratorProcess()
        return _shared_instance
    
    @classmethod
    def kill_shared(self):
        global _shared_instance
        if _shared_instance is None:
            return
        _shared_instance.kill()
        _shared_instance = None
    
    def kill(self):
        self.killed = True
        self.process.kill()
    
    def prompt2image(self, args, step_callback, image_callback, info_callback, exception_callback):
        self.args = args
        stdin = self.process.stdin
        stdin.write(Intent.PROMPT_TO_IMAGE.to_bytes(ACTION_BYTE_LENGTH, sys.byteorder, signed=False))
        stdin.flush()
        b = bytes(json.dumps(args), encoding='utf-8')
        stdin.write(len(b).to_bytes(8,sys.byteorder,signed=False))
        stdin.write(b)
        stdin.flush()

        queue = self.queue
        callbacks = {
            Action.INFO: info_callback,
            Action.IMAGE: image_callback,
            Action.STEP_IMAGE: step_callback,
            Action.STEP_NO_SHOW: step_callback,
            Action.EXCEPTION: exception_callback
        }

        for i in range(0,args['iterations']):
            while True:
                while len(queue) == 0:
                    yield # nothing in queue, let blender resume
                tup = queue.pop()
                action = tup[0]
                callbacks[action](**tup[1])
                if action == Action.IMAGE:
                    break
                elif action == Action.EXCEPTION:
                    return
    
    def upscale(self, args, image_callback, info_callback, exception_callback):
        stdin = self.process.stdin
        stdin.write(Intent.UPSCALE.to_bytes(ACTION_BYTE_LENGTH, sys.byteorder, signed=False))
        # stdin.flush()
        b = bytes(json.dumps(args), encoding='utf-8')
        stdin.write(len(b).to_bytes(8,sys.byteorder,signed=False))
        stdin.write(b)
        stdin.flush()

        queue = self.queue
        callbacks = {
            Action.INFO: info_callback,
            Action.IMAGE_PATH: image_callback,
            Action.EXCEPTION: exception_callback
        }

        while True:
            while len(queue) == 0:
                yield
            tup = queue.pop()
            action = tup[0]
            callbacks[action](**tup[1])
            if action == Action.IMAGE_PATH:
                break
            elif action == Action.EXCEPTION:
                return

    def _run(self):
        reader = self.reader
        def readUInt(length):
            return int.from_bytes(reader.read(length),sys.byteorder,signed=False)

        queue = self.queue
        def queue_exception_msg(msg):
            queue.append((Action.EXCEPTION, {'fatal': True, 'msg': msg, 'trace': None}))

        image_buffer = bytearray(512*512*16)
        while not self.killed:
            action = readUInt(ACTION_BYTE_LENGTH)
            if action == Action.CLOSED:
                if not self.killed:
                    queue_exception_msg("Process closed unexpectedly")
                return
            kwargs_len = readUInt(8)
            kwargs = {} if kwargs_len == 0 else json.loads(reader.read(kwargs_len))
            payload_len = readUInt(8)

            if action in [Action.INFO, Action.STEP_NO_SHOW, Action.IMAGE_PATH]:
                queue.append((action, kwargs))
            elif action in [Action.IMAGE, Action.STEP_IMAGE]:
                expected_len = kwargs['width']*kwargs['height']*16
                if payload_len != expected_len:
                    queue_exception_msg(f"Internal error, received image payload of {payload_len} bytes, expected {expected_len} bytes for {kwargs['width']}x{kwargs['height']} image")
                    return
                if expected_len > len(image_buffer):
                    image_buffer = bytearray(expected_len)
                m = memoryview(image_buffer)[:expected_len]
                reader.readinto(m)
                kwargs['pixels'] = np.frombuffer(m,dtype=np.float32)
                queue.append((action, kwargs))
            elif action == Action.EXCEPTION:
                queue.append((action, kwargs))
                if kwargs['fatal']:
                    return
            else:
                queue_exception_msg(f"Internal error, unexpected action id: {action}")
                return

def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stdout = open(os.devnull, 'w') # prevent stable diffusion logs from breaking ipc
    stderr = sys.stderr

    def send_action(action, *, payload = None, **kwargs):
        """Sends action messages to frontend.

        Arguments:
        * action -- Action enum or int
        * payload -- Bytes-like value that is not suitable for json
        * **kwargs -- json serializable key-value pairs used for callback function arguments
        """
        if Action(action) == Action.UNKNOWN:
            raise ValueError(f"Internal error, invalid Action: {action}")
        kwargs_len = payload_len = b'\x00'*8
        if kwargs:
            kwargs = bytes(json.dumps(kwargs), encoding='utf-8')
            kwargs_len = len(kwargs).to_bytes(len(kwargs_len), sys.byteorder, signed=False)
        if payload is not None:
            payload = memoryview(payload)
            payload_len = len(payload).to_bytes(len(payload_len), sys.byteorder, signed=False)
        # keep all checks before writing so ipc doesn't get broken actions

        def split_write(mv):
            for i in range(0,len(mv),1024*64):
                stdout.write(bytes(mv[i:i+1024*64])) # writing fails when using memoryview slices directly, wrap byte() first
            # stdout.write(bytes(mv)) # writing full image has caused the subprocess to crash without raising any exception, safer not to use

        stdout.write(action.to_bytes(ACTION_BYTE_LENGTH, sys.byteorder, signed=False))
        stdout.write(kwargs_len)
        if kwargs:
            split_write(kwargs)
        stdout.write(payload_len)
        if payload:
            split_write(payload)
        stdout.flush()

    def send_info(msg):
        """Sends information to be shown to the user before generation begins."""
        send_action(Action.INFO, msg=msg)

    def send_exception(fatal = True, msg: str = None, trace: str = None):
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
        send_action(Action.EXCEPTION, fatal=fatal, msg=msg, trace=trace)

    try:
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

        from stable_diffusion.ldm.generate import Generate
        from omegaconf import OmegaConf
        from PIL import Image, ImageOps
        from io import StringIO
    except ModuleNotFoundError as e:
        min_files = 10 # bump this up if more files get added to .python_dependencies in source
                       # don't set too high so it can still pass info on individual missing modules
        if not os.path.exists(".python_dependencies") or len(os.listdir()) < min_files:
            send_exception(msg=MISSING_DEPENDENCIES_ERROR)
        else:
            send_exception()
        return
    except:
        send_exception()
        return

    models_config  = absolute_path('stable_diffusion/configs/models.yaml')
    model   = 'stable-diffusion-1.4'

    models  = OmegaConf.load(models_config)
    config  = absolute_path('stable_diffusion/' + models[model].config)
    weights = absolute_path('stable_diffusion/' + models[model].weights)

    byte_to_normalized = 1.0 / 255.0
    def image_to_bytes(image):
        return (np.asarray(ImageOps.flip(image).convert('RGBA'),dtype=np.float32) * byte_to_normalized).tobytes()

    def image_writer(image, seed, upscaled=False, first_seed=None):
        # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
        if not upscaled:
            send_action(Action.IMAGE, payload=image_to_bytes(image), seed=seed, width=image.width, height=image.height)
    
    step = 0
    def view_step(samples):
        nonlocal step
        if args['show_steps']:
            image = generator._sample_to_image(samples)
            send_action(Action.STEP_IMAGE, payload=image_to_bytes(image), step=step, width=image.width, height=image.height)
        else:
            send_action(Action.STEP_NO_SHOW, step=step)
        step += 1

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
            send_info(f"Downloading {model_name} (0%)")

        def update_decorator(original):
            def update(self, n=1):
                result = original(self, n)
                nonlocal current_model_name
                frac = self.n / self.total
                percentage = int(frac * 100)
                if self.n - self.last_print_n >= self.miniters:
                    send_info(f"Downloading {current_model_name} ({percentage}%)")
                return result
            return update
        old_update = tqdm.update
        tqdm.update = update_decorator(tqdm.update)

        import warnings
        import transformers
        transformers.logging.set_verbosity_error()

        start_preloading("BERT tokenizer")
        transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

        send_info("Preloading `kornia` requirements")
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

    generator = None
    while True:
        intent = Intent.from_bytes(stdin.read(ACTION_BYTE_LENGTH), sys.byteorder, signed=False)
        if intent == Intent.PROMPT_TO_IMAGE:
            json_len = int.from_bytes(stdin.read(8),sys.byteorder,signed=False)
            if json_len == 0:
                return # stdin closed
            args = json.loads(stdin.read(json_len))

            if generator is None or (generator.full_precision != args['full_precision'] and sys.platform != 'darwin'):
                send_info("Loading Model")
                try:
                    generator = Generate(
                        conf=models_config,
                        model=model,
                        # These args are deprecated, but we need them to specify an absolute path to the weights.
                        weights=weights,
                        config=config,
                        full_precision=args['full_precision']
                    )
                    generator.free_gpu_mem = False # Not sure what this is for, and why it isn't a flag but read from Args()?
                    generator.load_model()
                except:
                    send_exception()
                    return
            send_info("Starting")
            
            try:
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
                            send_exception(False, f"{low_ram[1]}{' or disable full precision' if args['full_precision'] else ''}{low_ram[2]}", s)
                        elif s.find("CUDA out of memory. Tried to allocate") != -1:
                            send_exception(False, f"Not enough memory, use lower resolution{' or disable full precision' if args['full_precision'] else ''}", s)
                        else:
                            send_exception(True, msg=None, trace=s) # consider all unknown exceptions to be fatal so the generator process is fully restarted next time
                            return
            except:
                send_exception()
                return
            finally:
                sys.stderr = stderr
        elif intent == Intent.UPSCALE:
            tmp_stderr = sys.stderr = StringIO()
            json_len = int.from_bytes(stdin.read(8),sys.byteorder,signed=False)
            if json_len == 0:
                return # stdin closed
            args = json.loads(stdin.read(json_len))
            send_info("Starting")
            try:
                from absolute_path import REAL_ESRGAN_WEIGHTS_PATH
                import cv2
                from realesrgan import RealESRGANer
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                # image = Image.open(args['input'])
                image = cv2.imread(args['input'], cv2.IMREAD_UNCHANGED)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                netscale = 4
                send_info("Loading Upsampler")
                upsampler = RealESRGANer(
                    scale=netscale,
                    model_path=REAL_ESRGAN_WEIGHTS_PATH,
                    dni_weight=1,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    gpu_id=None
                )
                send_info("Enhancing")
                output, _ = upsampler.enhance(image, outscale=4)
                send_info("Upscaled")
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output = Image.fromarray(output)
                output_path = tempfile.NamedTemporaryFile().name
                output.save(output_path, format="png")
                send_action(Action.IMAGE_PATH, output_path=output_path)
            except:
                send_exception()
                return
            finally:
                sys.stderr = stderr
        else:
            send_exception(True, f"Unknown intent {intent} sent to process. Expected one of {Intent._member_names_}.", "")

if __name__ == "__main__":
    main()