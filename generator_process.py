import json
import subprocess
import sys
import os
import threading
import site
import numpy as np
from enum import IntEnum as Lawsuit, auto

MISSING_DEPENDENCIES_ERROR = "Python dependencies are missing. Click Download Latest Release to fix."

# IPC message types from subprocess
class Action(Lawsuit): # can't help myself
    UNKNOWN = -1
    CLOSED = 0
    INFO = 1
    IMAGE = 2
    STEP_IMAGE = 3
    STEP_NO_SHOW = 4
    EXCEPTION = 5

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

class GeneratorProcess():
    def __init__(self):
        self.process = subprocess.Popen([sys.executable,'generator_process.py'],cwd=os.path.dirname(os.path.realpath(__file__)),stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        self.reader = self.process.stdout
        self.queue = []
        self.args = None
        self.killed = False
        self.thread = threading.Thread(target=self._run,daemon=True,name="BackgroundReader")
        self.thread.start()
    
    def kill(self):
        self.killed = True
        self.process.kill()
    
    def prompt2image(self, args, step_callback, image_callback, info_callback, exception_callback):
        self.args = args
        stdin = self.process.stdin
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
                callbacks[action](*tup[1:])
                if action == Action.IMAGE:
                    break
                elif action == Action.EXCEPTION:
                    return
    
    def _run(self):
        reader = self.reader
        def readStr():
            return str(reader.read(readUInt(4)), encoding='utf-8')
        def readUInt(length):
            return int.from_bytes(reader.read(length),sys.byteorder,signed=False)

        queue = self.queue
        def queue_exception(fatal, err):
            queue.append((Action.EXCEPTION, fatal, err))

        image_buffer = bytearray(512*512*16)
        while not self.killed:
            action = readUInt(1)
            if action == Action.CLOSED:
                if not self.killed:
                    queue_exception(True, "Process closed unexpectedly")
                return
            elif action == Action.INFO:
                queue.append((action, readStr()))
            elif action == Action.IMAGE or action == Action.STEP_IMAGE:
                seed = readUInt(4)
                width = readUInt(4)
                height = readUInt(4)
                length = width*height*16
                import math
                w = math.floor(self.args['width']/64)*64 # stable diffusion rounds down internally
                h = math.floor(self.args['height']/64)*64
                if width != w or height != h:
                    queue_exception(True, f"Internal error, received image of wrong resolution {width}x{height}, expected {w}x{h}")
                    return
                if length > len(image_buffer):
                    image_buffer = bytearray(length)
                m = memoryview(image_buffer)[:length]
                reader.readinto(m)
                image = np.frombuffer(m,dtype=np.float32)
                queue.append((action, seed, width, height, image))
            elif action == Action.STEP_NO_SHOW:
                queue.append((action, readUInt(4)))
            elif action == Action.EXCEPTION:
                fatal = readUInt(1) != 0
                queue_exception(fatal, readStr())
                if fatal:
                    return
            else:
                queue_exception(True, f"Internal error, unexpected action id: {action}")
                return

def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stdout = open(os.devnull, 'w') # prevent stable diffusion logs from breaking ipc
    stderr = sys.stderr

    def writeUInt(length, value):
        stdout.write(value.to_bytes(length,sys.byteorder,signed=False))

    def writeStr(string):
        b = bytes(string,encoding='utf-8')
        writeUInt(4,len(b))
        stdout.write(b)

    def writeInfo(msg):
        writeUInt(1,Action.INFO)
        writeStr(msg)
        stdout.flush()

    def writeException(fatal, e):
        writeUInt(1,Action.EXCEPTION)
        writeUInt(1,1 if fatal else 0)
        writeStr(e)
        stdout.flush()

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
        from PIL import ImageOps
        from io import StringIO
    except ModuleNotFoundError as e:
        min_files = 10 # bump this up if more files get added to .python_dependencies in source
                       # don't set too high so it can still pass info on individual missing modules
        if not os.path.exists(".python_dependencies") or len(os.listdir()) < min_files:
            e = MISSING_DEPENDENCIES_ERROR
        else:
            e = repr(e)
        writeException(True, e)
        return
    except Exception as e:
        writeException(True, repr(e))
        return

    models_config  = absolute_path('stable_diffusion/configs/models.yaml')
    model   = 'stable-diffusion-1.4'

    models  = OmegaConf.load(models_config)
    config  = absolute_path('stable_diffusion/' + models[model].config)
    weights = absolute_path('stable_diffusion/' + models[model].weights)

    byte_to_normalized = 1.0 / 255.0
    def write_pixels(image):
        writeUInt(4,image.width)
        writeUInt(4,image.height)
        b = (np.asarray(ImageOps.flip(image).convert('RGBA'),dtype=np.float32) * byte_to_normalized).tobytes()
        for i in range(0,len(b),1024*64):
            stdout.write(b[i:i+1024*64])
            # stdout.write(memoryview(b)[i:i+1024*64]) # won't accept memoryview for some reason, writer thinks it needs serialized but fails
        # stdout.write(b) # writing full image has caused the subprocess to crash without raising any exception, safer not to use
        stdout.flush()

    def image_writer(image, seed, upscaled=False):
        # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
        if not upscaled:
            writeUInt(1,Action.IMAGE)
            writeUInt(4,seed)
            write_pixels(image)
            stdout.flush()
    
    def view_step(samples, step):
        if args['show_steps']:
            pixels = generator._sample_to_image(samples) # May run out of memory, keep before any writing
            writeUInt(1,Action.STEP_IMAGE)
            writeUInt(4,step)
            write_pixels(pixels)
        else:
            writeUInt(1,Action.STEP_NO_SHOW)
            writeUInt(4,step)
        stdout.flush()

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
            writeInfo(f"Downloading {model_name} (0%)")

        def update_decorator(original):
            def update(self, n=1):
                result = original(self, n)
                nonlocal current_model_name
                frac = self.n / self.total
                percentage = int(frac * 100)
                if self.n - self.last_print_n >= self.miniters:
                    writeInfo(f"Downloading {current_model_name} ({percentage}%)")
                return result
            return update
        old_update = tqdm.update
        tqdm.update = update_decorator(tqdm.update)

        import warnings
        import transformers
        transformers.logging.set_verbosity_error()

        start_preloading("BERT tokenizer")
        transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

        writeInfo("Preloading `kornia` requirements")
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
        json_len = int.from_bytes(stdin.read(8),sys.byteorder,signed=False)
        if json_len == 0:
            return # stdin closed
        args = json.loads(stdin.read(json_len))

        if generator is None or (generator.full_precision != args['full_precision'] and sys.platform != 'darwin'):
            writeInfo("Loading Model")
            try:
                generator = Generate(
                    conf=models_config,
                    model=model,
                    # These args are deprecated, but we need them to specify an absolute path to the weights.
                    weights=weights,
                    config=config,
                    full_precision=args['full_precision']
                )
                generator.load_model()
            except Exception as e:
                writeException(True, repr(e))
                return
        writeInfo("Starting")
        
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
                        writeException(False, f"{low_ram[1]}{' or disable full precision' if args['full_precision'] else ''}{low_ram[2]}")
                    elif s.find("CUDA out of memory. Tried to allocate") != -1:
                        writeException(False, f"Not enough memory, use lower resolution{' or disable full precision' if args['full_precision'] else ''}")
                    else:
                        writeException(True, s) # consider all unknown exceptions to be fatal so the generator process is fully restarted next time
                        return
        except Exception as e:
            writeException(True, repr(e))
            return
        finally:
            sys.stderr = stderr

if __name__ == "__main__":
    main()