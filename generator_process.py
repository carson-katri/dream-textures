import json
import subprocess
import sys
import os
import numpy as np

from enum import IntEnum

class Action(IntEnum):
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
    
    def kill(self):
        self.process.kill()
    
    def prompt2image(self, args, step_callback, image_callback, info_callback, exception_callback):
        stdin = self.process.stdin
        stdout = self.process.stdout
        b = bytes(json.dumps(args), 'utf-8')
        stdin.write(len(b).to_bytes(8,sys.byteorder,signed=False))
        stdin.write(b)
        stdin.flush()

        def readStr():
            return str(stdout.read(readUInt(4)), encoding='utf-8')

        def readUInt(length):
            return int.from_bytes(stdout.read(length),sys.byteorder,signed=False)

        for i in range(0,args['iterations']):
            while True:
                action = readUInt(1)
                if action == Action.CLOSED:
                    exception_callback(True, "Process closed unexpectedly")
                    return
                elif action == Action.INFO:
                    info_callback(readStr())
                elif action == Action.IMAGE or action == Action.STEP_IMAGE:
                    seed = readUInt(4)
                    width = readUInt(4)
                    height = readUInt(4)
                    image = np.frombuffer(stdout.read(width*height*4*4),dtype=np.float32)
                    if action == Action.IMAGE:
                        image_callback(seed, width, height, image, False)
                        break
                    else:
                        step_callback(seed, width, height, image) # seed is step number in this case
                elif action == Action.STEP_NO_SHOW:
                    step_callback(readUInt(4))
                elif action == Action.EXCEPTION:
                    exception_callback(readUInt(1) != 0, readStr())
                    return
                else:
                    exception_callback(True, f"Unexpected action id: {action}")



def main():
    from absolute_path import absolute_path
    # Support Apple Silicon GPUs as much as possible.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    sys.path.append(absolute_path("stable_diffusion/"))
    sys.path.append(absolute_path("stable_diffusion/src/clip"))
    sys.path.append(absolute_path("stable_diffusion/src/k-diffusion"))
    sys.path.append(absolute_path("stable_diffusion/src/taming-transformers"))
    from stable_diffusion.ldm.generate import Generate
    from omegaconf import OmegaConf
    from PIL import ImageOps
    from io import StringIO

    models_config  = absolute_path('stable_diffusion/configs/models.yaml')
    model   = 'stable-diffusion-1.4'

    models  = OmegaConf.load(models_config)
    config  = absolute_path('stable_diffusion/' + models[model].config)
    weights = absolute_path('stable_diffusion/' + models[model].weights)

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

    byte_to_normalized = 1.0 / 255.0
    def write_pixels(image):
        writeUInt(4,image.width)
        writeUInt(4,image.height)
        b = (np.asarray(ImageOps.flip(image).convert('RGBA'),dtype=np.float32) * byte_to_normalized).tobytes()
        for i in range(0,len(b),1024*64):
            stdout.write(b[i:i+1024*64])
        # stdout.write(b) # writing it all at once was causing this to exit without error

    def image_writer(image, seed, upscaled=False):
        # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
        if not upscaled:
            writeUInt(1,Action.IMAGE)
            writeUInt(4,seed)
            write_pixels(image)
            stdout.flush()
    
    def view_step(samples, step):
        if args['show_steps']:
            writeUInt(1,Action.STEP_IMAGE)
            writeUInt(4,step)
            write_pixels(generator._sample_to_image(samples))
        else:
            writeUInt(1,Action.STEP_NO_SHOW)
            writeUInt(4,step)
        stdout.flush()

    generator = None
    while True:
        json_len = int.from_bytes(stdin.read(8),sys.byteorder,signed=False)
        if json_len == 0:
            return # stdin closed
        args = json.loads(stdin.read(json_len))

        if generator is None or generator.full_precision != args['full_precision']:
            writeInfo("Initializing Generator")
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
                writeException(True, str(e))
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
                    else:
                        writeException(True, s) # consider all unknown exceptions to be fatal so the generator process is fully restarted next time
        except Exception as e:
            writeException(True, str(e))
            return
        finally:
            sys.stderr = stderr

if __name__ == "__main__":
    main()