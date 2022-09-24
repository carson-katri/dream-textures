import json
import subprocess
import sys
import os
import numpy as np

class GeneratorProcess():
    def __init__(self):
        self.process = subprocess.Popen([sys.executable,'generator_process.py'],cwd=os.path.dirname(os.path.realpath(__file__)),stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    
    def kill(self):
        self.process.kill()
    
    def prompt2image(self, args, step_callback, image_callback, info_callback):
        stdin = self.process.stdin
        stdout = self.process.stdout
        b = bytes(json.dumps(args), 'utf-8')
        stdin.write(len(b).to_bytes(8,sys.byteorder,signed=False))
        stdin.write(b)
        stdin.flush()

        def readUInt(length):
            return int.from_bytes(stdout.read(length),sys.byteorder,signed=False)

        for i in range(0,args['iterations']):
            while True:
                action = readUInt(1)
                if action == 0:
                    return # stdout closed
                elif action == 1:
                    info_callback(str(stdout.read(readUInt(4)), encoding='utf-8'))
                elif action == 2 or action == 3:
                    seed = readUInt(4)
                    width = readUInt(4)
                    height = readUInt(4)
                    image = np.frombuffer(stdout.read(width*height*4*4),dtype=np.float32)
                    if action == 2:
                        image_callback(seed, width, height, image, False)
                        break
                    else:
                        step_callback(seed, width, height, image) # seed is step number in this case
                else:
                    raise RuntimeError(f"Unexpected action id {action}")



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

    models_config  = absolute_path('stable_diffusion/configs/models.yaml')
    model   = 'stable-diffusion-1.4'

    models  = OmegaConf.load(models_config)
    config  = absolute_path('stable_diffusion/' + models[model].config)
    weights = absolute_path('stable_diffusion/' + models[model].weights)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stdout = open(os.devnull, 'w') # prevent stable diffusion logs from breaking ipc

    def writeInfo(msg):
        writeUInt(1,1)
        b = bytes(msg,encoding='utf-8')
        writeUInt(4,len(b))
        stdout.write(b)
        stdout.flush()

    def writeUInt(length, value):
        stdout.write(value.to_bytes(length,sys.byteorder,signed=False))

    byte_to_normalized = 1.0 / 255.0
    def write_pixels(image):
        writeUInt(4,image.width)
        writeUInt(4,image.height)
        b = (np.asarray(ImageOps.flip(image).convert('RGBA'),dtype=np.float32) * byte_to_normalized).tobytes()
        for i in range(0,len(b),1024*64):
            stdout.write(b[i:i+1024*64])
        # stdout.write(b) # writing it all at once was causing this to exit without error
        stdout.flush()

    def image_writer(image, seed, upscaled=False):
        # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
        if not upscaled:
            writeUInt(1,2)
            writeUInt(4,seed)
            write_pixels(image)
    
    def view_step(samples, step):
        writeUInt(1,3)
        writeUInt(4,step)
        write_pixels(generator._sample_to_image(samples))

    generator = None
    while True:
        json_len = int.from_bytes(stdin.read(8),sys.byteorder,signed=False)
        if json_len == 0:
            return # stdin closed
        args = json.loads(stdin.read(json_len))

        if generator is None or generator.full_precision != args['full_precision']:
            writeInfo("Initializing Generator")
            generator = Generate(
                conf=models_config,
                model=model,
                # These args are deprecated, but we need them to specify an absolute path to the weights.
                weights=weights,
                config=config,
                full_precision=args['full_precision']
            )
            generator.load_model()
        writeInfo("Starting")
        generator.prompt2image(
            # a function or method that will be called each step
            step_callback=view_step if args['show_steps'] else None,
            # a function or method that will be called each time an image is generated
            image_callback=image_writer,
            **args
        )

if __name__ == "__main__":
    main()