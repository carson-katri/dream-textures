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
    
    def prompt2image(self, args, step_callback, image_callback):
        b = bytes(json.dumps(args), "utf-8")
        self.process.stdin.write(len(b).to_bytes(8,sys.byteorder,signed=False))
        self.process.stdin.write(b)
        self.process.stdin.flush()
        if image_callback:
            for i in range(0,args['iterations']):
                image_callback(*self.get_image())

    def get_image(self):
        seed = int.from_bytes(self.process.stdout.read(4),sys.byteorder,signed=False)
        width = int.from_bytes(self.process.stdout.read(4),sys.byteorder,signed=False)
        height = int.from_bytes(self.process.stdout.read(4),sys.byteorder,signed=False)
        image = np.frombuffer(self.process.stdout.read(width*height*4*4),dtype=np.float32)
        return (seed, width, height, image)



def main():
    from absolute_path import absolute_path
    from stable_diffusion.ldm.generate import Generate
    from omegaconf import OmegaConf
    from PIL import ImageOps

    # Support Apple Silicon GPUs as much as possible.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    models_config  = absolute_path('stable_diffusion/configs/models.yaml')
    model   = 'stable-diffusion-1.4'

    models  = OmegaConf.load(models_config)
    config  = absolute_path('stable_diffusion/' + models[model].config)
    weights = absolute_path('stable_diffusion/' + models[model].weights)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stdout = open(os.devnull, 'w') # stable diffusion logs won't break get_image() now

    byte_to_normalized = 1.0 / 255.0
    def image_writer(image, seed, upscaled=False):
        # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
        if not upscaled:
            stdout.write(seed.to_bytes(4,sys.byteorder,signed=False))
            stdout.write(image.width.to_bytes(4,sys.byteorder,signed=False))
            stdout.write(image.height.to_bytes(4,sys.byteorder,signed=False))
            b = (np.asarray(ImageOps.flip(image).convert('RGBA'),dtype=np.float32) * byte_to_normalized).tobytes()
            for i in range(0,len(b),1024*64):
                stdout.write(b[i:i+1024*64])
            # stdout.write(b) # writing it all at once was causing this to exit without error
            stdout.flush()
            return

    generator = None
    while True:
        json_len = int.from_bytes(stdin.read(8),sys.byteorder,signed=False)
        if json_len == 0:
            return # stdin closed
        args = json.loads(stdin.read(json_len))

        if generator is None or generator.full_precision != args['full_precision']:
            generator = Generate(
                conf=models_config,
                model=model,
                # These args are deprecated, but we need them to specify an absolute path to the weights.
                weights=weights,
                config=config,
                full_precision=args['full_precision']
            )
            generator.load_model()
        
        generator.prompt2image(
            # a function or method that will be called each step
            step_callback=None,
            # a function or method that will be called each time an image is generated
            image_callback=image_writer,
            **args
        )

if __name__ == "__main__":
    main()