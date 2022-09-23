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
    
    def prompt2image(
            self,
            prompt,
            iterations       = None,
            steps            = None,
            seed             = None,
            cfg_scale        = None,
            ddim_eta         = None,
            width            = None,
            height           = None,
            sampler_name     = None,
            seamless         = False,
            init_img         = None,
            fit              = False,
            strength         = None,
            full_precision   = False,
            step_callback    = None,
            image_callback   = None):
        b = bytes(json.dumps({
            'prompt': prompt,
            'iterations': iterations,
            'steps': steps,
            'seed': seed,
            'cfg_scale': cfg_scale,
            'ddim_eta': ddim_eta,
            'width': width,
            'height': height,
            'sampler_name': sampler_name,
            'seamless': seamless,
            'init_img': init_img,
            'fit': fit,
            'strength': strength,
            'full_precision': full_precision
        }), "utf-8")
        self.process.stdin.write(len(b).to_bytes(8,sys.byteorder,signed=False))
        self.process.stdin.write(b)
        self.process.stdin.flush()
        if image_callback:
            for i in range(0,iterations or 1):
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
            # written = stdout.write(b) # writing it all at once was causing this to exit without error
            stdout.flush()
            return

    generator = None
    while True:
        args = json.loads(stdin.read(int.from_bytes(stdin.read(8),sys.byteorder,signed=False)))

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
            # prompt string (no default)
            prompt=args['prompt'],
            # iterations (1); image count=iterations
            iterations=args['iterations'],
            # refinement steps per iteration
            steps=args['steps'],
            # seed for random number generator
            seed=args['seed'],
            # width of image, in multiples of 64 (512)
            width=args['width'],
            # height of image, in multiples of 64 (512)
            height=args['height'],
            # how strongly the prompt influences the image (7.5) (must be >1)
            cfg_scale=args['cfg_scale'],
            # path to an initial image - its dimensions override width and height
            init_img=args['init_img'],

            # generate tileable/seamless textures
            seamless=args['seamless'],

            fit=args['fit'],
            # strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
            strength=args['strength'],
            # strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
            gfpgan_strength=0.0, # 0 disables upscaling, which is currently not supported by the addon.
            # image randomness (eta=0.0 means the same seed always produces the same image)
            ddim_eta=0.0,
            # a function or method that will be called each step
            step_callback=None,
            # a function or method that will be called each time an image is generated
            image_callback=image_writer,
            
            sampler_name=args['sampler_name']
        )

if __name__ == "__main__":
    main()