from ..block_in_use import block_in_use
from ..action import Action
from ..intent import Intent
from ..registrar import BackendTarget, registrar
import sys
import os

@registrar.generator_intent
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

@registrar.intent_backend(Intent.PROMPT_TO_IMAGE, BackendTarget.LOCAL)
def prompt_to_image(self):
    args = yield
    self.send_info("Importing Dependencies")
    from absolute_path import absolute_path
    from stable_diffusion.ldm.generate import Generate
    from stable_diffusion.ldm.invoke.devices import choose_precision
    from io import StringIO
    
    models_config  = absolute_path('weights/config.yml')
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
        import urllib
        import ssl
        urlopen = urllib.request.urlopen
        def urlopen_decroator(func):
            def urlopen(*args, **kwargs):
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                return func(*args, **kwargs, context=ssl_context)
            return urlopen
        urllib.request.urlopen = urlopen_decroator(urllib.request.urlopen)

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

        start_preloading("CLIP Segmentation")
        from absolute_path import CLIPSEG_WEIGHTS_PATH
        from clipseg_models.clipseg import CLIPDensePredT
        CLIPDensePredT(version='ViT-B/16', reduce_dim=64)

        tqdm.update = old_update
        urllib.request.urlopen = urlopen
    
    from transformers.utils.hub import TRANSFORMERS_CACHE
    model_paths = {'bert-base-uncased', 'openai--clip-vit-large-patch14'}
    if any(not os.path.isdir(os.path.join(TRANSFORMERS_CACHE, f'models--{path}')) for path in model_paths) or not os.path.exists(os.path.join(os.path.expanduser("~/.cache/clip"), 'ViT-B-16.pt')):
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
                    precision=args['precision']
                )
                generator.free_gpu_mem = False # Not sure what this is for, and why it isn't a flag but read from Args()?
                generator.load_model()
                self.check_stop()
            self.send_info("Starting")
            
            tmp_stderr = sys.stderr = StringIO() # prompt2image writes exceptions straight to stderr, intercepting
            prompt_list = args['prompt'] if isinstance(args['prompt'], list) else [args['prompt']]
            for prompt in prompt_list:
                generator_args = args.copy()
                generator_args['prompt'] = prompt
                generator_args['seamless_axes'] = list(generator_args['seamless_axes'])
                if args['init_img_action'] == 'inpaint' and args['inpaint_mask_src'] == 'prompt':
                    generator_args['text_mask'] = (generator_args['text_mask'], generator_args['text_mask_confidence'])
                else:
                    generator_args['text_mask'] = None
                if args['use_init_img'] and args['init_img_action'] == 'outpaint':
                    args['fit'] = False
                    # Extend the image in the specified directions
                    from PIL import Image, ImageFilter
                    init_img = Image.open(args['init_img'])
                    extended_size = (init_img.size[0] + args['outpaint_left'] + args['outpaint_right'], init_img.size[1] + args['outpaint_top'] + args['outpaint_bottom'])
                    extended_img = Image.new('RGBA', extended_size, (0, 0, 0, 0))
                    blurred_fill = init_img.resize(extended_size).filter(filter=ImageFilter.GaussianBlur(radius=args['outpaint_blend']))
                    blurred_fill.putalpha(0)
                    extended_img.paste(blurred_fill, (0, 0))
                    extended_img.paste(init_img, (args['outpaint_left'], args['outpaint_top']))
                    extended_img.save(generator_args['init_img'], 'png')
                    generator_args['width'] = extended_size[0]
                    generator_args['height'] = extended_size[1]
                generator.prompt2image(
                    # a function or method that will be called each step
                    step_callback=view_step,
                    # a function or method that will be called each time an image is generated
                    image_callback=image_writer,
                    **generator_args
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

@registrar.intent_backend(Intent.PROMPT_TO_IMAGE, BackendTarget.STABILITY_SDK)
def prompt_to_image_stability_sdk(self):
    args = yield
    self.send_info("Importing Dependencies")

    from stability_sdk import client, interfaces
    from PIL import Image
    import io
    import random
    from multiprocessing.shared_memory import SharedMemory

    # Some of these names are abbreviated.
    algorithms = client.algorithms.copy()
    algorithms['k_euler_a'] = algorithms['k_euler_ancestral']
    algorithms['k_dpm_2_a'] = algorithms['k_dpm_2_ancestral']

    stability_inference = client.StabilityInference(key="KEY")

    def image_writer(image, seed, upscaled=False, first_seed=None):
        # Only use the non-upscaled texture, as upscaling is a separate step in this addon.
        if not upscaled:
            self.send_action(Action.IMAGE, shared_memory_name=self.share_image_memory(image), seed=seed, width=image.width, height=image.height)

    while True:
        self.check_stop()

        self.send_info("Generating...")
        
        seed = random.randrange(0, 4294967295) if args['seed'] is None else args['seed']
        shared_init_img = None
        # if args['init_img_shared_memory'] is not None:
        #     init_img_memory = SharedMemory(args['init_img_shared_memory'])
        #     shared_init_img = Image.frombytes('RGBA', (args['init_img_shared_memory_width'], args['init_img_shared_memory_height']), init_img_memory.buf.tobytes())
        #     shared_init_img.save('/Users/carsonkatri/Documents/Art/Add-ons/Custom/Blender/dream_textures/test.png')
        #     shared_init_img = shared_init_img.resize((512, round(((shared_init_img.height / shared_init_img.width) * 512) / 64)*64))
        #     init_img_memory.close()
        #     shared_init_img.save('/Users/carsonkatri/Documents/Art/Add-ons/Custom/Blender/dream_textures/test_scaled.png')
        answers = stability_inference.generate(
            prompt=args['prompt'],
            init_image=shared_init_img if shared_init_img is not None else (Image.open(args['init_img']) if args['init_img'] is not None else None),
            # mask_image: Optional[Image.Image] = None,
            width=shared_init_img.width if shared_init_img is not None else args['width'],
            height=shared_init_img.height if shared_init_img is not None else args['height'],
            start_schedule=1.0 * args['strength'],
            end_schedule=0.01,
            cfg_scale=args['cfg_scale'],
            sampler=algorithms[args['sampler_name']],
            steps=args['steps'],
            seed=seed,
            samples=args['iterations'],
            # safety: bool = True,
            # classifiers: Optional[generation.ClassifierParameters] = None,
            # guidance_preset: generation.GuidancePreset = generation.GUIDANCE_PRESET_NONE,
            # guidance_cuts: int = 0,
            # guidance_strength: Optional[float] = None,
            # guidance_prompt: Union[str, generation.Prompt] = None,
            # guidance_models: List[str] = None,
        )

        for answer in answers:
            for artifact in answer.artifacts:
                if artifact.finish_reason == interfaces.gooseai.generation.generation_pb2.FILTER:
                    self.send_exception(False, "Your request activated Dream Studio's safety filter. Please modify your prompt and try again.")
                if artifact.type == interfaces.gooseai.generation.generation_pb2.ARTIFACT_IMAGE:
                    response = Image.open(io.BytesIO(artifact.binary))
                    image_writer(response, seed)
        
        self.send_info("Done")
        args = yield