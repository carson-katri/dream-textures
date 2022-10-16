from ..block_in_use import block_in_use
from ..action import Action
from ..intent import Intent
from ..registrar import registrar

@registrar.generator_intent
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

@registrar.intent_backend(Intent.UPSCALE)
def upscale(self):
    args = yield
    self.send_info("Starting")
    from absolute_path import REAL_ESRGAN_WEIGHTS_PATH
    import cv2
    from PIL import Image
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from torch import nn
    while True:
        image = cv2.imread(args['input'], cv2.IMREAD_UNCHANGED)
        real_esrgan_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        if args['seamless']:
            for m in real_esrgan_model.body:
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m.padding_mode = 'circular'
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
        self.send_info("Converting Result")
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = Image.fromarray(output)
        self.send_action(Action.IMAGE, shared_memory_name=self.share_image_memory(output), seed=args['name'], width=output.width, height=output.height)
        args = yield