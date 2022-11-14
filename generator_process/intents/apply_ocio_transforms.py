import math
import sys
import numpy as np
from ..registrar import registrar
from ..intent import Intent
from ..action import Action

@registrar.generator_intent
def apply_ocio_transforms(self, args, image_callback, exception_callback):
    self.send_intent(Intent.APPLY_OCIO_TRANSFORMS, **args)

    queue = self.queue
    callbacks = {
        Action.INFO: lambda *args, **kwargs: None,
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

@registrar.intent_backend(Intent.APPLY_OCIO_TRANSFORMS)
def _apply_ocio_transforms(self):
    args = yield
    import PyOpenColorIO as OCIO
    from multiprocessing.shared_memory import SharedMemory
    while True:
        ocio_config = OCIO.Config.CreateFromFile(args['config_path'])
        
        # A reimplementation of `OCIOImpl::createDisplayProcessor` from the Blender source.
        # https://github.com/dfelinto/blender/blob/87a0770bb969ce37d9a41a04c1658ea09c63933a/intern/opencolorio/ocio_impl.cc#L643
        def create_display_processor(
            config,
            input_colorspace,
            view,
            display,
            look,
            scale, # Exposure
            exponent, # Gamma
            inverse=False
        ):
            group = OCIO.GroupTransform()

            # Exposure
            if scale != 1:
                # Always apply exposure in scene linear.
                color_space_transform = OCIO.ColorSpaceTransform()
                color_space_transform.setSrc(input_colorspace)
                color_space_transform.setDst(OCIO.ROLE_SCENE_LINEAR)
                group.appendTransform(color_space_transform)

                # Make further transforms aware of the color space change
                input_colorspace = OCIO.ROLE_SCENE_LINEAR

                # Apply scale
                matrix_transform = OCIO.MatrixTransform([scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 1.0])
                group.appendTransform(matrix_transform)
            
            # Add look transform
            use_look = look is not None and len(look) > 0
            if use_look:
                look_output = config.getLook(look).getProcessSpace()
                if look_output is not None and len(look_output) > 0:
                    look_transform = OCIO.LookTransform()
                    look_transform.setSrc(input_colorspace)
                    look_transform.setDst(look_output)
                    look_transform.setLooks(look)
                    group.appendTransform(look_transform)
                    # Make further transforms aware of the color space change.
                    input_colorspace = look_output
                else:
                    # For empty looks, no output color space is returned.
                    use_look = False
            
            # Add view and display transform
            display_view_transform = OCIO.DisplayViewTransform()
            display_view_transform.setSrc(input_colorspace)
            display_view_transform.setLooksBypass(True)
            display_view_transform.setView(view)
            display_view_transform.setDisplay(display)
            group.appendTransform(display_view_transform)

            # Gamma
            if exponent != 1:
                exponent_transform = OCIO.ExponentTransform([exponent, exponent, exponent, 1.0])
                group.appendTransform(exponent_transform)
            
            # Create processor from transform. This is the moment were OCIO validates
            # the entire transform, no need to check for the validity of inputs above.
            try:
                if inverse:
                    group.setDirection(OCIO.TransformDirection.TRANSFORM_DIR_INVERSE)
                processor = config.getProcessor(group)
                if processor is not None:
                    return processor
            except Exception as e:
                self.send_exception(True, msg=str(e), trace="")
            
            return None
        
        # Exposure and gamma transformations derived from Blender source:
        # https://github.com/dfelinto/blender/blob/87a0770bb969ce37d9a41a04c1658ea09c63933a/source/blender/imbuf/intern/colormanagement.c#L825
        scale = math.pow(2, args['exposure'])
        exponent = 1 if args['gamma'] == 1 else (1 / (args['gamma'] if args['gamma'] > sys.float_info.epsilon else sys.float_info.epsilon))
        processor = create_display_processor(ocio_config, OCIO.ROLE_SCENE_LINEAR, args['view_transform'], args['display_device'], args['look'] if args['look'] != 'None' else None, scale, exponent, args['inverse'])

        shared_memory = SharedMemory(args['name'])
        input_image = np.frombuffer(shared_memory.buf, dtype=np.float32)
        processor.getDefaultCPUProcessor().applyRGBA(input_image)
        shared_memory.buf[:] = input_image.tobytes()
        self.send_action(Action.IMAGE, shared_memory_name=args['name'], seed=args['name'], width=0, height=0)
        del input_image
        shared_memory.close()
        args = yield