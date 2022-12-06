import os
from ..absolute_path import absolute_path
from .intent import Intent
from enum import IntEnum

class BackendTarget(IntEnum):
    """Which generator backend to use"""
    LOCAL = 0
    STABILITY_SDK = 1

    @staticmethod
    def local_available():
        return os.path.exists(absolute_path(".python_dependencies/diffusers"))

    def __str__(self):
        return self.name
    
    def init_img_actions(self):
        match self:
            case BackendTarget.LOCAL:
                return ['modify', 'inpaint', 'outpaint']
            case BackendTarget.STABILITY_SDK:
                return ['modify', 'inpaint']
    
    def inpaint_mask_sources(self):
        match self:
            case BackendTarget.LOCAL:
                return ['alpha', 'prompt']
            case BackendTarget.STABILITY_SDK:
                return ['alpha']
    
    def color_correction(self):
        match self:
            case BackendTarget.LOCAL:
                return True
            case BackendTarget.STABILITY_SDK:
                return False
    
    def negative_prompts(self):
        match self:
            case BackendTarget.LOCAL:
                return True
            case BackendTarget.STABILITY_SDK:
                return False
    
    def seamless(self):
        match self:
            case BackendTarget.LOCAL:
                return True
            case BackendTarget.STABILITY_SDK:
                return False
    
    def upscaling(self):
        match self:
            case BackendTarget.LOCAL:
                return True
            case BackendTarget.STABILITY_SDK:
                return False

class _GeneratorIntent:
    def __init__(self, func):
        self.name = func.__name__
        self.func = func

class _IntentBackend:
    def __init__(self, intent: Intent, backend: BackendTarget | None, func):
        self.intent = intent
        self.backend = backend
        self.func = func

class _IntentRegistrar:
    def __init__(self):
        self._generator_intents: list[_GeneratorIntent] = []
        self._intent_backends: list[_IntentBackend] = []

    def generator_intent(self, func):
        '''
        Registers an intent as a function on the `GeneratorProcess` class.
        '''
        intent = _GeneratorIntent(func)
        self._generator_intents.append(intent)
        return intent

    def intent_backend(self, intent: Intent, backend_target: BackendTarget | None = None):
        def decorator(func):
            backend = _IntentBackend(intent, backend_target, func)
            self._intent_backends.append(backend)
            return backend
        return decorator

registrar = _IntentRegistrar()
