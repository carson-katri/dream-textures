from .intent import Intent
from enum import IntEnum

class BackendTarget(IntEnum):
    """Which generator backend to use"""
    LOCAL = 0
    STABILITY_SDK = 1

    def __str__(self):
        return self.name

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