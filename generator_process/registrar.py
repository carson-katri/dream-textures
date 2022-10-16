from .intent import Intent

class _GeneratorIntent:
    def __init__(self, func):
        self.name = func.__name__
        self.func = func

class _IntentBackend:
    def __init__(self, intent: Intent, func):
        self.intent = intent
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

    def intent_backend(self, intent: Intent):
        def decorator(func):
            backend = _IntentBackend(intent, func)
            self._intent_backends.append(backend)
            return backend
        return decorator

registrar = _IntentRegistrar()