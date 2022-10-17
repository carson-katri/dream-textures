from ..intent import Intent
from ..registrar import registrar

@registrar.generator_intent
def send_stop(self, stop_intent):
    self.send_intent(Intent.STOP, stop_intent=stop_intent)

# The Intent backend is implemented as a special case.