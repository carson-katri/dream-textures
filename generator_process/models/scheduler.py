import enum

class Scheduler(enum.Enum):
    DDIM = "DDIM"
    DDPM = "DDPM"
    DEIS_MULTISTEP = "DEIS Multistep"
    DPM_SOLVER_MULTISTEP = "DPM Solver Multistep"
    DPM_SOLVER_SINGLESTEP = "DPM Solver Singlestep"
    EULER_DISCRETE = "Euler Discrete"
    EULER_ANCESTRAL_DISCRETE = "Euler Ancestral Discrete"
    HEUN_DISCRETE = "Heun Discrete"
    KDPM2_DISCRETE = "KDPM2 Discrete" # Non-functional on mps
    KDPM2_ANCESTRAL_DISCRETE = "KDPM2 Ancestral Discrete"
    LMS_DISCRETE = "LMS Discrete"
    PNDM = "PNDM"

    def create(self, pipeline, pretrained):
        import diffusers
        def scheduler_class():
            match self:
                case Scheduler.DDIM:
                    return diffusers.schedulers.DDIMScheduler
                case Scheduler.DDPM:
                    return diffusers.schedulers.DDPMScheduler
                case Scheduler.DEIS_MULTISTEP:
                    return diffusers.schedulers.DEISMultistepScheduler
                case Scheduler.DPM_SOLVER_MULTISTEP:
                    return diffusers.schedulers.DPMSolverMultistepScheduler
                case Scheduler.DPM_SOLVER_SINGLESTEP:
                    return diffusers.schedulers.DPMSolverSinglestepScheduler
                case Scheduler.EULER_DISCRETE:
                    return diffusers.schedulers.EulerDiscreteScheduler
                case Scheduler.EULER_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.EulerAncestralDiscreteScheduler
                case Scheduler.HEUN_DISCRETE:
                    return diffusers.schedulers.HeunDiscreteScheduler
                case Scheduler.KDPM2_DISCRETE:
                    return diffusers.schedulers.KDPM2DiscreteScheduler
                case Scheduler.KDPM2_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.KDPM2AncestralDiscreteScheduler
                case Scheduler.LMS_DISCRETE:
                    return diffusers.schedulers.LMSDiscreteScheduler
                case Scheduler.PNDM:
                    return diffusers.schedulers.PNDMScheduler
        if pretrained is not None:
            return scheduler_class().from_pretrained(pretrained['model_path'], subfolder=pretrained['subfolder'])
        else:
            return scheduler_class().from_config(pipeline.scheduler.config)