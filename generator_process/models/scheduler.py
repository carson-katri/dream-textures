import enum

class Scheduler(enum.Enum):
    DDIM = "DDIM"
    DDPM = "DDPM"
    DEIS_MULTISTEP = "DEIS Multistep"
    DPM_SOLVER_MULTISTEP = "DPM Solver Multistep"
    DPM_SOLVER_MULTISTEP_KARRAS = "DPM Solver Multistep Karras"
    DPM_SOLVER_SINGLESTEP = "DPM Solver Singlestep"
    DPM_SOLVER_SINGLESTEP_KARRAS = "DPM Solver Singlestep Karras"
    EULER_DISCRETE = "Euler Discrete"
    EULER_DISCRETE_KARRAS = "Euler Discrete Karras"
    EULER_ANCESTRAL_DISCRETE = "Euler Ancestral Discrete"
    HEUN_DISCRETE = "Heun Discrete"
    HEUN_DISCRETE_KARRAS = "Heun Discrete Karras"
    KDPM2_DISCRETE = "KDPM2 Discrete" # Non-functional on mps
    KDPM2_ANCESTRAL_DISCRETE = "KDPM2 Ancestral Discrete"
    LMS_DISCRETE = "LMS Discrete"
    LMS_DISCRETE_KARRAS = "LMS Discrete Karras"
    PNDM = "PNDM"
    UNIPC_MULTISTEP = "UniPC Multistep"

    def create(self, pipeline):
        import diffusers
        def scheduler_class():
            match self:
                case Scheduler.DDIM:
                    return diffusers.schedulers.DDIMScheduler
                case Scheduler.DDPM:
                    return diffusers.schedulers.DDPMScheduler
                case Scheduler.DEIS_MULTISTEP:
                    return diffusers.schedulers.DEISMultistepScheduler
                case Scheduler.DPM_SOLVER_MULTISTEP | Scheduler.DPM_SOLVER_MULTISTEP_KARRAS:
                    return diffusers.schedulers.DPMSolverMultistepScheduler
                case Scheduler.DPM_SOLVER_SINGLESTEP | Scheduler.DPM_SOLVER_SINGLESTEP_KARRAS:
                    return diffusers.schedulers.DPMSolverSinglestepScheduler
                case Scheduler.EULER_DISCRETE | Scheduler.EULER_DISCRETE_KARRAS:
                    return diffusers.schedulers.EulerDiscreteScheduler
                case Scheduler.EULER_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.EulerAncestralDiscreteScheduler
                case Scheduler.HEUN_DISCRETE | Scheduler.HEUN_DISCRETE_KARRAS:
                    return diffusers.schedulers.HeunDiscreteScheduler
                case Scheduler.KDPM2_DISCRETE:
                    return diffusers.schedulers.KDPM2DiscreteScheduler
                case Scheduler.KDPM2_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.KDPM2AncestralDiscreteScheduler
                case Scheduler.LMS_DISCRETE | Scheduler.LMS_DISCRETE_KARRAS:
                    return diffusers.schedulers.LMSDiscreteScheduler
                case Scheduler.PNDM:
                    return diffusers.schedulers.PNDMScheduler
                case Scheduler.UNIPC_MULTISTEP:
                    return diffusers.schedulers.UniPCMultistepScheduler
        original_config = getattr(pipeline.scheduler, "_original_config", pipeline.scheduler.config)
        scheduler = scheduler_class().from_config(original_config, use_karras_sigmas=self.name.endswith("KARRAS"))
        scheduler._original_config = original_config
        pipeline.scheduler = scheduler
        return scheduler
