import bpy
prompt = bpy.context.scene.dream_textures_prompt

prompt.steps = 50
prompt.cfg_scale = 7.5
prompt.scheduler = 'DPM Solver Multistep'
prompt.step_preview_mode = 'Fast'
prompt.optimizations_attention_slicing = True
prompt.optimizations_attention_slice_size_src = 'auto'
prompt.optimizations_attention_slice_size = 1
prompt.optimizations_cudnn_benchmark = False
prompt.optimizations_tf32 = False
prompt.optimizations_amp = False
prompt.optimizations_half_precision = True
prompt.optimizations_sequential_cpu_offload = False
prompt.optimizations_channels_last_memory_format = False
prompt.optimizations_batch_size = 1
prompt.optimizations_vae_slicing = True
prompt.optimizations_cpu_only = False
