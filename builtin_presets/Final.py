import bpy
prompt = bpy.context.scene.dream_textures_prompt

prompt.precision = 'auto'
prompt.random_seed = True
prompt.seed = '0'
prompt.steps = 50
prompt.cfg_scale = 7.5
prompt.sampler_name = 'k_lms'
prompt.show_steps = False
