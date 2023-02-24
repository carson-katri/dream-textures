from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .load_pipe import load_pipe, configure_model_padding
from ..models import *
from .detect_seamless import SeamlessAxes
from .depth_to_image import depth_to_image

class RegionType(enum.IntEnum):
    KEEP = 0
    REFINE = 1
    GENERATE = 2

def mesh_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    strength: float,
    prompt: str | list[str],
    steps: int,
    seed: int,

    width: int | None,
    height: int | None,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,

    seamless_axes: SeamlessAxes | str | bool | tuple[bool, bool] | None,

    step_preview_mode: StepPreviewMode,

    # mesh data
    verts: NDArray,
    verts_normals: NDArray,
    faces: NDArray,
    faces_uvs: NDArray,
    verts_uvs: NDArray,

    inpaint_model: str,
    inpaint_steps: range,

    **kwargs
) -> Generator[ImageGenerationResult, None, None]:
    """
    Texture generation based on the [TEXTure paper](https://arxiv.org/abs/2302.01721) (Elad Richardson, Gal Metzer, et al.).
    """
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            import PIL.Image
            import PIL.ImageOps

            from pytorch3d.structures.meshes import Meshes
            from pytorch3d.renderer.mesh.shader import ShaderBase
            from pytorch3d.renderer import MeshRenderer, MeshRasterizer, RasterizationSettings, TexturesUV, FoVPerspectiveCameras, HardFlatShader, DirectionalLights, look_at_view_transform, FoVOrthographicCameras, hard_rgb_blend
            from pytorch3d.renderer.mesh.rasterizer import Fragments

            from torchvision.transforms.functional import gaussian_blur

            class FixedInterleavedDepth2ImagePipeline(diffusers.StableDiffusionDepth2ImgPipeline):
                # copied from diffusers.StableDiffusionPipeline
                def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
                    shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    if isinstance(generator, list) and len(generator) != batch_size:
                        raise ValueError(
                            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                        )

                    if latents is None:
                        latents = diffusers.utils.randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                    else:
                        latents = latents.to(device)

                    # scale the initial noise by the standard deviation required by the scheduler
                    latents = latents * self.scheduler.init_noise_sigma
                    return latents

                @torch.no_grad()
                def __call__(
                    self,

                    prompt: Union[str, List[str]] = None,
                    image: Union[torch.FloatTensor, PIL.Image.Image] = None,
                    depth_map: Optional[torch.FloatTensor] = None,
                    strength: float = 0.8,
                    num_inference_steps: Optional[int] = 50,
                    guidance_scale: Optional[float] = 7.5,
                    negative_prompt: Optional[Union[str, List[str]]] = None,
                    num_images_per_prompt: Optional[int] = 1,
                    eta: Optional[float] = 0.0,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    prompt_embeds: Optional[torch.FloatTensor] = None,
                    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                    output_type: Optional[str] = "pil",
                    return_dict: bool = True,
                    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                    callback_steps: Optional[int] = 1,

                    fixed_latents: torch.FloatTensor | None = None,
                    fixed_latents_mask: torch.FloatTensor | None = None,
                    step_slice: slice | None = None,
                    return_latents: bool = False,
                ):
                    # 1. Check inputs
                    self.check_inputs(prompt, strength, callback_steps)

                    if image is None:
                        raise ValueError("`image` input cannot be undefined.")

                    # 2. Define call parameters
                    batch_size = 1 if isinstance(prompt, str) else len(prompt)
                    device = self._execution_device
                    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                    # corresponds to doing no classifier free guidance.
                    do_classifier_free_guidance = guidance_scale > 1.0

                    # 3. Encode input prompt
                    prompt_embeds = self._encode_prompt(
                        prompt,
                        device,
                        num_images_per_prompt,
                        do_classifier_free_guidance,
                        negative_prompt,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                    )

                    # 4. Prepare depth mask
                    depth_mask = self.prepare_depth_map(
                        image,
                        depth_map,
                        batch_size * num_images_per_prompt,
                        do_classifier_free_guidance,
                        prompt_embeds.dtype,
                        device,
                    )

                    # 5. Preprocess image
                    image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_depth2img.preprocess(image)

                    # 6. Set timesteps
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
                    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

                    # 7. Prepare latent variables
                    latents = self.prepare_latents(
                        # NOTE: Don't use initial image latents
                        batch_size, 4, depth_map.shape[2], depth_map.shape[1], prompt_embeds.dtype, device, generator, None
                        # image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
                    )

                    # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
                    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

                    # NOTE: Create noise to add to fixed latents
                    noise = torch.randn_like(latents)

                    # 9. Denoising loop
                    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                    with self.progress_bar(total=num_inference_steps) as progress_bar:
                        # NOTE: slice timesteps
                        for i, t in list(enumerate(timesteps))[step_slice or slice(0, len(timesteps))]:
                            # NOTE: Fixed latents transform.
                            if fixed_latents is not None:
                                noised_truth = self.scheduler.add_noise(fixed_latents, noise, t)
                                latents = latents * fixed_latents_mask + noised_truth * (1 - fixed_latents_mask)

                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                            latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

                            # predict the noise residual
                            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            # compute the previous noisy sample x_t -> x_t-1
                            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                            # call the callback, if provided
                            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                                progress_bar.update()
                                if callback is not None and i % callback_steps == 0:
                                    callback(i, t, latents)

                    # NOTE: just return the latents.
                    if return_latents:
                        return latents
                    else:
                        return self.decode_latents(latents)

            # 1a. Setup for rendering
            size = (width or 512, height or 512)

            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device()
            dtype = torch.float32
            index_dtype = torch.long

            texture = torch.from_numpy(np.array(PIL.Image.new('RGB', size, (1, 1, 1)))).to(dtype=dtype)
            faces_uvs = [torch.from_numpy(faces_uvs).to(dtype=index_dtype)]
            verts_uvs = [torch.from_numpy(verts_uvs).to(dtype=dtype)]
            textures_uv = TexturesUV(maps=[texture], faces_uvs=faces_uvs, verts_uvs=verts_uvs)
            verts = [torch.from_numpy(verts).to(dtype=dtype)]
            verts[0][:, 0] = 1 - verts[0][:, 0]
            verts[0] = torch.gather(verts[0], 1, torch.tensor([0, 2, 1]).repeat(verts[0].shape[0], 1))
            faces = [torch.from_numpy(faces).to(dtype=index_dtype)]
            verts_normals = [torch.from_numpy(verts_normals).to(dtype=dtype)]
            verts_normals[0] = torch.gather(verts_normals[0], 1, torch.tensor([0, 2, 1]).repeat(verts_normals[0].shape[0], 1))
            meshes = Meshes(verts, faces, textures_uv, verts_normals=verts_normals)

            def fov_perspective_cameras(dist, elev, azim, at):
                R, T = look_at_view_transform(dist, elev, azim, at=[at])
                return FoVPerspectiveCameras(R=R, T=T)

            def render_depth(fragments) -> torch.Tensor:
                """Get the normalized depth from the rasterizer output"""
                mask = fragments.zbuf == -1
                zbuf = torch.clone(fragments.zbuf)
                zbuf[mask] = zbuf.max()
                zbuf = (zbuf.max() - zbuf)
                zbuf /= zbuf.max()
                return zbuf.squeeze(dim=-1)

            # set the camera angle to a default angle
            cameras = fov_perspective_cameras(4, 0, 0, (1, 0, 0))
            
            # create the shared rasterizer
            raster_settings = RasterizationSettings(image_size=size)
            rasterizer = MeshRasterizer(cameras, raster_settings=raster_settings)

            # 1b. Load Stable Diffusion pipelines.
            generator = torch.Generator(device="cpu" if device in ("mps", "privateuseone") else device) # MPS and DML do not support the `Generator` API
            generator.manual_seed(random.randrange(0, np.iinfo(np.uint32).max) if seed is None else seed)

            depth_pipe = load_pipe(self, "fixed_interleaved_depth", FixedInterleavedDepth2ImagePipeline, model, optimizations, scheduler, device)
            # inpaint_pipe = load_pipe(self, "inpaint", diffusers.StableDiffusionInpaintPipeline, inpaint_model, optimizations, scheduler, device)
            depth_init_image = PIL.Image.new('RGB', size) # The `image` argument is required by the default pipeline for some reason?

            # 2. Render the initial view with depth to image.
            fragments = rasterizer(meshes)
            depth = render_depth(fragments)

            initial_image = torch.tensor(depth_pipe(
                prompt=prompt,
                image=depth_init_image,
                depth_map=depth,
                strength=1.0,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                negative_prompt=negative_prompt,
                generator=generator
            )[0]).to(dtype=dtype)
            # initial_image = torch.from_numpy(np.array(PIL.Image.open('initial_image.png'))).to(dtype=dtype) / 255

            if initial_image is None:
                raise ValueError("Failed to generate the initial view with depth to image.")

            # 3. Reproject the initial view onto the UVs.
            # create a UV map by using the UVs as the world space coordinates.
            reprojection_meshes = Meshes([torch.nn.functional.pad(verts_uvs[0], pad=(0, 1, 0, 0), mode='constant', value=1)], faces)
            class HardShader(ShaderBase):
                def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
                    texels = meshes.sample_textures(fragments)
                    blend_params = kwargs.get("blend_params", self.blend_params)
                    images = hard_rgb_blend(texels, fragments, blend_params)
                    return images
            def bake(image):
                screen_space_uvs = cameras.transform_points_screen(verts[0], image_size=size)[:, :2] / torch.tensor(size)
                screen_space_uvs[:, 1] = 1 - screen_space_uvs[:, 1]
                reprojection_meshes.textures = TexturesUV(maps=[image], faces_uvs=faces_uvs, verts_uvs=[screen_space_uvs])
                reproj_cameras = FoVOrthographicCameras(min_y=0, max_y=1, min_x=0, max_x=1)
                renderer = MeshRenderer(rasterizer, HardShader())
                return renderer(reprojection_meshes, cameras=reproj_cameras)
                # PIL.Image.fromarray(np.uint8((output.squeeze() * 255).detach().numpy())).save('initial_repoj.png')
            # start the final output with the baked result.
            output = bake(initial_image)

            # 4. Create the "viewpoint cache," which stores which perspectives have been seen from a good angle.
            def update_cache(cache, cameras):
                class CacheShader(HardFlatShader):
                    """A shader that lights the scene using different cameras and meshes than the renderer."""
                    unflattened_mesh: Meshes

                    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
                        return super().forward(fragments, self.unflattened_mesh, **kwargs)

                reproj_cameras = FoVOrthographicCameras(min_y=0, max_y=1, min_x=0, max_x=1)
                
                # use a light in the direction of the camera to calculate facing direction
                lights = DirectionalLights(ambient_color=[(0, 0, 0)], diffuse_color=[(1, 1, 1)], specular_color=[(0, 0, 0)], direction=cameras.get_camera_center())
                
                # the cache shader passes a different mesh to the `HardFlatShader`.
                shader = CacheShader(cameras=cameras, lights=lights)
                shader.unflattened_mesh = meshes

                # render the cache
                renderer = MeshRenderer(rasterizer, shader)
                output = renderer(reprojection_meshes, cameras=reproj_cameras)
                output[output[:, :, :, -1] == 0, :] = 0

                # merge with the previous
                return (cache + output[0, :, :, :1]).clip(0, 1)
                # PIL.Image.fromarray(np.uint8((output.squeeze() * 255).detach().numpy())).save('camera_normals.png')
            cache = update_cache(torch.zeros((*size, 1)), cameras)

            # 5. Begin main loop
            def partition_regions(cache, cameras):
                """Returns the (generate, refine, keep) regions"""
                new_angle = update_cache(torch.zeros_like(cache), cameras)
                return torch.stack((
                    cache == 0, # generate, any pure black areas of the cache
                    new_angle > cache, # refine, any area that is facing the current perspective more than previously
                    new_angle <= cache # keep, any area that is facing less or equal the previous perspective
                ), dim=2).squeeze()
            
            def project_image(image, cameras):
                """Renders the image back onto the mesh."""
                meshes.textures = TexturesUV(maps=image[:, :, :, :3], faces_uvs=faces_uvs, verts_uvs=verts_uvs)
                renderer = MeshRenderer(rasterizer, HardShader())
                return renderer(meshes, cameras=cameras)

            for perspective in [
                fov_perspective_cameras(4, 0, 90, (1, 0, 0)),
                fov_perspective_cameras(4, 0, 180, (1, 0, 0)),
                fov_perspective_cameras(4, 0, 270, (1, 0, 0)),
            ]:
                regions = partition_regions(cache, perspective)
                fragments = rasterizer(meshes, cameras=perspective)
                depth = render_depth(fragments)
                rendered_output = project_image(output, perspective)
                rendered_generate = project_image(regions[None, :, :, 0, None].to(dtype=dtype), perspective)
                image = depth_pipe(
                    prompt=prompt,
                    image=depth_init_image,
                    depth_map=depth,
                    strength=1.0,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    output_type=None,
                    fixed_latents=depth_pipe.vae.config.scaling_factor * depth_pipe.vae.encode((2 * rendered_output[:, :, :, :3].to(device).permute(0, 3, 1, 2) - 1)).latent_dist.sample(),
                    fixed_latents_mask=torch.nn.functional.interpolate(rendered_generate[None, 0, :, :, 0, None].permute(0, 3, 1, 2).to(device=device), (64, 64), mode='nearest'),
                    return_latents=False
                )
                image = (torch.from_numpy(np.array(PIL.Image.open('rendered_output.png'))).to(dtype=dtype) / 255).unsqueeze(0)
                cache = update_cache(cache, perspective)
                new_output = bake(torch.tensor(image[0]).to(dtype=dtype))
                generate_region = regions[:, :, 0, None].to(dtype=dtype)
                generate_region = gaussian_blur(generate_region.permute(2, 0, 1), 21, 16).permute(1, 2, 0).unsqueeze(0)
                output = (new_output * generate_region) + (output * (1 - generate_region))
                yield ImageGenerationResult(
                    images=[output[0].detach().numpy()],
                    seeds=[0],
                    step=1,
                    final=False
                )
            
            def debug():
                PIL.Image.fromarray(np.uint8(initial_image * 255)).save('initial_image.png')
                PIL.Image.fromarray(np.uint8(output[0] * 255)).save('initial_bake.png')
                PIL.Image.fromarray(np.uint8(cache[:, :, 0] * 255)).save('cache.png')
                PIL.Image.fromarray(np.uint8(rendered_generate[0, :, :, 0] * 255)).save('generate_regions.png')
            
        case Pipeline.STABILITY_SDK:
            import stability_sdk
            raise NotImplementedError()
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")