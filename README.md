# Dream Textures

> Stable Diffusion built-in to the Blender shader editor.

* Create textures, concept art, background assets, and more with a simple text prompt
* Use the 'Seamless' option to create textures that tile perfectly with no visible seam
* Quickly create variations on an existing texture
* Experiment with AI image generation
* Run the models on your machine to iterate without slowdowns from a service

## Installation
1. Download the [latest version](https://github.com/carson-katri/dream-textures/releases/tag/0.0.4) from the Releases tab.
2. Install the addon in Blender's preferences window.
3. Follow the steps in the 'Dream Textures' preferences window to install the necessary dependencies.

| Enter a prompt | Generate a unique texture in a few seconds |
| -------------- | ------------------------------------------ |
| ![](readme_assets/brick_wall_texture_prompt.png) | ![](readme_assets/brick_wall_texture.png) |
| ![](readme_assets/uneven_stone_path_prompt.png) | ![](readme_assets/uneven_stone_path.png) |

| Take an existing texture | Modify it with a text prompt |
| ------------------------ | ---------------------------- |
| ![](readme_assets/marble.jpg) | ![](readme_assets/marble_brick_wall_texture.png) |

> On Windows, you will need to run Blender as an administrator for the installation to complete successfully.

## Usage
Dream Textures provides most of the configuration options available when using Stable Diffusion directly.

1. Open the Shader editor
2. Select the 'Dream Textures' menu in the far right, and choose the 'Dream Texture' operation.
3. Enter a prompt. There are some presets you can use to automatically fine tune your result:
    1. *Texture* - simply adds the word *texture* to the end of your prompt to help the model create a texture.
    2. *Photography* - provides many options for specifying a photorealistic result.
    3. *Custom* - passes the prompt directly to SD with no additional keywords.
4. Click 'OK'. Wait a moment for the model to load. After the model loads, the generation runs asynchronously, so you Blender shouldn't completely freeze up.
5. If you have an Image Viewer open, the current progress of the model will appear there with the current step # as the image name (for example, *Step 1/25*).
6. After the image finishes generating, a new Image Texture node will be added to the open shader editor. The image is packed, so you may want to export it to a separate file.

> The name of the generated image is the random seed used to create it. If you want to use the same seed, copy the name of the image into the 'Seed' parameter under the 'Advanced' section.

## Init Image
Use an init image to create a variation of an existing texture.

![](readme_assets/init_image.png)

1. Enable 'Init Image'
2. Select an image from your filesystem
> Currently, this doesn't work with packed textures. So select an image from your file system, or export the packed texture to a file and select that.
3. You can change the strength and specify if it should fit to the set size, or keep the original image size.
4. Enter a prompt that tells SD how to modify the image.
5. Click 'OK' and wait for it to finish.

## Advanced Configuration
These options match those of SD.

![](readme_assets/advanced_configuration.png)

* *Full Precision* - more VRAM intensive, but creates better results. Disable this if you have a lower-end graphics card.
* *Seed* - a random seed if set to `-1`, otherwise it uses the value you input. You can copy the random seed from the generated image's name to use the same seed again.
* *Iterations* - how many images to generate. This doesn't quite work yet, so keep it at `1`.
* *Steps* - higher is generally better, but I wouldn't recommend going past 50 until you've refined your prompt.
* *CFG Scale* - essentially how much the model takes your prompt into consideration. `7.5` is a good default for a collaborative experience, but if you want to force the model to follow instructions crank it up to `15-20`.
* *Sampler* - KLMS is a good speed/quality default, DDIM is generally faster, and you'll get different results with each so play around with it.

## Compatibility
Dream Textures has been tested with CUDA and Apple Silicon GPUs.

If you have an issue with a supported GPU, please create an issue.

## Future Directions
* Other image map types (normal, roughness, displacement, etc.) using a new LDM checkpoint and vocabulary.
* AI upscaling and face fixing with ESRGAN and GFPGAN
