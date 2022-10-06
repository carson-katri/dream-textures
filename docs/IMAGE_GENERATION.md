# Image Generation
1. To open Dream Textures, go to an Image Editor or Shader Editor
1. Ensure the sidebar is visible by pressing *N* or checking *View* > *Sidebar*
2. Select the 'Dream' panel to open the interface

![A screenshot showing the 'Dream' panel in an Image Editor space](../readme_assets/opening-ui.png)

Enter a prompt then click *Generate*. It can take anywhere from a few seconds to a few minutes to generate, depending on your graphics card.

## Options

### Prompt

A few presets are available to help you create great prompts. They work by asking you to fill in a few simple fields, then generate a full prompt string that is passed to Stable Diffusion.

The default preset is *Texture*. It asks for a subject, and adds the word `texture` to the end. So if you enter `brick wall`, it will use the prompt `brick wall texture`.

### Seamless
Checking seamless will use a circular convolution to create a perfectly seamless image, which works great for textures.

### Negative
Enabling negative prompts gives you finer control over your image. For example, if you asked for a `cloud city`, but you wanted to remove the buildings it added, you could enter the negative prompt `building`. This would tell Stable Diffusion to avoid drawing buildings. You can add as much content you want to the negative prompt, and it will avoid everything entered.

### Size
The target image dimensions. The width/height should be a multiple of 64, or it will round to the closest one for you.

Most graphics cards with 4+GB of VRAM should be able to generate 512x512 images. However, if you are getting CUDA memory errors, try decreasing the size.

> Stable Diffusion was trained on 512x512 images, so you will get the best results at this size (or at least when leaving one dimensions at 512).

### Inpaint Open Image
See [Inpainting](INPAINTING.md) for more information.

### Init Image
Specifies an image to mix with the latent noise. Open any image, and Stable Diffusion will match the style, composition, etc. from it.

Stength specifies how much latent noise to mix with the image. A higher strength means more latent noise, and more deviation from the init image. If you want it to stick to the image more, decrease the strength.

> Depending on the strength value, some steps will be skipped. For example, if you specified `10` steps and set strength to `0.5`, only `5` steps would be used.

Fit to width/height will ensure the image is contained within the configured size.

### Advanced
You can have more control over the generation by trying different values for these parameters:
* Precision - the math precision
    * Automatic - chooses the best option for your GPU
    * Full Precision - uses 32-bit floats, required on some GPUs
    * Half Precision - uses 16-bit floats, faster
    * Autocast - uses the correct precision for each PyTorch operation
* Random Seed - when enabled, a seed will be selected for you
    * Seed - the value used to seed RNG, if text is input instead of a number its hash will be used
* Steps - number of sampler steps, higher steps will give the sampler more time to converge and clear up artifacts
* CFG Scale - how strongly the prompt influences the output
* Sampler - the sampling method to use, all samplers (except for KEULER_A and KDPM_2A) will produce the same image if given enough steps
* Show Steps - whether to show each step in the Image Editor, can slow down generation significantly