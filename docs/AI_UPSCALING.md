# AI Upscaling
Real-ESRGAN is built-in to the addon to upscale any generated image 2-4x the original size.

> You must setup the Real-ESRGAN weights separately from the Stable Diffusion weights before upscaling. The *AI Upscaling* panel contains instructions for downloading them.

1. Open the image to upscale an *Image Editor* space
2. Expand the *AI Upscaling* panel, located in the *Dream* sidebar tab
3. Choose a target size and click *Upscale*

> Some GPUs will require Full Precision to be enabled.

![A screenshot of the AI Upscaling panel set to 2 times target size and full precision enabled](../readme_assets/upscaling.png)

The upscaled image will be opened in the *Image Editor*. The image will be named `Source Image Name (Upscaled)`.