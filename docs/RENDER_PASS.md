# Render Pass
A custom 'Dream Textures' render pass is available for Cycles. This can allow you to run Dream Textures on the render result each time the scene is rendered. It works well with animations as well, and can be used to perform style transfer on each frame of an animation.

> The render pass and Cycles will both use significant amounts of VRAM depending on the scene. You can use the CPU to render on Cycles to save resources for Dream Textures.

1. In the *Render Properties* panel, switch to the *Cycles* render engine
2. Enable the *Dream Textures* render pass, and enter a text prompt.

> In the *Output Properties* panel, ensure the image size is reasonable for your GPU and Stable Diffusion. 512x512 is a good place to start.

![A screenshot of the Render Properties panel with the Cycles render engine selected, and the Dream Textures render pass checked](../readme_assets/render-pass.png)

3. To use the Dream Textures generated image as the final result, open the *Compositor* space
4. Enable *Use Nodes*
5. Connect the *Dream Textures* socket from the *Render Layers* node to the *Image* socket of the *Composite* node

![A screenshot of the Compositor space with Use Nodes checked and the Dream Textures socket from the Render Layers node connected to the Image socket of the Composite node](../readme_assets/render-pass-compositor.png)

And now each frame of our render will use the generated image. Here's that Rembrandt of the default cube promised by the prompt:

![The default cube in a painting style](../readme_assets/rembrandt-default-cube.png)

## Controlling the Output

### Strength
The strength parameter is very important when using this render pass. It is the same as Init Image strength described in [Image Generation](IMAGE_GENERATION.md). If you want your scene composition, colors, etc. to be preserved, use a lower strength value. If you want Stable Diffusion to take more control, use a higher strength value.

### Seed
Enabling *Random Seed* can give you some cool effects, and allow for more experimentation. However, if you are trying to do simple style transfer on an animation, using a consistent seed can help the animation be more coherent.

## Animation
You can animate most of the properties when using the render pass. Simply create keyframes as you typically would in Blender, and the properties will automatically be updated for each frame.