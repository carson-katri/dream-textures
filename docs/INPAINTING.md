# Inpainting
Use Stable Diffusion to fill in gaps in images, add new content, fix artifacting, make existing textures seamless, and more.

1. Open an image in the Image Editor
2. Select the *Paint* mode
3. Use the *Mark Inpaint Area* brush to erase the alpha channel from the desired part of the image
4. Check *Inpaint Open Image*
5. Enter a prompt describing what you want and click *Generate*

![A screenshot of an Image Editor space in 'Paint' mode with the 'Mark Inpaint Area' brush active, a section of the image alpha erased, and the 'Inpaint Open Image' option checked in the Dream Textures' UI](assets/inpainting.png)

## Making Textures Seamless
Inpainting can also be used to make an existing texture seamless.

1. Use the *Mark Inpaint Area* brush to remove the edges of the image
2. Enter a prompt that describes the texture, and check *Seamless*
3. Click *Generate*

![A screenshot of an Image Editor space with the edges of a brick texture at 0% alpha, and the 'Seamless' and 'Inpaint Open Image' options checked in the Dream Textures' UI](assets/inpainting-seamless.png)

## Adding New Content
Note that adding new content with inpainting may require a bit of manual work at this time. Filling the area to inpaint with noise before attempting to generate something new in that spot can help prevent Stable Diffusion from just copying what was there before. There is no option to do this automatically at this time.