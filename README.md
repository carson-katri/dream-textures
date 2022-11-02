![Dream Textures, subtitle: Stable Diffusion built-in to Blender](readme_assets/banner.png)

[![Latest Release](https://flat.badgen.net/github/release/carson-katri/dream-textures)](https://github.com/carson-katri/dream-textures/releases/latest)
[![Join the Discord](https://flat.badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/EmDJ8CaWZ7)
[![Total Downloads](https://img.shields.io/github/downloads/carson-katri/dream-textures/total?style=flat-square)](https://github.com/carson-katri/dream-textures/releases/latest)

* Create textures, concept art, background assets, and more with a simple text prompt
* Use the 'Seamless' option to create textures that tile perfectly with no visible seam
* Quickly create variations on an existing texture
* Re-style animations with the Cycles render pass
* Run the models on your machine to iterate without slowdowns from a service

# Installation
Download the [latest release](https://github.com/carson-katri/dream-textures/releases/latest) and follow the instructions there to get up and running.

> On macOS, it is possible you will run into a quarantine issue with the dependencies. To work around this, run the following command in the app `Terminal`: `xattr -r -d com.apple.quarantine ~/Library/Application\ Support/Blender/3.3/scripts/addons/dream_textures/.python_dependencies`. This will allow the PyTorch `.dylib`s and `.so`s to load without having to manually allow each one in System Preferences.

If you want a visual guide to installation, see this video tutorial from Ashlee Martino-Tarr: https://youtu.be/kEcr8cNmqZk
> Ensure you always install the [latest version](https://github.com/carson-katri/dream-textures/releases/latest) of the add-on if any guides become out of date.

# Usage

Here's a few quick guides:

## [Image Generation](docs/IMAGE_GENERATION.md)
Create textures, concept art, and more with text prompts. Learn how to use the various configuration options to get exactly what you're looking for.

## [Inpainting](docs/INPAINTING.md)
Fix up images and convert existing textures into seamless ones automatically.

## [Render Pass](docs/RENDER_PASS.md)
Perform style transfer and create novel animations with Stable Diffusion as a post processing step.

## [AI Upscaling](docs/AI_UPSCALING.md)
Convert your low-res generations to 2K, 4K, and higher with Real-ESRGAN built-in.

## [History](docs/HISTORY.md)
Recall, export, and import history entries for later use.

# Compatibility
Dream Textures has been tested with CUDA and Apple Silicon GPUs. Over 4GB of VRAM is recommended.

If you have an issue with a supported GPU, please create an issue.

### Cloud Processing
If your hardware is unsupported, you can use DreamStudio to process in the cloud. Follow the instructions in the release notes to setup with DreamStudio.

# Contributing
After cloning the repository, there a few more steps you need to complete to setup your development environment:
1. Install submodules:
```sh
git submodule update --init --recursive
```
2. I recommend the [Blender Development](https://marketplace.visualstudio.com/items?itemName=JacquesLucke.blender-development) extension for VS Code for debugging. If you just want to install manually though, you can put the `dream_textures` repo folder in Blender's addon directory.
3. After running the local add-on in Blender, setup the model weights like normal.
4. Install dependencies locally
    * Open Blender's preferences window
    * Enable *Interface* > *Display* > *Developer Extras*
    * Then install dependencies for development under *Add-ons* > *Dream Textures* > *Development Tools*
    * This will download all pip dependencies for the selected platform into `.python_dependencies`

### Tips

1. On Apple Silicon, with the `requirements-dream-studio.txt` you may run into an error with gRPC using an incompatible binary. If so, please use the following command to install the correct gRPC version:
```sh
pip install --no-binary :all: grpcio --ignore-installed --target .python_dependencies --upgrade
```
