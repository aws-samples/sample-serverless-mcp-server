#!/usr/bin/env python3
"""
ComfyUI Image/Video Generation MCP Server (STDIO)

This MCP server provides image and video generation capabilities using ComfyUI,
supporting local file paths for image-to-image and image-to-video workflows.
Built with FastMCP library.
"""

import os
import base64
import sys
from typing import Dict, Any
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Add the server directory to the path to import ComfyUI generator
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from comfyui_generator import ComfyUIGenerator, ComfyUIConfig, ImageUtils

# Initialize FastMCP server
mcp = FastMCP("ComfyUI Generator")

# Initialize ComfyUI generator
generator = ComfyUIGenerator()


def read_image_file(file_path: str) -> str:
    """Read image file and convert to data URL format"""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Determine MIME type based on file extension
        ext = path.suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }

        mime_type = mime_types.get(ext, 'image/jpeg')

        # Read and encode the image
        with open(path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        return f"data:{mime_type};base64,{image_data}"

    except Exception as e:
        raise ValueError(f"Failed to read image file {file_path}: {str(e)}")


def save_result_file(data_url: str, output_path: str) -> str:
    """Save generated image/video to file"""
    try:
        # Parse data URL
        if not data_url.startswith('data:'):
            raise ValueError("Invalid data URL format")

        header, data = data_url.split(',', 1)
        mime_type = header.split(';')[0].split(':')[1]

        # Decode base64 data
        file_data = base64.b64decode(data)

        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(output_path, 'wb') as f:
            f.write(file_data)

        return str(output_path.absolute())

    except Exception as e:
        raise ValueError(f"Failed to save result file: {str(e)}")


@mcp.tool()
def generate_image_from_text(
    prompt: str,
    output_path: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    negative_prompt: str = "bad quality, blurry, low resolution"
) -> str:
    """Generate image from text prompt using ComfyUI flux_t2i workflow

    Args:
        prompt: Text description for image generation
        output_path: Output file path for generated image
        width: Image width (default: 1024)
        height: Image height (default: 1024)
        steps: Number of sampling steps (default: 20)
        cfg_scale: CFG guidance scale (default: 7.0)
        seed: Random seed, -1 for random (default: -1)
        negative_prompt: Negative prompt (default: "bad quality, blurry, low resolution")

    Returns:
        Success message with output path
    """
    try:
        # Generate image using ComfyUI flux_t2i workflow
        result = generator.generate_text_to_image(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            negative_prompt=negative_prompt
        )

        if result.get('error'):
            raise Exception(result['error'])

        # Save result to file
        saved_path = save_result_file(result['data'], output_path)

        return f"Image generated successfully and saved to {saved_path}"

    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")


@mcp.tool()
def generate_image_from_image(
    prompt: str,
    input_image_path: str,
    output_path: str,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    denoise_strength: float = 0.75,
    negative_prompt: str = "bad quality, blurry, low resolution"
) -> str:
    """Generate image from input image and text prompt using ComfyUI flux_kontext workflow

    Args:
        prompt: Text description for image transformation
        input_image_path: Path to input image file
        output_path: Output file path for generated image
        steps: Number of sampling steps (default: 20)
        cfg_scale: CFG guidance scale (default: 7.0)
        seed: Random seed, -1 for random (default: -1)
        denoise_strength: Denoising strength 0.0-1.0 (default: 0.75)
        negative_prompt: Negative prompt (default: "bad quality, blurry, low resolution")

    Returns:
        Success message with input and output paths
    """
    try:
        # Read input image
        image_data = read_image_file(input_image_path)

        # Generate image using ComfyUI flux_kontext workflow
        result = generator.generate_image_to_image(
            prompt=prompt,
            image_data=image_data,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            denoise_strength=denoise_strength,
            negative_prompt=negative_prompt
        )

        if result.get('error'):
            raise Exception(result['error'])

        # Save result to file
        saved_path = save_result_file(result['data'], output_path)

        return f"Image generated successfully from {input_image_path} and saved to {saved_path}"

    except Exception as e:
        raise Exception(f"Failed to generate image from image: {str(e)}")


@mcp.tool()
def generate_video_from_text(
    prompt: str,
    output_path: str,
    steps: int = 15,
    cfg_scale: float = 6.0,
    seed: int = -1,
    frame_rate: int = 16,
    negative_prompt: str = "bad quality video, blurry, low resolution"
) -> str:
    """Generate video from text prompt using ComfyUI wanv_t2v workflow

    Args:
        prompt: Text description for video generation
        output_path: Output file path for generated video
        steps: Number of sampling steps (default: 15)
        cfg_scale: CFG guidance scale (default: 6.0)
        seed: Random seed, -1 for random (default: -1)
        frame_rate: Video frame rate (default: 16)
        negative_prompt: Negative prompt (default: "bad quality video, blurry, low resolution")

    Returns:
        Success message with output path
    """
    try:
        # Generate video using ComfyUI wanv_t2v workflow
        result = generator.generate_text_to_video(
            prompt=prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            frame_rate=frame_rate,
            negative_prompt=negative_prompt
        )

        if result.get('error'):
            raise Exception(result['error'])

        # Save result to file
        saved_path = save_result_file(result['data'], output_path)

        return f"Video generated successfully and saved to {saved_path}"

    except Exception as e:
        raise Exception(f"Failed to generate video: {str(e)}")


@mcp.tool()
def generate_video_from_image(
    prompt: str,
    input_image_path: str,
    output_path: str,
    steps: int = 15,
    cfg_scale: float = 6.0,
    seed: int = -1,
    frame_rate: int = 16,
    negative_prompt: str = "bad quality video, blurry, low resolution"
) -> str:
    """Generate video from input image and text prompt using ComfyUI wan_i2v workflow

    Args:
        prompt: Text description for video generation
        input_image_path: Path to input image file
        output_path: Output file path for generated video
        steps: Number of sampling steps (default: 15)
        cfg_scale: CFG guidance scale (default: 6.0)
        seed: Random seed, -1 for random (default: -1)
        frame_rate: Video frame rate (default: 16)
        negative_prompt: Negative prompt (default: "bad quality video, blurry, low resolution")

    Returns:
        Success message with input and output paths
    """
    try:
        # Read input image
        image_data = read_image_file(input_image_path)

        # Generate video using ComfyUI wan_i2v workflow
        result = generator.generate_image_to_video(
            prompt=prompt,
            image_data=image_data,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            frame_rate=frame_rate,
            negative_prompt=negative_prompt
        )

        if result.get('error'):
            raise Exception(result['error'])

        # Save result to file
        saved_path = save_result_file(result['data'], output_path)

        return f"Video generated successfully from {input_image_path} and saved to {saved_path}"

    except Exception as e:
        raise Exception(f"Failed to generate video from image: {str(e)}")


if __name__ == "__main__":
    mcp.run()