from lambda_mcp.lambda_mcp import LambdaMCPServer
import boto3
import os
import requests
from typing import Dict
from comfyui_generator import ComfyUIGenerator, ImageUtils, ConfigManager

SERPAPI_API_KEY = "*******"
# Get session table name from environment variable
session_table = os.environ.get('MCP_SESSION_TABLE', 'mcp_sessions')

# Create the MCP server instance
mcp_server = LambdaMCPServer(name="mcp-lambda-server", version="1.0.0", session_table=session_table)

# Create ComfyUI generator instance
comfyui_generator = ComfyUIGenerator()

@mcp_server.tool()
def search_website(search_term:str) -> Dict:
    """querying something which we don't know and need to search the website to get the exactly and latest information
    Args:
        search_term: user query text
        
    Returns:
        search result string
    """
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "q": search_term,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }

    url = "https://serpapi.com/search"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        organic_results = results.get('organic_results', [])
        return {"search_result":organic_results}
    else:
        return {"search_result":response.text}



@mcp_server.tool()
def count_s3_buckets() -> int:
    """Count the number of S3 buckets."""
    s3 = boto3.client('s3')
    response = s3.list_buckets()
    return len(response['Buckets'])

@mcp_server.tool()
def generate_image_with_context(prompt: str, context_image_base64: str = None, workflow_type: str = "text_to_image", width: int = 1024, height: int = 1024, steps: int = 20, cfg_scale: float = 7.0, seed: int = -1) -> Dict:
    """Generate image using ComfyUI with optional context image for img2img scenarios.

    Args:
        prompt: Text description for image generation
        context_image_base64: Base64 encoded context image with data URL prefix (optional)
        workflow_type: Type of workflow (text_to_image, image_to_image, inpainting)
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        steps: Number of sampling steps (default: 20)
        cfg_scale: CFG scale for guidance (default: 7.0)
        seed: Random seed (-1 for random, default: -1)

    Returns:
        Generated image data with metadata
    """
    try:
        if workflow_type == "text_to_image":
            return comfyui_generator.generate_text_to_image(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed
            )
        elif workflow_type == "image_to_image" and context_image_base64:
            return comfyui_generator.generate_image_to_image(
                prompt=prompt,
                image_data=context_image_base64,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed
            )

        else:
            # Fallback to text-to-image if no valid context provided
            return comfyui_generator.generate_text_to_image(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed
            )

    except Exception as e:
        # Fallback to mock image for testing
        print(f"ComfyUI API error, using mock image: {str(e)}")
        return comfyui_generator.get_mock_image_response(
            operation=workflow_type,
            prompt=prompt,
            dimensions=f"{width}x{height}",
            has_context_image=context_image_base64 is not None
        )



@mcp_server.tool()
def get_comfyui_config() -> Dict:
    """Get ComfyUI configuration and available workflows.

    Returns:
        Configuration information including available workflows and presets
    """
    try:
        return {
            'server_url': comfyui_generator.config.server_url,
            'available_workflows': ConfigManager.get_available_workflows(),
            'workflow_presets': ConfigManager.get_workflow_presets(),
            'optimal_dimensions': {
                'square': ConfigManager.get_optimal_dimensions('square'),
                'portrait': ConfigManager.get_optimal_dimensions('portrait'),
                'landscape': ConfigManager.get_optimal_dimensions('landscape'),
                'wide': ConfigManager.get_optimal_dimensions('wide'),
                'tall': ConfigManager.get_optimal_dimensions('tall')
            },
            'config': {
                'timeout': comfyui_generator.config.timeout,
                'poll_interval': comfyui_generator.config.poll_interval,
                'max_retries': comfyui_generator.config.max_retries,
                'enable_fallback': comfyui_generator.config.enable_fallback,
                'workflow_dir': comfyui_generator.config.workflow_dir
            }
        }
    except Exception as e:
        return {"error": f"Failed to get ComfyUI configuration: {str(e)}"}

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    return mcp_server.handle_request(event, context)

