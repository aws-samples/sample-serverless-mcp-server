from mcp.server.fastmcp import FastMCP
import boto3
import os
import requests
from typing import Dict
import base64
import json
import time
import uuid

# 创建 FastMCP 实例，配置为 stateless HTTP
mcp = FastMCP(host="0.0.0.0", stateless_http=True)

# 从环境变量获取配置
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY', 'your_serpapi_key_here')
COMFYUI_SERVER_URL = os.environ.get('COMFYUI_SERVER_URL', 'http://localhost:8188')
COMFYUI_TIMEOUT = int(os.environ.get('COMFYUI_TIMEOUT', '300'))
COMFYUI_POLL_INTERVAL = int(os.environ.get('COMFYUI_POLL_INTERVAL', '2'))
COMFYUI_MAX_RETRIES = int(os.environ.get('COMFYUI_MAX_RETRIES', '3'))

# ComfyUI helper functions
def generate_seed() -> int:
    """Generate random seed"""
    return int(time.time() * 1000) % 2147483647

def validate_image_data(image_data: str) -> tuple[str, str]:
    """Validate and extract image data"""
    if not image_data.startswith('data:image/'):
        raise ValueError("Invalid image format. Expected data URL format (data:image/...)")
    
    try:
        header, data = image_data.split(',', 1)
        mime_type = header.split(';')[0].split(':')[1]
        # Validate base64 data
        base64.b64decode(data)
        return mime_type, data
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def test_comfyui_connectivity() -> bool:
    """Test ComfyUI server connectivity"""
    try:
        response = requests.get(f"{COMFYUI_SERVER_URL}/queue", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_flux_t2i_workflow(prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int) -> Dict:
    """Create FLUX text-to-image workflow (based on original project's flux_t2i.json)"""
    return {
        "6": {
            "inputs": {
                "text": prompt,
                "speak_and_recognation": {
                    "__value__": [False, True]
                },
                "clip": ["41", 0]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Positive Prompt)"
            }
        },
        "8": {
            "inputs": {
                "samples": ["31", 0],
                "vae": ["40", 0]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE解码"
            }
        },
        "9": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "保存图像"
            }
        },
        "31": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["38", 0],
                "positive": ["6", 0],
                "negative": ["33", 0],
                "latent_image": ["42", 0]
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "K采样器"
            }
        },
        "33": {
            "inputs": {
                "text": "bad quality, blurry, low resolution",
                "speak_and_recognation": {
                    "__value__": [False, True]
                },
                "clip": ["41", 0]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Negative Prompt)"
            }
        },
        "38": {
            "inputs": {
                "unet_name": "flux1-dev-fp8.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            },
            "class_type": "UNETLoader",
            "_meta": {
                "title": "UNet加载器"
            }
        },
        "40": {
            "inputs": {
                "vae_name": "FLUX1/ae.safetensors"
            },
            "class_type": "VAELoader",
            "_meta": {
                "title": "加载VAE"
            }
        },
        "41": {
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux",
                "device": "default"
            },
            "class_type": "DualCLIPLoader",
            "_meta": {
                "title": "双CLIP加载器"
            }
        },
        "42": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "空Latent图像"
            }
        }
    }

def create_wanvideo_t2v_workflow(prompt: str, steps: int, cfg_scale: float, seed: int, frame_rate: int) -> Dict:
    """创建 WanVideo 文本到视频工作流（基于原始项目的 wanv_t2v.json）"""
    return {
        "11": {
            "inputs": {
                "t5_model_name": "umt5-xxl-enc-bf16.safetensors",
                "dtype": "bf16",
                "device": "offload_device"
            },
            "class_type": "LoadWanVideoT5TextEncoder",
            "_meta": {
                "title": "Load WanVideo T5 Text Encoder"
            }
        },
        "16": {
            "inputs": {
                "positive_prompt": prompt,
                "negative_prompt": "bad quality video, blurry, low resolution, static",
                "enable_positive_prompt": True,
                "t5": ["11", 0]
            },
            "class_type": "WanVideoTextEncode",
            "_meta": {
                "title": "WanVideo Text Encode"
            }
        },
        "22": {
            "inputs": {
                "model_name": "WanVideo/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors",
                "dtype": "bf16",
                "weight_dtype": "fp8_e4m3fn",
                "device": "offload_device",
                "attention_mode": "sdpa",
                "block_swap_args": ["39", 0]
            },
            "class_type": "WanVideoModelLoader",
            "_meta": {
                "title": "WanVideo Model Loader"
            }
        },
        "27": {
            "inputs": {
                "steps": steps,
                "cfg": cfg_scale,
                "denoise": 5,
                "seed": seed,
                "seed_mode": "fixed",
                "enable_vae_tiling": True,
                "sampler_name": "dpm++",
                "scheduler": 0,
                "model": ["22", 0],
                "text_embeds": ["16", 0],
                "image_embeds": ["37", 0]
            },
            "class_type": "WanVideoSampler",
            "_meta": {
                "title": "WanVideo Sampler"
            }
        },
        "28": {
            "inputs": {
                "enable_vae_tiling": True,
                "tile_sample_min_height": 272,
                "tile_sample_min_width": 272,
                "tile_overlap_factor_height": 144,
                "tile_overlap_factor_width": 128,
                "vae": ["38", 0],
                "samples": ["27", 0]
            },
            "class_type": "WanVideoDecode",
            "_meta": {
                "title": "WanVideo Decode"
            }
        },
        "30": {
            "inputs": {
                "frame_rate": frame_rate,
                "loop_count": 0,
                "filename_prefix": "WanVideo_T2V",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["28", 0]
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {
                "title": "Video Combine"
            }
        },
        "37": {
            "inputs": {
                "width": 832,
                "height": 480,
                "frames": 81
            },
            "class_type": "WanVideoEmptyEmbeds",
            "_meta": {
                "title": "WanVideo Empty Embeds"
            }
        },
        "38": {
            "inputs": {
                "vae_name": "wanvideo/Wan2_1_VAE_bf16.safetensors",
                "dtype": "bf16"
            },
            "class_type": "WanVideoVAELoader",
            "_meta": {
                "title": "WanVideo VAE Loader"
            }
        },
        "39": {
            "inputs": {
                "block_swap_memory_threshold": 20
            },
            "class_type": "WanVideoBlockSwap",
            "_meta": {
                "title": "WanVideo Block Swap"
            }
        }
    }

def submit_comfyui_workflow(workflow: Dict) -> Dict:
    """提交工作流到 ComfyUI 并等待结果"""
    if not test_comfyui_connectivity():
        return {"error": "ComfyUI server is not accessible"}

    try:
        # 生成唯一的 prompt ID
        prompt_id = str(uuid.uuid4())

        # 提交工作流
        submit_url = f"{COMFYUI_SERVER_URL}/prompt"
        submit_payload = {
            "prompt": workflow,
            "client_id": prompt_id
        }

        response = requests.post(submit_url, json=submit_payload, timeout=30)
        if response.status_code != 200:
            return {"error": f"Failed to submit workflow (HTTP {response.status_code}): {response.text}"}

        submit_result = response.json()
        if not submit_result.get("prompt_id"):
            return {"error": "No prompt_id returned from ComfyUI"}

        actual_prompt_id = submit_result["prompt_id"]

        # 轮询完成状态
        return poll_comfyui_completion(actual_prompt_id)

    except Exception as e:
        return {"error": f"ComfyUI workflow submission failed: {str(e)}"}

def poll_comfyui_completion(prompt_id: str) -> Dict:
    """轮询 ComfyUI 工作流完成状态"""
    start_time = time.time()

    while time.time() - start_time < COMFYUI_TIMEOUT:
        try:
            # 检查队列状态
            queue_url = f"{COMFYUI_SERVER_URL}/queue"
            queue_response = requests.get(queue_url, timeout=30)

            if queue_response.status_code != 200:
                time.sleep(COMFYUI_POLL_INTERVAL)
                continue

            queue_data = queue_response.json()
            running = queue_data.get("queue_running", [])
            pending = queue_data.get("queue_pending", [])

            prompt_in_queue = any(item[1] == prompt_id for item in running + pending)

            if not prompt_in_queue:
                # 任务完成，获取结果
                return get_comfyui_result(prompt_id, start_time)

            time.sleep(COMFYUI_POLL_INTERVAL)

        except Exception as e:
            time.sleep(COMFYUI_POLL_INTERVAL)

    return {"error": f"ComfyUI generation timeout after {COMFYUI_TIMEOUT} seconds"}

def get_comfyui_result(prompt_id: str, start_time: float) -> Dict:
    """获取 ComfyUI 工作流结果"""
    try:
        # 获取历史记录
        history_url = f"{COMFYUI_SERVER_URL}/history/{prompt_id}"
        history_response = requests.get(history_url, timeout=30)

        if history_response.status_code != 200:
            return {"error": f"Failed to get ComfyUI history (HTTP {history_response.status_code})"}

        history_data = history_response.json()

        if prompt_id not in history_data:
            return {"error": "Prompt not found in history"}

        prompt_data = history_data[prompt_id]
        if "status" in prompt_data and prompt_data["status"].get("status_str") == "error":
            error_details = prompt_data["status"].get("messages", [])
            return {"error": f"ComfyUI execution error: {error_details}"}

        outputs = prompt_data.get("outputs", {})

        # 查找图像或视频输出
        for node_id, node_output in outputs.items():
            # 处理图像输出
            if "images" in node_output:
                images = node_output["images"]
                if images:
                    image_info = images[0]
                    if image_info.get('filename'):
                        image_url = f"{COMFYUI_SERVER_URL}/view?filename={image_info['filename']}&subfolder={image_info.get('subfolder', '')}&type={image_info.get('type', 'output')}"

                        image_response = requests.get(image_url, timeout=30)
                        if image_response.status_code == 200 and len(image_response.content) > 0:
                            image_base64 = base64.b64encode(image_response.content).decode('utf-8')
                            return {
                                "image_data": image_base64,
                                "generation_time": time.time() - start_time,
                                "prompt_id": prompt_id,
                                "image_size": len(image_response.content)
                            }

            # 处理视频输出
            elif "gifs" in node_output:
                videos = node_output["gifs"]
                if videos:
                    video_info = videos[0]
                    if video_info.get('filename'):
                        video_url = f"{COMFYUI_SERVER_URL}/view?filename={video_info['filename']}&subfolder={video_info.get('subfolder', '')}&type={video_info.get('type', 'output')}"

                        video_response = requests.get(video_url, timeout=60)
                        if video_response.status_code == 200 and len(video_response.content) > 0:
                            video_base64 = base64.b64encode(video_response.content).decode('utf-8')
                            return {
                                "video_data": video_base64,
                                "generation_time": time.time() - start_time,
                                "prompt_id": prompt_id,
                                "video_size": len(video_response.content),
                                "filename": video_info['filename']
                            }

        return {"error": "No valid images or videos found in ComfyUI output"}

    except Exception as e:
        return {"error": f"Failed to get workflow result: {str(e)}"}



@mcp.tool()
def search_website(search_term: str) -> Dict:
    """查询网站获取最新信息
    Args:
        search_term: 用户查询文本

    Returns:
        搜索结果字典
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
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            results = response.json()
            organic_results = results.get('organic_results', [])
            return {"search_result": organic_results}
        else:
            return {"search_result": f"Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"search_result": f"Error: {str(e)}"}

@mcp.tool()
def count_s3_buckets() -> int:
    """计算 S3 存储桶的数量"""
    try:
        s3 = boto3.client('s3')
        response = s3.list_buckets()
        return len(response['Buckets'])
    except Exception as e:
        return f"Error counting S3 buckets: {str(e)}"

@mcp.tool()
def generate_image_with_context(
    prompt: str,
    context_image_base64: str = None,
    workflow_type: str = "text_to_image",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1
) -> Dict:
    """使用 ComfyUI 生成图像，支持可选的上下文图像

    Args:
        prompt: 图像生成的文本描述
        context_image_base64: Base64 编码的上下文图像（可选）
        workflow_type: 工作流类型 (text_to_image, image_to_image, inpainting)
        width: 图像宽度（默认：1024）
        height: 图像高度（默认：1024）
        steps: 采样步数（默认：20）
        cfg_scale: CFG 引导比例（默认：7.0）
        seed: 随机种子（-1 为随机，默认：-1）

    Returns:
        生成的图像数据和元数据
    """
    try:
        # 验证输入
        if not prompt or not prompt.strip():
            return {"error": "Prompt cannot be empty"}

        if width <= 0 or height <= 0:
            return {"error": "Width and height must be positive"}

        if seed == -1:
            seed = generate_seed()

        # 检查 ComfyUI 连接
        if not test_comfyui_connectivity():
            return {"error": "ComfyUI server is not accessible"}

        # 使用与原始项目相同的 FLUX 文本到图像工作流
        workflow = create_flux_t2i_workflow(prompt, width, height, steps, cfg_scale, seed)

        # 提交工作流
        result = submit_comfyui_workflow(workflow)

        if result.get('error'):
            return {"error": result['error']}

        return {
            'type': 'image',
            'data': f"data:image/png;base64,{result['image_data']}",
            'mimeType': 'image/png',
            'metadata': {
                'prompt': prompt,
                'workflow_type': workflow_type,
                'dimensions': f"{width}x{height}",
                'steps': steps,
                'cfg_scale': cfg_scale,
                'seed': seed,
                'generation_time': result.get('generation_time', 0),
                'image_size': result.get('image_size', 0),
                'server_url': COMFYUI_SERVER_URL
            }
        }

    except Exception as e:
        return {"error": f"Image generation failed: {str(e)}"}

@mcp.tool()
def get_comfyui_config() -> Dict:
    """获取 ComfyUI 配置和可用工作流

    Returns:
        配置信息，包括可用工作流和预设
    """
    return {
        'server_url': COMFYUI_SERVER_URL,
        'available_workflows': ['text_to_image', 'image_to_image', 'text_to_video', 'image_to_video'],
        'workflow_presets': {
            'fast': {'steps': 15, 'cfg_scale': 6.0},
            'balanced': {'steps': 20, 'cfg_scale': 7.0},
            'quality': {'steps': 30, 'cfg_scale': 8.0}
        },
        'optimal_dimensions': {
            'square': (1024, 1024),
            'portrait': (768, 1024),
            'landscape': (1024, 768),
            'wide': (1152, 768),
            'tall': (768, 1152)
        },
        'config': {
            'timeout': COMFYUI_TIMEOUT,
            'poll_interval': COMFYUI_POLL_INTERVAL,
            'max_retries': COMFYUI_MAX_RETRIES
        },
        'connectivity': {
            'server_accessible': test_comfyui_connectivity(),
            'last_check': time.time()
        }
    }

@mcp.tool()
def generate_video_with_context(
    prompt: str,
    context_image_base64: str = None,
    workflow_type: str = "text_to_video",
    steps: int = 15,
    cfg_scale: float = 6.0,
    seed: int = -1,
    frame_rate: int = 16
) -> Dict:
    """使用 ComfyUI 生成视频，支持可选的上下文图像

    Args:
        prompt: 视频生成的文本描述
        context_image_base64: Base64 编码的上下文图像（image_to_video 需要）
        workflow_type: 工作流类型 (text_to_video, image_to_video)
        steps: 采样步数（默认：15）
        cfg_scale: CFG 引导比例（默认：6.0）
        seed: 随机种子（-1 为随机，默认：-1）
        frame_rate: 视频帧率（默认：16）

    Returns:
        生成的视频数据和元数据
    """
    try:
        # 验证输入
        if not prompt or not prompt.strip():
            return {"error": "Prompt cannot be empty"}

        if workflow_type == "image_to_video" and not context_image_base64:
            return {"error": "image_to_video 工作流需要 context_image_base64 参数"}

        if seed == -1:
            seed = generate_seed()

        # 检查 ComfyUI 连接
        if not test_comfyui_connectivity():
            return {"error": "ComfyUI server is not accessible"}

        # 使用与原始项目相同的 WanVideo 文本到视频工作流
        workflow = create_wanvideo_t2v_workflow(prompt, steps, cfg_scale, seed, frame_rate)

        # 如果有上下文图像，添加图像输入节点
        if context_image_base64 and workflow_type == "image_to_video":
            try:
                _, base64_data = validate_image_data(context_image_base64)
                workflow["6"] = {
                    "inputs": {
                        "image": base64_data
                    },
                    "class_type": "LoadImageBase64"
                }
                # 修改采样器以使用图像输入
                workflow["3"]["inputs"]["image"] = ["6", 0]
            except Exception as e:
                return {"error": f"Invalid context image: {str(e)}"}

        # 提交工作流
        result = submit_comfyui_workflow(workflow)

        if result.get('error'):
            return {"error": result['error']}

        # 检查是否获得了视频数据
        if 'video_data' in result:
            return {
                'type': 'video',
                'data': f"data:video/mp4;base64,{result['video_data']}",
                'mimeType': 'video/mp4',
                'metadata': {
                    'prompt': prompt,
                    'workflow_type': workflow_type,
                    'steps': steps,
                    'cfg_scale': cfg_scale,
                    'seed': seed,
                    'frame_rate': frame_rate,
                    'generation_time': result.get('generation_time', 0),
                    'video_size': result.get('video_size', 0),
                    'filename': result.get('filename', ''),
                    'server_url': COMFYUI_SERVER_URL
                }
            }
        else:
            return {"error": "No video data received from ComfyUI"}

    except Exception as e:
        return {"error": f"Video generation failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
