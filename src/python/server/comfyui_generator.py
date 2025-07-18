"""
ComfyUI Generator Class for Image Generation

This module provides a clean interface for ComfyUI operations, supporting
text-to-image and image-to-image workflows loaded from JSON files.
"""

import os
import time
import uuid
import base64
import requests
from typing import Dict, Optional
import json


class ComfyUIConfig:
    """Configuration class for ComfyUI settings"""

    def __init__(self):
        self.server_url = os.environ.get('COMFYUI_SERVER_URL', 'http://localhost:8188')
        self.timeout = int(os.environ.get('COMFYUI_TIMEOUT', '300'))  # 5 minutes
        self.poll_interval = int(os.environ.get('COMFYUI_POLL_INTERVAL', '2'))  # 2 seconds
        self.max_retries = int(os.environ.get('COMFYUI_MAX_RETRIES', '3'))
        self.request_timeout = int(os.environ.get('COMFYUI_REQUEST_TIMEOUT', '30'))
        self.enable_fallback = os.environ.get('COMFYUI_ENABLE_FALLBACK', 'true').lower() == 'true'
        self.workflow_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'workflows')


class WorkflowLoader:
    """Load and manage ComfyUI workflows from JSON files"""

    def __init__(self, workflow_dir: str):
        self.workflow_dir = workflow_dir
        self._workflows = {}

    def load_workflow(self, workflow_name: str) -> Dict:
        """Load workflow from JSON file"""
        if workflow_name in self._workflows:
            return self._workflows[workflow_name]

        workflow_file = os.path.join(self.workflow_dir, f"{workflow_name}.json")

        if not os.path.exists(workflow_file):
            raise FileNotFoundError(f"Workflow file not found: {workflow_file}")

        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow = json.load(f)

            self._workflows[workflow_name] = workflow
            return workflow

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workflow file {workflow_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow {workflow_file}: {e}")

    def update_workflow_parameters(self, workflow: Dict, parameters: Dict, workflow_type: str = "text_to_image") -> Dict:
        """Update workflow parameters dynamically based on workflow type"""
        # Create a deep copy to avoid modifying the original
        import copy
        updated_workflow = copy.deepcopy(workflow)

        if workflow_type == "text_to_image":
            # flux_t2i.json specific updates
            self._update_flux_t2i_parameters(updated_workflow, parameters)
        elif workflow_type == "image_to_image":
            # flux_kontext.json specific updates
            self._update_flux_kontext_parameters(updated_workflow, parameters)
        elif workflow_type == "text_to_video":
            # wanv_t2v.json specific updates
            self._update_wan_t2v_parameters(updated_workflow, parameters)
        elif workflow_type == "image_to_video":
            # wan_i2v.json specific updates
            self._update_wan_i2v_parameters(updated_workflow, parameters)

        return updated_workflow

    def _update_flux_t2i_parameters(self, workflow: Dict, parameters: Dict):
        """Update parameters for flux_t2i workflow"""
        # Node 6: Positive prompt (CLIPTextEncode)
        if "6" in workflow and "prompt" in parameters:
            workflow["6"]["inputs"]["text"] = parameters["prompt"]

        # Node 33: Negative prompt (CLIPTextEncode)
        if "33" in workflow and "negative_prompt" in parameters:
            workflow["33"]["inputs"]["text"] = parameters["negative_prompt"]

        # Node 31: KSampler parameters
        if "31" in workflow:
            ksampler_inputs = workflow["31"]["inputs"]
            if "seed" in parameters:
                ksampler_inputs["seed"] = parameters["seed"]
            if "steps" in parameters:
                ksampler_inputs["steps"] = parameters["steps"]
            if "cfg" in parameters:
                ksampler_inputs["cfg"] = parameters["cfg"]
            if "denoise" in parameters:
                ksampler_inputs["denoise"] = parameters["denoise"]

        # Node 42: EmptyLatentImage dimensions
        if "42" in workflow:
            latent_inputs = workflow["42"]["inputs"]
            if "width" in parameters:
                latent_inputs["width"] = parameters["width"]
            if "height" in parameters:
                latent_inputs["height"] = parameters["height"]

    def _update_flux_kontext_parameters(self, workflow: Dict, parameters: Dict):
        """Update parameters for flux_kontext workflow"""
        # Node 196: Text Multiline (prompt input)
        if "196" in workflow and "prompt" in parameters:
            workflow["196"]["inputs"]["text"] = parameters["prompt"]

        # Node 197: Image input (ETN_LoadImageBase64)
        if "197" in workflow and "image" in parameters:
            workflow["197"]["inputs"]["image"] = parameters["image"]

        # Node 31: KSampler parameters
        if "31" in workflow:
            ksampler_inputs = workflow["31"]["inputs"]
            if "seed" in parameters:
                ksampler_inputs["seed"] = parameters["seed"]
            if "steps" in parameters:
                ksampler_inputs["steps"] = parameters["steps"]
            if "cfg" in parameters:
                ksampler_inputs["cfg"] = parameters["cfg"]
            if "denoise" in parameters:
                ksampler_inputs["denoise"] = parameters["denoise"]

        # Node 35: FluxGuidance
        if "35" in workflow and "guidance" in parameters:
            workflow["35"]["inputs"]["guidance"] = parameters["guidance"]

    def _update_wan_t2v_parameters(self, workflow: Dict, parameters: Dict):
        """Update parameters for wanv_t2v workflow (text to video)"""
        # Node 16: WanVideoTextEncode (text prompt input)
        if "16" in workflow:
            if "prompt" in parameters:
                workflow["16"]["widgets_values"][0] = parameters["prompt"]
            if "negative_prompt" in parameters:
                workflow["16"]["widgets_values"][1] = parameters["negative_prompt"]

        # Node 27: WanVideoSampler parameters
        if "27" in workflow:
            sampler_widgets = workflow["27"]["widgets_values"]
            if "steps" in parameters:
                sampler_widgets[0] = parameters["steps"]  # steps
            if "cfg" in parameters:
                sampler_widgets[1] = parameters["cfg"]  # cfg
            if "seed" in parameters:
                sampler_widgets[3] = parameters["seed"]  # seed

        # Node 30: VHS_VideoCombine (video output settings)
        if "30" in workflow and "frame_rate" in parameters:
            workflow["30"]["widgets_values"]["frame_rate"] = parameters["frame_rate"]

    def _update_wan_i2v_parameters(self, workflow: Dict, parameters: Dict):
        """Update parameters for wan_i2v workflow (image to video)"""
        # Node 16: WanVideoTextEncode (text prompt input)
        if "16" in workflow:
            if "prompt" in parameters:
                workflow["16"]["widgets_values"][0] = parameters["prompt"]
            if "negative_prompt" in parameters:
                workflow["16"]["widgets_values"][1] = parameters["negative_prompt"]

        # Node 18: LoadImage (input image)
        if "18" in workflow and "image_filename" in parameters:
            workflow["18"]["widgets_values"][0] = parameters["image_filename"]

        # Node 27: WanVideoSampler parameters
        if "27" in workflow:
            sampler_widgets = workflow["27"]["widgets_values"]
            if "steps" in parameters:
                sampler_widgets[0] = parameters["steps"]  # steps
            if "cfg" in parameters:
                sampler_widgets[1] = parameters["cfg"]  # cfg
            if "seed" in parameters:
                sampler_widgets[3] = parameters["seed"]  # seed

        # Node 30: VHS_VideoCombine (video output settings) - first occurrence
        if "30" in workflow and "frame_rate" in parameters:
            workflow["30"]["widgets_values"]["frame_rate"] = parameters["frame_rate"]

        # Node 38: VHS_VideoCombine (video output settings) - second occurrence
        if "38" in workflow and "frame_rate" in parameters:
            workflow["38"]["widgets_values"]["frame_rate"] = parameters["frame_rate"]



class ComfyUIGenerator:
    """Main ComfyUI generator class for image operations"""

    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self.workflow_loader = WorkflowLoader(self.config.workflow_dir)
    
    def generate_seed(self) -> int:
        """Generate a random seed"""
        return int(time.time() * 1000) % 2147483647
    
    def validate_image_data(self, image_data: str) -> tuple[str, str]:
        """Validate and extract image data from data URL format"""
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

    def submit_workflow(self, workflow: Dict) -> Dict:
        """Submit workflow to ComfyUI server and wait for result with enhanced error handling"""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Test server connectivity first
                if not self._test_server_connectivity():
                    last_error = "ComfyUI server is not accessible"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    break

                # Generate unique prompt ID
                prompt_id = str(uuid.uuid4())

                # Submit workflow
                submit_url = f"{self.config.server_url}/prompt"
                submit_payload = {
                    "prompt": workflow,
                    "client_id": prompt_id
                }

                response = requests.post(
                    submit_url,
                    json=submit_payload,
                    timeout=self.config.request_timeout
                )

                if response.status_code != 200:
                    last_error = f"Failed to submit workflow (HTTP {response.status_code}): {response.text}"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    break

                submit_result = response.json()
                if not submit_result.get("prompt_id"):
                    last_error = "No prompt_id returned from ComfyUI"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    break

                actual_prompt_id = submit_result["prompt_id"]

                # Poll for completion with enhanced error handling
                result = self._poll_for_completion(actual_prompt_id)
                if result.get('error'):
                    last_error = result['error']
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    break

                return result

            except requests.exceptions.Timeout:
                last_error = f"Request timeout after {self.config.request_timeout} seconds"
            except requests.exceptions.ConnectionError:
                last_error = "Failed to connect to ComfyUI server"
            except requests.exceptions.RequestException as e:
                last_error = f"Network error: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"

            if attempt < self.config.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        return {"error": f"ComfyUI workflow submission failed after {self.config.max_retries} attempts: {last_error}"}

    def _test_server_connectivity(self) -> bool:
        """Test if ComfyUI server is accessible"""
        try:
            response = requests.get(
                f"{self.config.server_url}/queue",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def _poll_for_completion(self, prompt_id: str) -> Dict:
        """Poll for workflow completion with enhanced error handling"""
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 5

        while time.time() - start_time < self.config.timeout:
            try:
                # Check queue status
                queue_url = f"{self.config.server_url}/queue"
                queue_response = requests.get(
                    queue_url,
                    timeout=self.config.request_timeout
                )

                if queue_response.status_code != 200:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        return {"error": f"Too many consecutive queue check failures"}
                    time.sleep(self.config.poll_interval)
                    continue

                consecutive_errors = 0  # Reset on success
                queue_data = queue_response.json()

                # Check if our prompt is still in queue
                running = queue_data.get("queue_running", [])
                pending = queue_data.get("queue_pending", [])

                prompt_in_queue = any(item[1] == prompt_id for item in running + pending)

                if not prompt_in_queue:
                    # Prompt completed, get result
                    return self._get_workflow_result(prompt_id, start_time)

                time.sleep(self.config.poll_interval)

            except requests.exceptions.Timeout:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    return {"error": "Too many timeout errors during polling"}
                time.sleep(self.config.poll_interval)
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    return {"error": f"Too many polling errors: {str(e)}"}
                time.sleep(self.config.poll_interval)

        return {"error": f"ComfyUI generation timeout after {self.config.timeout} seconds"}

    def _get_workflow_result(self, prompt_id: str, start_time: float) -> Dict:
        """Get the result of a completed workflow with enhanced error handling"""
        try:
            # Get workflow history
            history_url = f"{self.config.server_url}/history/{prompt_id}"
            history_response = requests.get(
                history_url,
                timeout=self.config.request_timeout
            )

            if history_response.status_code != 200:
                return {"error": f"Failed to get ComfyUI history (HTTP {history_response.status_code})"}

            history_data = history_response.json()

            if prompt_id not in history_data:
                return {"error": "Prompt not found in history"}

            # Check for execution errors
            prompt_data = history_data[prompt_id]
            if "status" in prompt_data and prompt_data["status"].get("status_str") == "error":
                error_details = prompt_data["status"].get("messages", [])
                return {"error": f"ComfyUI execution error: {error_details}"}

            outputs = prompt_data.get("outputs", {})

            # Find the SaveImage or VHS_VideoCombine node output
            for node_id, node_output in outputs.items():
                # Handle image outputs
                if "images" in node_output:
                    images = node_output["images"]
                    if images:
                        # Get the first image
                        image_info = images[0]

                        # Validate image info
                        if not image_info.get('filename'):
                            continue

                        image_url = f"{self.config.server_url}/view?filename={image_info['filename']}&subfolder={image_info.get('subfolder', '')}&type={image_info.get('type', 'output')}"

                        # Download the image with retry logic
                        for attempt in range(3):
                            try:
                                image_response = requests.get(
                                    image_url,
                                    timeout=self.config.request_timeout
                                )

                                if image_response.status_code == 200:
                                    # Validate image content
                                    if len(image_response.content) == 0:
                                        if attempt < 2:
                                            time.sleep(1)
                                            continue
                                        return {"error": "Downloaded image is empty"}

                                    image_base64 = base64.b64encode(image_response.content).decode('utf-8')

                                    return {
                                        "image_data": image_base64,
                                        "generation_time": time.time() - start_time,
                                        "prompt_id": prompt_id,
                                        "image_size": len(image_response.content)
                                    }
                                else:
                                    if attempt < 2:
                                        time.sleep(1)
                                        continue
                                    return {"error": f"Failed to download image (HTTP {image_response.status_code})"}

                            except requests.exceptions.Timeout:
                                if attempt < 2:
                                    time.sleep(1)
                                    continue
                                return {"error": "Timeout downloading image"}
                            except Exception as e:
                                if attempt < 2:
                                    time.sleep(1)
                                    continue
                                return {"error": f"Error downloading image: {str(e)}"}

                # Handle video outputs
                elif "gifs" in node_output:
                    videos = node_output["gifs"]
                    if videos:
                        # Get the first video
                        video_info = videos[0]

                        # Validate video info
                        if not video_info.get('filename'):
                            continue

                        video_url = f"{self.config.server_url}/view?filename={video_info['filename']}&subfolder={video_info.get('subfolder', '')}&type={video_info.get('type', 'output')}"

                        # Download the video with retry logic
                        for attempt in range(3):
                            try:
                                video_response = requests.get(
                                    video_url,
                                    timeout=self.config.request_timeout * 2  # Videos may take longer
                                )

                                if video_response.status_code == 200:
                                    # Validate video content
                                    if len(video_response.content) == 0:
                                        if attempt < 2:
                                            time.sleep(1)
                                            continue
                                        return {"error": "Downloaded video is empty"}

                                    video_base64 = base64.b64encode(video_response.content).decode('utf-8')

                                    return {
                                        "video_data": video_base64,
                                        "generation_time": time.time() - start_time,
                                        "prompt_id": prompt_id,
                                        "video_size": len(video_response.content),
                                        "filename": video_info['filename']
                                    }
                                else:
                                    if attempt < 2:
                                        time.sleep(1)
                                        continue
                                    return {"error": f"Failed to download video (HTTP {video_response.status_code})"}

                            except requests.exceptions.Timeout:
                                if attempt < 2:
                                    time.sleep(1)
                                    continue
                                return {"error": "Timeout downloading video"}
                            except Exception as e:
                                if attempt < 2:
                                    time.sleep(1)
                                    continue
                                return {"error": f"Error downloading video: {str(e)}"}

            return {"error": "No valid images or videos found in ComfyUI output"}

        except requests.exceptions.Timeout:
            return {"error": "Timeout getting workflow result"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error getting result: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to get workflow result: {str(e)}"}

    def generate_text_to_image(self, prompt: str, width: int = 1024, height: int = 1024,
                              steps: int = 20, cfg_scale: float = 7.0, seed: int = -1,
                              negative_prompt: str = "bad quality, blurry, low resolution") -> Dict:
        """Generate image from text prompt using flux_t2i.json workflow"""
        try:
            # Validate inputs
            if not prompt or not prompt.strip():
                return {"error": "Prompt cannot be empty"}

            if width <= 0 or height <= 0:
                return {"error": "Width and height must be positive"}

            if seed == -1:
                seed = self.generate_seed()

            # Load workflow template
            workflow = self.workflow_loader.load_workflow("flux_t2i")

            # Update workflow parameters
            parameters = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg_scale,
                "seed": seed
            }

            updated_workflow = self.workflow_loader.update_workflow_parameters(workflow, parameters, "text_to_image")

            # Submit workflow
            result = self.submit_workflow(updated_workflow)

            if result.get('error'):
                if self.config.enable_fallback:
                    return self.get_mock_image_response(
                        operation="text_to_image",
                        prompt=prompt,
                        dimensions=f"{width}x{height}",
                        error=result['error']
                    )
                return {"error": result['error']}

            return {
                'type': 'image',
                'data': f"data:image/png;base64,{result['image_data']}",
                'mimeType': 'image/png',
                'metadata': {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'workflow_type': 'text_to_image',
                    'workflow_file': 'flux_t2i.json',
                    'dimensions': f"{width}x{height}",
                    'steps': steps,
                    'cfg_scale': cfg_scale,
                    'seed': seed,
                    'generation_time': result.get('generation_time', 0),
                    'image_size': result.get('image_size', 0)
                }
            }

        except FileNotFoundError:
            error_msg = "flux_t2i.json workflow file not found"
            if self.config.enable_fallback:
                return self.get_mock_image_response(
                    operation="text_to_image",
                    prompt=prompt,
                    dimensions=f"{width}x{height}",
                    error=error_msg
                )
            return {"error": error_msg}
        except Exception as e:
            if self.config.enable_fallback:
                return self.get_mock_image_response(
                    operation="text_to_image",
                    prompt=prompt,
                    dimensions=f"{width}x{height}",
                    error=str(e)
                )
            return {"error": f"Text-to-image generation failed: {str(e)}"}

    def generate_image_to_image(self, prompt: str, image_data: str,
                               steps: int = 20, cfg_scale: float = 7.0, seed: int = -1,
                               denoise_strength: float = 0.75,
                               negative_prompt: str = "bad quality, blurry, low resolution") -> Dict:
        """Generate image from image and text prompt using flux_kontext.json workflow"""
        try:
            # Validate input image
            _, base64_data = self.validate_image_data(image_data)

            if seed == -1:
                seed = self.generate_seed()

            # Load workflow template
            workflow = self.workflow_loader.load_workflow("flux_kontext")

            # Update workflow parameters
            parameters = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": base64_data,  # Input image data
                "steps": steps,
                "cfg": cfg_scale,
                "seed": seed,
                "denoise": denoise_strength
            }

            updated_workflow = self.workflow_loader.update_workflow_parameters(workflow, parameters, "image_to_image")

            # Submit workflow
            result = self.submit_workflow(updated_workflow)

            if result.get('error'):
                if self.config.enable_fallback:
                    return self.get_mock_image_response(
                        operation="image_to_image",
                        prompt=prompt,
                        error=result['error']
                    )
                return {"error": result['error']}

            return {
                'type': 'image',
                'data': f"data:image/png;base64,{result['image_data']}",
                'mimeType': 'image/png',
                'metadata': {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'workflow_type': 'image_to_image',
                    'workflow_file': 'flux_kontext.json',
                    'steps': steps,
                    'cfg_scale': cfg_scale,
                    'seed': seed,
                    'denoise_strength': denoise_strength,
                    'generation_time': result.get('generation_time', 0),
                    'image_size': result.get('image_size', 0)
                }
            }

        except FileNotFoundError:
            error_msg = "flux_kontext.json workflow file not found"
            if self.config.enable_fallback:
                return self.get_mock_image_response(
                    operation="image_to_image",
                    prompt=prompt,
                    error=error_msg
                )
            return {"error": error_msg}
        except Exception as e:
            if self.config.enable_fallback:
                return self.get_mock_image_response(
                    operation="image_to_image",
                    prompt=prompt,
                    error=str(e)
                )
            return {"error": f"Image-to-image generation failed: {str(e)}"}

    def generate_text_to_video(self, prompt: str, steps: int = 15, cfg_scale: float = 6.0,
                              seed: int = -1, frame_rate: int = 16,
                              negative_prompt: str = "bad quality video, blurry, low resolution") -> Dict:
        """Generate video from text prompt using wanv_t2v.json workflow"""
        try:
            # Validate inputs
            if not prompt or not prompt.strip():
                return {"error": "Prompt cannot be empty"}

            if seed == -1:
                seed = self.generate_seed()

            # Load workflow template
            workflow = self.workflow_loader.load_workflow("wanv_t2v")

            # Update workflow parameters
            parameters = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg": cfg_scale,
                "seed": seed,
                "frame_rate": frame_rate
            }

            updated_workflow = self.workflow_loader.update_workflow_parameters(workflow, parameters, "text_to_video")

            # Submit workflow
            result = self.submit_workflow(updated_workflow)

            if result.get('error'):
                if self.config.enable_fallback:
                    return self.get_mock_video_response(
                        operation="text_to_video",
                        prompt=prompt,
                        error=result['error']
                    )
                return {"error": result['error']}

            # Check if we got video data
            if 'video_data' in result:
                return {
                    'type': 'video',
                    'data': f"data:video/mp4;base64,{result['video_data']}",
                    'mimeType': 'video/mp4',
                    'metadata': {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'workflow_type': 'text_to_video',
                        'workflow_file': 'wanv_t2v.json',
                        'steps': steps,
                        'cfg_scale': cfg_scale,
                        'seed': seed,
                        'frame_rate': frame_rate,
                        'generation_time': result.get('generation_time', 0),
                        'video_size': result.get('video_size', 0),
                        'filename': result.get('filename', '')
                    }
                }
            else:
                return {"error": "No video data received from ComfyUI"}

        except FileNotFoundError:
            error_msg = "wanv_t2v.json workflow file not found"
            if self.config.enable_fallback:
                return self.get_mock_video_response(
                    operation="text_to_video",
                    prompt=prompt,
                    error=error_msg
                )
            return {"error": error_msg}
        except Exception as e:
            if self.config.enable_fallback:
                return self.get_mock_video_response(
                    operation="text_to_video",
                    prompt=prompt,
                    error=str(e)
                )
            return {"error": f"Text-to-video generation failed: {str(e)}"}

    def generate_image_to_video(self, prompt: str, image_data: str, steps: int = 15,
                               cfg_scale: float = 6.0, seed: int = -1, frame_rate: int = 16,
                               negative_prompt: str = "bad quality video, blurry, low resolution") -> Dict:
        """Generate video from image and text prompt using wan_i2v.json workflow"""
        try:
            # Validate input image
            _, base64_data = self.validate_image_data(image_data)

            if seed == -1:
                seed = self.generate_seed()

            # Load workflow template
            workflow = self.workflow_loader.load_workflow("wan_i2v")

            # For image-to-video, we need to reference the image filename
            # In a real implementation, you would upload the image to ComfyUI's input directory
            # For now, we'll use a placeholder filename based on the image data
            temp_filename = f"temp_input_{int(time.time())}.png"

            # Update workflow parameters
            parameters = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_filename": temp_filename,  # Reference to the uploaded image
                "image_data": base64_data,  # Store the image data for potential use
                "steps": steps,
                "cfg": cfg_scale,
                "seed": seed,
                "frame_rate": frame_rate
            }

            updated_workflow = self.workflow_loader.update_workflow_parameters(workflow, parameters, "image_to_video")

            # Submit workflow
            result = self.submit_workflow(updated_workflow)

            if result.get('error'):
                if self.config.enable_fallback:
                    return self.get_mock_video_response(
                        operation="image_to_video",
                        prompt=prompt,
                        error=result['error']
                    )
                return {"error": result['error']}

            # Check if we got video data
            if 'video_data' in result:
                return {
                    'type': 'video',
                    'data': f"data:video/mp4;base64,{result['video_data']}",
                    'mimeType': 'video/mp4',
                    'metadata': {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'workflow_type': 'image_to_video',
                        'workflow_file': 'wan_i2v.json',
                        'steps': steps,
                        'cfg_scale': cfg_scale,
                        'seed': seed,
                        'frame_rate': frame_rate,
                        'generation_time': result.get('generation_time', 0),
                        'video_size': result.get('video_size', 0),
                        'filename': result.get('filename', '')
                    }
                }
            else:
                return {"error": "No video data received from ComfyUI"}

        except FileNotFoundError:
            error_msg = "wan_i2v.json workflow file not found"
            if self.config.enable_fallback:
                return self.get_mock_video_response(
                    operation="image_to_video",
                    prompt=prompt,
                    error=error_msg
                )
            return {"error": error_msg}
        except Exception as e:
            if self.config.enable_fallback:
                return self.get_mock_video_response(
                    operation="image_to_video",
                    prompt=prompt,
                    error=str(e)
                )
            return {"error": f"Image-to-video generation failed: {str(e)}"}



    def get_mock_image_response(self, operation: str, **metadata) -> Dict:
        """Generate a mock image response for testing/fallback"""
        mock_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        return {
            'type': 'image',
            'data': f"data:image/png;base64,{mock_image_base64}",
            'mimeType': 'image/png',
            'metadata': {
                'operation': f"{operation} (mock)",
                'note': 'Mock image due to ComfyUI unavailability',
                'server_url': self.config.server_url,
                'fallback_enabled': self.config.enable_fallback,
                **metadata
            }
        }

    def get_mock_video_response(self, operation: str, **metadata) -> Dict:
        """Generate a mock video response for testing/fallback"""
        # A minimal MP4 video (1 frame, black screen) encoded in base64
        mock_video_base64 = "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"

        return {
            'type': 'video',
            'data': f"data:video/mp4;base64,{mock_video_base64}",
            'mimeType': 'video/mp4',
            'metadata': {
                'operation': f"{operation} (mock)",
                'note': 'Mock video due to ComfyUI unavailability',
                'server_url': self.config.server_url,
                'fallback_enabled': self.config.enable_fallback,
                **metadata
            }
        }


class ConfigManager:
    """Configuration management for ComfyUI settings"""

    @staticmethod
    def get_workflow_presets() -> Dict[str, Dict]:
        """Get predefined workflow presets for Flux models"""
        return {
            'fast': {
                'steps': 15,
                'cfg_scale': 6.0,
                'description': 'Fast generation with Flux'
            },
            'balanced': {
                'steps': 20,
                'cfg_scale': 7.0,
                'description': 'Balanced speed and quality with Flux'
            },
            'quality': {
                'steps': 30,
                'cfg_scale': 8.0,
                'description': 'High quality with Flux'
            }
        }

    @staticmethod
    def get_optimal_dimensions(aspect_ratio: str = "square") -> tuple[int, int]:
        """Get optimal dimensions for different aspect ratios"""
        dimensions = {
            'square': (1024, 1024),
            'portrait': (768, 1024),
            'landscape': (1024, 768),
            'wide': (1152, 768),
            'tall': (768, 1152)
        }
        return dimensions.get(aspect_ratio, (1024, 1024))

    @staticmethod
    def get_available_workflows() -> list[str]:
        """Get list of available workflow files"""
        return ['flux_t2i', 'flux_kontext', 'wanv_t2v', 'wan_i2v']


class ImageUtils:
    """Utility functions for image processing"""

    @staticmethod
    def get_image_dimensions(image_bytes: bytes) -> tuple[int, int]:
        """Get image dimensions from image bytes"""
        try:
            # PNG format
            if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                # PNG IHDR chunk starts at byte 16
                if len(image_bytes) >= 24:
                    width = int.from_bytes(image_bytes[16:20], 'big')
                    height = int.from_bytes(image_bytes[20:24], 'big')
                    return width, height

            # JPEG format
            elif image_bytes.startswith(b'\xff\xd8'):
                # Simple JPEG dimension parsing
                i = 2
                while i < len(image_bytes) - 8:
                    if image_bytes[i:i+2] == b'\xff\xc0':  # SOF0 marker
                        height = int.from_bytes(image_bytes[i+5:i+7], 'big')
                        width = int.from_bytes(image_bytes[i+7:i+9], 'big')
                        return width, height
                    i += 1

            # Default fallback
            return 1024, 1024

        except Exception:
            # Fallback dimensions
            return 1024, 1024

    @staticmethod
    def analyze_image_basic(image_data: str) -> Dict:
        """Perform basic image analysis"""
        try:
            if not image_data.startswith('data:image/'):
                return {"error": "Invalid image format. Expected data URL format"}

            header, data = image_data.split(',', 1)
            image_bytes = base64.b64decode(data)
            mime_type = header.split(';')[0].split(':')[1]

            # Basic image analysis
            width, height = ImageUtils.get_image_dimensions(image_bytes)

            analysis = f"Image Analysis:\n"
            analysis += f"- Format: {mime_type}\n"
            analysis += f"- Dimensions: {width}x{height} pixels\n"
            analysis += f"- File size: {len(image_bytes)} bytes\n"
            analysis += f"- Aspect ratio: {width/height:.2f}\n"

            if width > height:
                analysis += "- Orientation: Landscape\n"
            elif height > width:
                analysis += "- Orientation: Portrait\n"
            else:
                analysis += "- Orientation: Square\n"

            # Add resolution category
            total_pixels = width * height
            if total_pixels > 2000000:  # > 2MP
                analysis += "- Resolution: High resolution\n"
            elif total_pixels > 500000:  # > 0.5MP
                analysis += "- Resolution: Medium resolution\n"
            else:
                analysis += "- Resolution: Low resolution\n"

            return {
                'analysis': analysis,
                'image_info': {
                    'mime_type': mime_type,
                    'width': width,
                    'height': height,
                    'size_bytes': len(image_bytes),
                    'data_url_length': len(image_data),
                    'total_pixels': total_pixels
                }
            }

        except Exception as e:
            return {
                'analysis': f"Basic image validation completed. Unable to perform detailed analysis: {str(e)}",
                'image_info': {
                    'mime_type': 'unknown',
                    'size_bytes': len(base64.b64decode(image_data.split(',')[1])) if ',' in image_data else 0,
                    'note': 'Fallback response due to analysis error'
                }
            }
