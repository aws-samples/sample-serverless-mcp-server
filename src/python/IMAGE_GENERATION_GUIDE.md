# ComfyUI MCP Server 使用指南

## 概述

本项目提供了两种 MCP Server 实现：
1. **HTTP MCP Server** - 基于 Starlette 的 Web 服务器
2. **STDIO MCP Server** - 基于标准输入输出的命令行服务器

两种实现都支持通过 ComfyUI 进行图像和视频生成，包括文生图、图生图、文生视频、图生视频等功能。

## STDIO MCP Server (推荐)

### 快速开始

STDIO MCP Server (`mcp_server_stdio_sample.py`) 是一个基于 FastMCP 库的轻量级命令行 MCP 服务器，支持本地文件路径操作。

#### 启动服务器
```bash
cd src/python
python mcp_server_stdio_sample.py
```

#### 与 AWS Strands Agent 集成
```python
from strands import Agent
from strands.tools.mcp import MCPTool

# 创建 MCP 工具 (使用 FastMCP)
mcp_tool = MCPTool(
    name="comfyui_generator",
    command=["python", "mcp_server_stdio_sample.py"],
    description="ComfyUI image and video generation tools built with FastMCP"
)

# 初始化 Agent
agent = Agent(
    name="ComfyUI Generator Agent",
    tools=[mcp_tool]
)
```

### FastMCP 特性

- **简化开发**: 使用装饰器定义工具，代码更简洁
- **自动类型检查**: 基于函数签名自动生成工具模式
- **内置错误处理**: 自动处理异常和错误响应
- **标准兼容**: 完全兼容 MCP 协议规范

## 支持的工具

### STDIO MCP Server 工具

#### 1. `generate_image_from_text` - 文生图
- **功能**: 从文本描述生成图像
- **参数**:
  - `prompt` (必需): 文本描述
  - `output_path` (必需): 输出图像文件路径
  - `width` (可选): 图像宽度 (默认: 1024)
  - `height` (可选): 图像高度 (默认: 1024)
  - `steps` (可选): 采样步数 (默认: 20)
  - `cfg_scale` (可选): CFG 引导强度 (默认: 7.0)
  - `seed` (可选): 随机种子 (默认: -1)
  - `negative_prompt` (可选): 负面提示词

#### 2. `generate_image_from_image` - 图生图
- **功能**: 从输入图像和文本描述生成新图像
- **参数**:
  - `prompt` (必需): 文本描述
  - `input_image_path` (必需): 输入图像文件路径
  - `output_path` (必需): 输出图像文件路径
  - `steps` (可选): 采样步数 (默认: 20)
  - `cfg_scale` (可选): CFG 引导强度 (默认: 7.0)
  - `seed` (可选): 随机种子 (默认: -1)
  - `denoise_strength` (可选): 去噪强度 (默认: 0.75)
  - `negative_prompt` (可选): 负面提示词

#### 3. `generate_video_from_text` - 文生视频
- **功能**: 从文本描述生成视频
- **参数**:
  - `prompt` (必需): 文本描述
  - `output_path` (必需): 输出视频文件路径
  - `steps` (可选): 采样步数 (默认: 15)
  - `cfg_scale` (可选): CFG 引导强度 (默认: 6.0)
  - `seed` (可选): 随机种子 (默认: -1)
  - `frame_rate` (可选): 视频帧率 (默认: 16)
  - `negative_prompt` (可选): 负面提示词

#### 4. `generate_video_from_image` - 图生视频
- **功能**: 从输入图像和文本描述生成视频
- **参数**:
  - `prompt` (必需): 文本描述
  - `input_image_path` (必需): 输入图像文件路径
  - `output_path` (必需): 输出视频文件路径
  - `steps` (可选): 采样步数 (默认: 15)
  - `cfg_scale` (可选): CFG 引导强度 (默认: 6.0)
  - `seed` (可选): 随机种子 (默认: -1)
  - `frame_rate` (可选): 视频帧率 (默认: 16)
  - `negative_prompt` (可选): 负面提示词

### HTTP MCP Server 工具 (传统)

#### 1. `generateImageWithContext` - 图像生成
- **功能**: 文生图 / 图生图 (通过 ComfyUI)
- **参数**:
  - `prompt` (必需): 文本描述
  - `context_image_base64` (可选): 上下文图像 (data URL 格式)
  - `workflow_type` (可选): 工作流类型
  - `width`, `height`, `steps`, `cfg_scale`, `seed` 等

## 使用示例

### 基本用法

```python
# 文生图
response = await agent.run("""
Generate a beautiful sunset landscape image.
Save it to './outputs/sunset.png'
""")

# 图生图
response = await agent.run("""
Transform the image at './inputs/photo.jpg' into an oil painting style.
Save the result to './outputs/painting.png'
""")

# 文生视频
response = await agent.run("""
Create a video of ocean waves at sunset.
Save it to './outputs/waves.mp4'
""")

# 图生视频
response = await agent.run("""
Animate the image at './inputs/landscape.jpg' with moving clouds.
Save the video to './outputs/animated_landscape.mp4'
""")
```

### 高级参数控制

```python
response = await agent.run("""
Generate a high-quality portrait with these settings:
- Prompt: "Professional headshot of a business person"
- Output: './outputs/portrait.png'
- Size: 768x1024 (portrait orientation)
- Steps: 30 (high quality)
- CFG Scale: 8.0 (strong prompt adherence)
- Seed: 12345 (reproducible results)
""")
```

## 支持的模型和工作流

### ComfyUI 工作流
- **flux_t2i.json**: 文本生成图像 (Flux 模型)
- **flux_kontext.json**: 图像到图像转换 (Flux 模型)
- **wanv_t2v.json**: 文本生成视频 (Wan Video 模型)
- **wan_i2v.json**: 图像生成视频 (Wan Video 模型)

### 文件格式支持
- **输入图像**: JPG, PNG, GIF, BMP, WebP
- **输出图像**: PNG (默认)
- **输出视频**: MP4

## 最佳实践

### 性能优化
- 使用较低的步数 (15-20) 进行快速预览
- 使用较高的步数 (25-30) 获得最佳质量
- 视频生成建议使用 15 步和 16 fps

### 提示词建议
- 使用具体、描述性的语言
- 包含风格、光照、构图等细节
- 使用负面提示词排除不需要的元素

### 文件管理
- 使用有意义的文件名
- 创建输出目录结构
- 定期清理临时文件

## 部署配置

### 环境变量
```bash
# ComfyUI 服务器配置
export COMFYUI_SERVER_URL="http://localhost:8188"
export COMFYUI_ENABLE_FALLBACK="true"
export COMFYUI_TIMEOUT="300"
export COMFYUI_MAX_RETRIES="3"
```

### ComfyUI 服务器要求
1. **ComfyUI 服务器**: 运行在 localhost:8188
2. **必需模型**:
   - Flux 模型 (用于图像生成)
   - Wan Video 模型 (用于视频生成)
3. **工作流文件**: 确保 `workflows/` 目录包含所需的 JSON 工作流文件

### 依赖安装
```bash
cd src/python
pip install -r requirements.txt
```

## 测试

### 运行 Jupyter Notebook 示例
```bash
cd src/python
jupyter notebook aws_strands_agent_mcp_example.ipynb
```

### 命令行测试
```bash
# 启动 STDIO MCP Server
python mcp_server_stdio_sample.py

# 在另一个终端测试
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python mcp_server_stdio_sample.py
```

### 集成测试示例
```python
from strands import Agent
from strands.tools.mcp import MCPTool

# 创建 MCP 工具
mcp_tool = MCPTool(
    name="comfyui_generator",
    command=["python", "mcp_server_stdio_sample.py"]
)

# 测试文生图
agent = Agent(tools=[mcp_tool])
response = await agent.run("""
Generate a test image with prompt "Hello World"
and save to "./test_output.png"
""")
```

## 错误处理

### 常见错误
1. **Invalid image format**: 图像格式不正确，需要 data URL 格式
2. **Model not found**: 指定的模型不存在或无权限访问
3. **API quota exceeded**: Bedrock API 配额超限

### 降级机制
当 Bedrock API 不可用时，系统会：
1. 返回模拟图像 (用于测试)
2. 在 metadata 中标记 "mock" 状态
3. 记录错误日志便于调试

## 性能优化

### 图像大小限制
- 建议输入图像不超过 5MB
- 输出图像通常为 PNG 格式
- 使用适当的尺寸以平衡质量和性能

### 缓存策略
考虑实现：
- S3 缓存生成的图像
- DynamoDB 缓存分析结果
- CloudFront 分发图像内容

## 安全考虑

### 输入验证
- 验证图像格式和大小
- 检查 base64 编码有效性
- 过滤恶意提示词

### 访问控制
- 使用 API Gateway 认证
- 实现用户配额限制
- 监控异常使用模式
