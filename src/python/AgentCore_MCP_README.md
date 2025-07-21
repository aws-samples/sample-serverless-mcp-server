# AWS Bedrock AgentCore MCP Server 部署示例

本示例展示了如何使用 AWS Bedrock AgentCore SDK 将现有的 MCP (Model Context Protocol) server 工具部署到 AgentCore Runtime。

## 概述

这个示例包含了将 `server/app.py` 中的 MCP server 工具适配并部署到 AWS Bedrock AgentCore Runtime 的完整流程，包括：

- 网站搜索工具 (`search_website`)
- S3 存储桶计数工具 (`count_s3_buckets`)
- 图像生成工具 (`generate_image_with_context`) - 支持真实的 ComfyUI 调用
- ComfyUI 配置获取工具 (`get_comfyui_config`)
- 视频生成工具 (`generate_video_with_context`) - 支持真实的 ComfyUI 调用

### 新增功能

- **真实 ComfyUI 集成**: 支持连接到真实的 ComfyUI 服务器进行图像和视频生成
- **环境变量配置**: 通过环境变量配置 ComfyUI 服务器 URL 和其他参数
- **智能回退机制**: 当 ComfyUI 不可用时自动回退到模拟响应
- **连接性检测**: 自动检测 ComfyUI 服务器连接状态

## 文件说明

### 核心文件

1. **`agentcore_hosting_mcp_sample.ipynb`** - 完整的 Jupyter notebook 教程
2. **`agentcore_mcp_server.py`** - 适配 AgentCore Runtime 的 MCP server
3. **`agentcore_deployment_script.py`** - 自动化部署脚本
4. **`requirements.txt`** - Python 依赖

### 支持文件

- **`AgentCore_MCP_README.md`** - 本文档

## 环境变量配置

在部署之前，可以配置以下环境变量来自定义 ComfyUI 集成：

```bash
# ComfyUI 服务器配置
export COMFYUI_SERVER_URL="http://your-comfyui-server:8188"  # 默认: http://localhost:8188
export COMFYUI_TIMEOUT="300"                                 # 生成超时时间（秒），默认: 300
export COMFYUI_POLL_INTERVAL="2"                            # 轮询间隔（秒），默认: 2
export COMFYUI_MAX_RETRIES="3"                              # 最大重试次数，默认: 3
export COMFYUI_ENABLE_FALLBACK="true"                       # 启用回退模式，默认: true

# SerpAPI 配置（用于网站搜索）
export SERPAPI_API_KEY="your_serpapi_key_here"
```

### ComfyUI 配置说明

- **COMFYUI_SERVER_URL**: ComfyUI 服务器的完整 URL，包括端口号
- **COMFYUI_TIMEOUT**: 单次生成任务的最大等待时间（秒）
- **COMFYUI_POLL_INTERVAL**: 检查任务完成状态的轮询间隔（秒）
- **COMFYUI_MAX_RETRIES**: 连接失败时的最大重试次数
- **COMFYUI_ENABLE_FALLBACK**: 当 ComfyUI 不可用时是否启用模拟响应

**重要说明**：
- 如果 ComfyUI 服务器不可访问且启用了回退模式，系统将返回模拟的图像/视频数据
- 如果禁用回退模式，当 ComfyUI 不可用时将返回错误信息
- 确保 ComfyUI 服务器可以从 AgentCore Runtime 的网络环境访问
- 对于生产环境，建议将 ComfyUI 部署在同一 VPC 内以确保网络连通性

## 前提条件

### 系统要求

- Python 3.10+
- Docker 运行环境
- AWS CLI 已配置
- 有效的 AWS 凭证

### AWS 权限

确保您的 AWS 凭证具有以下权限：

- Amazon Bedrock AgentCore 服务权限
- Amazon Cognito 用户池管理
- IAM 角色创建和管理
- Amazon ECR 仓库管理
- AWS Systems Manager Parameter Store
- AWS Secrets Manager
- Amazon S3 (用于 S3 工具)

### 安装依赖

```bash
pip install mcp bedrock-agentcore-starter-toolkit boto3 requests fastapi uvicorn starlette
```

## 快速开始

### 步骤 0: 配置环境变量（可选）

如果您有运行中的 ComfyUI 服务器，请先配置环境变量：

```bash
# 配置 ComfyUI 服务器 URL
export COMFYUI_SERVER_URL="http://your-comfyui-server:8188"

# 配置 SerpAPI 密钥（用于网站搜索）
export SERPAPI_API_KEY="your_serpapi_key_here"

# 其他可选配置
export COMFYUI_TIMEOUT="300"
export COMFYUI_ENABLE_FALLBACK="true"
```

### 方法 1: 使用 Jupyter Notebook（推荐）

1. 打开 `agentcore_hosting_mcp_sample.ipynb`
2. 按顺序执行所有单元格
3. 按照提示配置认证和部署
4. 环境变量将自动传递给 AgentCore Runtime

### 方法 2: 使用自动化脚本

1. 确保所有必需文件在当前目录中
2. 运行部署脚本：

```bash
python agentcore_deployment_script.py
```

3. 脚本将自动：
   - 设置 Cognito 用户池
   - 创建 IAM 执行角色
   - 配置 AgentCore Runtime
   - 部署 MCP server
   - 存储配置信息

### 方法 3: 手动步骤

1. **创建 MCP server 文件**：
   ```bash
   # agentcore_mcp_server.py 已经创建
   # requirements.txt 已经创建
   ```

2. **部署到 AgentCore**：
   ```python
   # 在 Python 脚本或 notebook 中执行
   from agentcore_deployment_script import deploy_mcp_server
   result = deploy_mcp_server()
   ```

3. **测试已部署的服务**：
   使用 notebook 中的测试代码单元格

## 配置说明

### 环境变量

- `SERPAPI_API_KEY`: SerpAPI 密钥（用于网站搜索功能）
- `AWS_REGION`: AWS 区域（可选，默认从 AWS 配置获取）

### Cognito 认证

脚本会自动创建：
- Cognito 用户池
- 应用客户端
- 测试用户（用户名: `testuser`, 密码: `MyPassword123!`）
- JWT 访问令牌

### IAM 角色

自动创建的 IAM 角色包含以下权限：
- CloudWatch Logs 写入
- S3 存储桶列表
- 其他 AgentCore Runtime 必需权限

## 工具功能

### 1. search_website
- **功能**: 使用 SerpAPI 进行网站搜索
- **参数**: `search_term` (字符串)
- **返回**: 搜索结果字典

### 2. count_s3_buckets
- **功能**: 计算 AWS S3 存储桶数量
- **参数**: 无
- **返回**: 存储桶数量（整数）

### 3. generate_image_with_context
- **功能**: 图像生成（支持真实 ComfyUI 集成）
- **参数**:
  - `prompt`: 文本描述
  - `context_image_base64`: 上下文图像（可选）
  - `workflow_type`: 工作流类型 (text_to_image, image_to_image, inpainting)
  - `width`, `height`: 图像尺寸
  - `steps`, `cfg_scale`, `seed`: 生成参数
- **返回**:
  - 成功时：Base64 编码的图像数据和元数据
  - 失败时：错误信息或模拟响应（如果启用回退）
- **特性**:
  - 自动检测 ComfyUI 服务器连接状态
  - 支持智能回退到模拟响应
  - 实时轮询生成进度
  - 完整的错误处理和重试机制

### 4. get_comfyui_config
- **功能**: 获取 ComfyUI 配置信息和连接状态
- **参数**: 无
- **返回**:
  - 服务器 URL 和连接状态
  - 可用工作流类型
  - 工作流预设参数
  - 推荐图像尺寸
  - 超时和重试配置
- **特性**:
  - 实时连接性检测
  - 配置参数验证

### 5. generate_video_with_context
- **功能**: 视频生成（支持真实 ComfyUI 集成）
- **参数**:
  - `prompt`: 文本描述
  - `context_image_base64`: 上下文图像（image_to_video 必需）
  - `workflow_type`: 工作流类型 (text_to_video, image_to_video)
  - `steps`, `cfg_scale`, `seed`, `frame_rate`: 生成参数
- **返回**:
  - 成功时：Base64 编码的视频数据和元数据
  - 失败时：错误信息或模拟响应（如果启用回退）
- **特性**:
  - 支持文本到视频和图像到视频转换
  - 自动处理视频编码和格式转换
  - 扩展超时处理（视频生成通常需要更长时间）

## 测试

### 测试已部署的 MCP Server

使用 notebook 中的测试代码单元格直接测试已部署的 MCP server。

### 使用 MCP Inspector（可选）

```bash
# 安装并启动 MCP Inspector
npx @modelcontextprotocol/inspector

# 使用 AgentCore endpoint URL 和 Bearer token 进行远程测试
```

## 故障排除

### 常见问题

1. **Docker 未运行**
   ```
   错误: Cannot connect to the Docker daemon
   解决: 启动 Docker Desktop 或 Docker 服务
   ```

2. **AWS 权限不足**
   ```
   错误: AccessDenied
   解决: 检查 AWS 凭证和权限
   ```

3. **依赖缺失**
   ```
   错误: ModuleNotFoundError
   解决: pip install -r requirements.txt
   ```

4. **区域配置问题**
   ```
   错误: Region not found
   解决: 设置 AWS_REGION 环境变量或配置 AWS CLI
   ```

5. **ComfyUI 连接问题**
   ```
   错误: ComfyUI server is not accessible
   解决:
   - 检查 COMFYUI_SERVER_URL 环境变量是否正确
   - 确认 ComfyUI 服务器正在运行
   - 验证网络连通性（ping 测试）
   - 检查防火墙和安全组设置
   ```

6. **ComfyUI 生成超时**
   ```
   错误: ComfyUI generation timeout after 300 seconds
   解决:
   - 增加 COMFYUI_TIMEOUT 环境变量值
   - 检查 ComfyUI 服务器性能和负载
   - 减少生成参数复杂度（降低 steps 或分辨率）
   ```

7. **ComfyUI 工作流错误**
   ```
   错误: ComfyUI execution error
   解决:
   - 检查 ComfyUI 服务器日志
   - 确认所需的模型文件已安装
   - 验证工作流节点配置
   - 检查输入参数格式和范围
   ```

### 调试技巧

1. **查看日志**：
   - AgentCore Runtime 日志在 CloudWatch 中
   - 本地测试时查看终端输出

2. **验证配置**：
   ```python
   # 检查存储的配置
   import boto3
   ssm = boto3.client('ssm')
   response = ssm.get_parameter(Name='/mcp_server_sample/runtime/agent_arn')
   print(response['Parameter']['Value'])
   ```

3. **测试认证**：
   ```python
   # 验证 Cognito 令牌
   import boto3
   secrets = boto3.client('secretsmanager')
   response = secrets.get_secret_value(SecretId='mcp_server_sample/cognito/credentials')
   print(response['SecretString'])
   ```

4. **测试 ComfyUI 连接**：
   ```bash
   # 测试 ComfyUI 服务器连接
   curl http://your-comfyui-server:8188/queue

   # 查看 ComfyUI 系统信息
   curl http://your-comfyui-server:8188/system_stats
   ```

5. **ComfyUI 调试**：
   ```python
   # 测试 ComfyUI 配置工具
   import requests
   import os

   comfyui_url = os.environ.get('COMFYUI_SERVER_URL', 'http://localhost:8188')
   try:
       response = requests.get(f"{comfyui_url}/queue", timeout=5)
       print(f"ComfyUI 连接状态: {response.status_code}")
   except Exception as e:
       print(f"ComfyUI 连接失败: {e}")
   ```

6. **启用详细错误日志**：
   ```bash
   # 禁用回退模式以查看真实错误
   export COMFYUI_ENABLE_FALLBACK="false"

   # 增加调试输出
   export COMFYUI_POLL_INTERVAL="1"  # 更频繁的状态检查
   ```

## 清理资源

运行 notebook 中的清理单元格或手动删除：

1. AgentCore Runtime
2. ECR 仓库
3. IAM 角色和策略
4. Cognito 用户池
5. Parameter Store 参数
6. Secrets Manager 密钥

## 下一步

1. **扩展工具**: 添加更多 MCP 工具
2. **自定义认证**: 实现自定义 JWT 授权器
3. **监控**: 设置 CloudWatch 监控和告警
4. **集成**: 与其他 AWS 服务集成
5. **优化**: 根据使用情况优化性能

## ComfyUI 部署建议

### 生产环境部署

1. **网络配置**:
   - 将 ComfyUI 部署在与 AgentCore Runtime 相同的 VPC 内
   - 使用私有子网和安全组限制访问
   - 配置 Application Load Balancer 进行负载均衡

2. **性能优化**:
   - 使用 GPU 实例（如 EC2 G4/P3/P4 实例）
   - 配置足够的内存和存储空间
   - 预加载常用模型以减少启动时间

3. **高可用性**:
   - 部署多个 ComfyUI 实例
   - 使用 Auto Scaling Group 自动扩缩容
   - 配置健康检查和故障转移

4. **安全性**:
   - 使用 IAM 角色和安全组控制访问
   - 启用 VPC Flow Logs 监控网络流量
   - 定期更新 ComfyUI 和依赖项

### Docker 部署示例

```dockerfile
# Dockerfile for ComfyUI
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git
RUN git clone https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /ComfyUI
RUN pip3 install -r requirements.txt

EXPOSE 8188
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
```

```bash
# 构建和运行
docker build -t comfyui .
docker run -d --gpus all -p 8188:8188 --name comfyui comfyui
```

## 参考资源

### AWS 和 MCP 相关
- [AWS Bedrock AgentCore 文档](https://docs.aws.amazon.com/bedrock-agentcore/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [AgentCore 示例仓库](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
- [bedrock-agentcore-starter-toolkit](https://pypi.org/project/bedrock-agentcore-starter-toolkit/)

### ComfyUI 相关
- [ComfyUI 官方仓库](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI 文档](https://docs.comfy.org/)
- [ComfyUI API 文档](https://github.com/comfyanonymous/ComfyUI/blob/master/server.py)
- [ComfyUI 模型下载](https://huggingface.co/models?library=diffusers)
- [ComfyUI 工作流示例](https://comfyworkflows.com/)

### Docker 和部署
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [AWS EC2 GPU 实例](https://aws.amazon.com/ec2/instance-types/g4/)
- [AWS ECS 服务](https://aws.amazon.com/ecs/)

## 支持

如有问题，请查看：
1. AWS Bedrock AgentCore 文档
2. MCP 协议文档
3. 项目 GitHub Issues
