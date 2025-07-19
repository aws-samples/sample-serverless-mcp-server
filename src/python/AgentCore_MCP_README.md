# AWS Bedrock AgentCore MCP Server 部署示例

本示例展示了如何使用 AWS Bedrock AgentCore SDK 将现有的 MCP (Model Context Protocol) server 工具部署到 AgentCore Runtime。

## 概述

这个示例包含了将 `server/app.py` 中的 MCP server 工具适配并部署到 AWS Bedrock AgentCore Runtime 的完整流程，包括：

- 网站搜索工具 (`search_website`)
- S3 存储桶计数工具 (`count_s3_buckets`) 
- 图像生成工具 (`generate_image_with_context`)
- ComfyUI 配置获取工具 (`get_comfyui_config`)
- 视频生成工具 (`generate_video_with_context`)

## 文件说明

### 核心文件

1. **`agentcore_hosting_mcp_sample.ipynb`** - 完整的 Jupyter notebook 教程
2. **`agentcore_mcp_server.py`** - 适配 AgentCore Runtime 的 MCP server
3. **`agentcore_deployment_script.py`** - 自动化部署脚本
4. **`requirements.txt`** - Python 依赖

### 支持文件

- **`AgentCore_MCP_README.md`** - 本文档

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

### 方法 1: 使用 Jupyter Notebook（推荐）

1. 打开 `agentcore_hosting_mcp_sample.ipynb`
2. 按顺序执行所有单元格
3. 按照提示配置认证和部署

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
- **功能**: 图像生成（模拟 ComfyUI）
- **参数**: 
  - `prompt`: 文本描述
  - `context_image_base64`: 上下文图像（可选）
  - `workflow_type`: 工作流类型
  - `width`, `height`: 图像尺寸
  - `steps`, `cfg_scale`, `seed`: 生成参数
- **返回**: 生成结果和元数据

### 4. get_comfyui_config
- **功能**: 获取 ComfyUI 配置信息
- **参数**: 无
- **返回**: 配置字典

### 5. generate_video_with_context
- **功能**: 视频生成（模拟 ComfyUI）
- **参数**: 
  - `prompt`: 文本描述
  - `context_image_base64`: 上下文图像（可选）
  - `workflow_type`: 工作流类型
  - `steps`, `cfg_scale`, `seed`, `frame_rate`: 生成参数
- **返回**: 生成结果和元数据

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

## 参考资源

- [AWS Bedrock AgentCore 文档](https://docs.aws.amazon.com/bedrock-agentcore/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [AgentCore 示例仓库](https://github.com/awslabs/amazon-bedrock-agentcore-samples)

## 支持

如有问题，请查看：
1. AWS Bedrock AgentCore 文档
2. MCP 协议文档
3. 项目 GitHub Issues
