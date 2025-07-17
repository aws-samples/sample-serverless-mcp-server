# Lambda MCP Server with ComfyUI Integration (Streamable HTTP)

Based on [Lambda MCP demo](https://github.com/mikegc-aws/Lambda-MCP-Server), this project demonstrates how to build a stateless, serverless MCP server with minimal boilerplate and an excellent developer experience according to [MCP (Model Context Protocol)](https://github.com/modelcontextprotocol) tools using [AWS Lambda and AWS API gateway](https://aws.amazon.com/lambda/?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code).

**Key Features:**
- 🚀 **Serverless MCP Server**: Deploy to AWS Lambda with one command
- 🎨 **ComfyUI Integration**: Generate images using Flux models (text-to-image & image-to-image)
- 🔄 **Session Management**: Built-in state persistence across tool invocations
- 🛡️ **Authentication**: Bearer token authentication with API Gateway
- 📊 **Monitoring**: CloudWatch logs and metrics integration
- 🔧 **Easy Development**: Simple decorator-based tool creation

It also includes client demonstrations for integration with [Amazon Bedrock](https://aws.amazon.com/bedrock/?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code), enabling LLM function calling and intelligent agent applications.

## Project Structure

```
src/python/
├── workflows/                    # ComfyUI workflow definitions
│   ├── flux_t2i.json            # Text-to-image workflow
│   └── flux_kontext.json        # Image-to-image workflow
├── server/                       # MCP server implementation
│   ├── app.py                   # Main MCP server with tools
│   └── comfyui_generator.py     # ComfyUI integration logic
├── template.yaml                # AWS SAM template
├── samconfig-example.toml       # Deployment configuration template
├── deploy.sh                    # One-click deployment script
└── test_workflow_integration.py # Integration tests
```

## Example

After you git clone this repo , this is all the code you need for development & deployment of python Streamable http MCP server: 

```Python
from lambda_mcp.lambda_mcp import LambdaMCPServer

# Create the MCP server instance
mcp_server = LambdaMCPServer(name="mcp-lambda-server", version="1.0.0")

@mcp_server.tool()
def say_hello_world() -> int:
    """Say hello world!"""
    return "Hello MCP World!"

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    return mcp_server.handle_request(event, context) 
```

That's it! :)

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd src/python
```

### 2. Deploy with ComfyUI
```bash
# Deploy with your ComfyUI server URL
./deploy.sh http://your-comfyui-server:8188
```

### 3. Test the Deployment
```bash
# Run integration tests
python3 test_workflow_integration.py

# The deployment will output your MCP server URL:
# MCPServerApi: https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/Prod/mcp
```

### 4. Use with MCP Clients
Configure your MCP client with:
- **Server URL**: The API Gateway URL from deployment output
- **Authentication**: Bearer token (set in deployment configuration)

## Session State Management

The Lambda MCP Server includes built-in session state management that persists across tool invocations within the same conversation. This is particularly useful for tools that need to maintain context or share data between calls.

Session data is stored in a DynamoDB table against a sessionId key. This is all managed for you.

Here's an example of how to use session state:

```Python
from lambda_mcp.lambda_mcp import LambdaMCPServer

session_table = os.environ.get('MCP_SESSION_TABLE', 'mcp_sessions')

mcp_server = LambdaMCPServer(name="mcp-lambda-server", version="1.0.0", session_table=session_table)

@mcp_server.tool()
def increment_counter() -> int:
    """Increment a session-based counter."""
    # Get the current counter value from session state, default to 0 if not set
    counter = mcp_server.session.get('counter', 0)
    
    # Increment the counter
    counter += 1
    
    # Store the new value in session state
    mcp_server.session['counter'] = counter
    
    return counter

@mcp_server.tool()
def get_counter() -> int:
    """Get the current counter value."""
    return mcp_server.session.get('counter', 0)
```

The session state is automatically managed per conversation and persists across multiple tool invocations. This allows you to maintain stateful information without needing additional external storage, while still keeping your Lambda function stateless.

## Authentication

The sample server stack uses Bearer token authentication via an Authorization header, which is compliant with the MCP standard. This provides a basic level of security for your MCP server endpoints. Here's what you need to know:

1. **Bearer Token**: When you deploy the stack, a bearer token is configured through a custom authorizer in API Gateway
2. **Using the Bearer Token**: 
   - The client must include the bearer token in requests using the `Authorization` header with the format: `Bearer <your-token>`
   - The token value is provided in the stack outputs after deployment
   - The sample client is configured to automatically include this header when provided with the token

3. **Custom Authorizer**: The implementation uses a simple custom authorizer that validates a single bearer token. This can be easily extended or replaced with more sophisticated authentication systems like Amazon Cognito for production use.

The current bearer token implementation is primarily intended for demonstration and development purposes. For production systems handling sensitive data, implement appropriate additional security measures based on your specific requirements.

## What is this all about?

This is a proof-of-concept implementation of an python Streamable MCP server running on AWS Lambda or EC2, also provide a python client that demonstrates its function calling with AWS Bedrock api with claude models. The project consists of two main components:

1. **Lambda python MCP Server**: (./server/app.py) A Python-based serverless implementation that makes it incredibly simple to deploy cloud hosted MCP tools.
2. **Python MCP Server on EC2**: (./mcp_server_sample.py) A Pure Python-based serverless implementation which use fastapi/uvicorn web framwork to fully implements Streamable MCP tools&server.
3. **Sample python HTTP Client**: (./mcp_client_sample.py) A demonstration client that shows how to interact with the Lambda MCP server using Amazon Bedrock's Converse API (using pure http request based on MCP streamable transparent protocal)

## Example Tools

The server comes with several example tools that demonstrate different use cases:

1. `search_website()`: using serper API to do internet search
2. `count_s3_buckets()`: Counts [AWS S3](https://aws.amazon.com/s3/?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) buckets in your account
3. `generate_image_with_context()`: Generate images using ComfyUI with Flux models (text-to-image and image-to-image)
4. `get_comfyui_config()`: Get ComfyUI configuration and available workflows

## Getting Started

### Prerequisites

- [AWS Account](https://aws.amazon.com/free/?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) with appropriate permissions
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) installed
- [Python 3.9+](https://www.python.org/downloads/)
- Access to [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) in your AWS account
- [Bedrock Claude Model access](https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) enabled in your Amazon Bedrock model access settings

### Amazon Bedrock Setup

Before running the client, ensure you have:

1. [Enabled Amazon Bedrock access](https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) in your AWS account
2. [Enabled the Amazon Nova Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) in your Bedrock model access settings
3. Appropriate [IAM permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) to invoke Bedrock APIs

### Server Deployment

#### Quick Deploy (Recommended)

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd src/python
   ```

2. Deploy with ComfyUI integration:
   ```bash
   # One-step deployment with ComfyUI server URL
   ./deploy.sh http://your-comfyui-server:8188
   ```

#### Manual Deployment

1. Navigate to the server directory:
   ```bash
   cd src/python
   ```

2. Configure deployment parameters:
   ```bash
   # Copy configuration template
   cp samconfig-example.toml samconfig.toml

   # Edit samconfig.toml and update:
   # - McpAuthToken: Your secure authentication token
   # - ComfyUIServerUrl: Your ComfyUI server address
   # - Other ComfyUI parameters as needed
   ```

3. Deploy using SAM:
   ```bash
   sam build
   sam deploy
   ```

#### ComfyUI Configuration Parameters

The following parameters can be configured in `samconfig.toml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `McpAuthToken` | - | **Required**: Authentication token for MCP server |
| `ComfyUIServerUrl` | `http://localhost:8188` | **Required**: ComfyUI server address |
| `ComfyUITimeout` | `300` | Generation timeout in seconds |
| `ComfyUIPollInterval` | `2` | Polling interval in seconds |
| `ComfyUIMaxRetries` | `3` | Maximum retry attempts |
| `ComfyUIRequestTimeout` | `30` | Single request timeout in seconds |
| `ComfyUIEnableFallback` | `true` | Enable fallback to mock images |

#### Common Deployment Scenarios

**Local ComfyUI Server:**
```bash
./deploy.sh http://localhost:8188
```

**ComfyUI on EC2:**
```bash
./deploy.sh http://ec2-xx-xx-xx-xx.compute-1.amazonaws.com:8188
```

**ComfyUI behind Load Balancer:**
```bash
./deploy.sh http://your-alb-url.us-east-1.elb.amazonaws.com:8188
```

**External ComfyUI Server:**
```bash
./deploy.sh http://your-external-server.com:8188
```


## Adding New Tools

The Lambda MCP Server is designed to make tool creation as simple as possible. Here's how to add a new tool:

1. Open `server/app.py`
2. Add your new tool using the decorator pattern:

```python
@mcp_server.tool()
def my_new_tool(param1: str, param2: int) -> str:
    """Your tool description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    # Your tool implementation
    return f"Processed {param1} with value {param2}"
```

That's it! The decorator handles:
- Type validation
- Request parsing
- Response formatting
- Error handling
- MCP Documentation generation

## ComfyUI Integration

This MCP server includes integrated support for ComfyUI image generation using Flux models. The integration supports two main workflows:

### Supported Workflows

1. **Text-to-Image**: Generate images from text prompts using `flux_t2i.json`
2. **Image-to-Image**: Transform existing images using `flux_kontext.json`

### Workflow Files

The ComfyUI workflows are stored as JSON files in the `workflows/` directory:

```
src/python/
├── workflows/
│   ├── flux_t2i.json      # Text-to-image workflow
│   └── flux_kontext.json  # Image-to-image workflow
├── server/
│   ├── comfyui_generator.py  # ComfyUI integration logic
│   └── app.py               # MCP server with ComfyUI tools
```

### ComfyUI Requirements

To use the ComfyUI integration, ensure your ComfyUI server has:

**Models:**
- **UNet**: `flux1-dev-fp8.safetensors` or `flux1-dev-kontext_fp8_scaled.safetensors`
- **VAE**: `ae.safetensors`
- **CLIP**: `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors`

**Custom Nodes** (for flux_kontext workflow):
- `FluxKontextImageScale`
- `ETN_LoadImageBase64`
- `Text Multiline`
- `ImageStitch`
- `ReferenceLatent`

### Using ComfyUI Tools

#### Text-to-Image Generation
```python
# Example MCP tool call
result = generate_image_with_context(
    prompt="A beautiful mountain landscape with snow-capped peaks",
    workflow_type="text_to_image",
    width=1024,
    height=768,
    steps=25,
    cfg_scale=7.0
)
```

#### Image-to-Image Generation
```python
# Example MCP tool call
result = generate_image_with_context(
    prompt="Transform into an oil painting style",
    context_image_base64="data:image/jpeg;base64,/9j/4AAQ...",
    workflow_type="image_to_image",
    steps=20,
    cfg_scale=1.0
)
```

### Testing ComfyUI Integration

Run the integration test to verify everything is working:

```bash
cd src/python
python3 test_workflow_integration.py
```

This will test:
- Workflow file loading
- Parameter replacement
- Node ID validation
- Generation methods (with fallback if ComfyUI unavailable)

### Troubleshooting

**Common Issues:**

1. **ComfyUI Connection Failed**
   - Verify ComfyUI server is running on the specified URL
   - Check network connectivity and firewall settings
   - Ensure port 8188 is accessible

2. **Workflow Loading Errors**
   - Verify workflow JSON files are valid
   - Check that all required custom nodes are installed
   - Ensure model files are in the correct ComfyUI directories

3. **Generation Timeouts**
   - Increase `ComfyUITimeout` parameter
   - Optimize generation parameters (reduce steps, image size)
   - Check ComfyUI server performance

**Debug Commands:**
```bash
# View Lambda logs
sam logs -n McpServerFunction --stack-name mcp-comfyui-server

# Test workflow loading locally
python3 test_workflow_integration.py

# Check ComfyUI server status
curl http://your-comfyui-server:8188/queue
```

### Fallback Mechanism

When ComfyUI is unavailable, the server automatically falls back to returning mock images (1x1 pixel) with detailed error information in the metadata. This ensures the MCP server remains functional even when ComfyUI is down.

To disable fallback mode, set `ComfyUIEnableFallback=false` in your deployment configuration.


