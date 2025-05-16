# Lambda MCP Server development & deploy demo (Streamable HTTP) 


Based on [Lambda MCP demo](https://github.com/mikegc-aws/Lambda-MCP-Server), it demonstrates how to build a stateless, serverless MCP server with minimal boilerplate and an excellent developer experience according to [MCP (Model Context Protocol)](https://github.com/modelcontextprotocol) tools using [AWS Lambda and AWS API gateway](https://aws.amazon.com/lambda/?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code). 

It also included client demonstrates integration with [Amazon Bedrock](https://aws.amazon.com/bedrock/?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code), to build further application such as LLM function calling or intelligent agent.


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

The server comes with three example tools that demonstrate different use cases:

1. `search_website()`: using serper API to do internet search
2. `count_s3_buckets()`: Counts [AWS S3](https://aws.amazon.com/s3/?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) buckets in your account

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

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```

1. Navigate to the server directory:
   ```bash
   cd server-http-python-lambda
   ```

1. Deploy using SAM:
   ```bash
   sam build
   sam deploy --guided
   ```

   Note: You will be prompted for an `McpAuthToken`.  This is the Authorization Bearer token that will be requitred to call the endpoint. This simple implimentation uses an [AWS API Gateway authorizers](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-use-lambda-authorizer.html?trk=64e03f01-b931-4384-846e-db0ba9fa89f5&sc_channel=code) with the `McpAuthToken` passed in as an env var.  This can be swapped out for a production implimentation as required. 


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


