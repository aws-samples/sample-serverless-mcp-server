from lambda_mcp.lambda_mcp import LambdaMCPServer
import random
import boto3
import os
import requests
from typing import Dict, Any, Optional, List

SERPAPI_API_KEY = "*******"
# Get session table name from environment variable
session_table = os.environ.get('MCP_SESSION_TABLE', 'mcp_sessions')

# Create the MCP server instance
mcp_server = LambdaMCPServer(name="mcp-lambda-server", version="1.0.0", session_table=session_table)

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

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    return mcp_server.handle_request(event, context) 

