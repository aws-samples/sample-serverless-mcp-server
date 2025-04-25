import httpx
import json
import asyncio
import aiofiles
import nest_asyncio
import os
from typing import Optional, Dict

# 应用 nest_asyncio 以允许嵌套事件循环
nest_asyncio.apply()

class HttpMCPClient:
    def __init__(self, server_url: str, access_key_id='', secret_access_key='', region='us-east-1'):
        self.env = {
            'AWS_ACCESS_KEY_ID': access_key_id or os.environ.get('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY'),
            'AWS_REGION': region or os.environ.get('AWS_REGION'),
        }
        self.server_url = server_url
        self.session_id = "default"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "jsonrpc":"2.0"
        }

    async def initialize(self):
        """初始化会话。"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.server_url}",
                    headers=self.headers,
                    json={
                        "jsonrpc": "2.0",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "clientInfo": {"name": "MCP Client", "version": "1.0"},
                            "capabilities": {},
                        },
                    },
                )
                response.raise_for_status()
                self.session_id = response.headers.get("Mcp-Session-Id")
                print(f"Session ID: {self.session_id}")
                return self.session_id
            except Exception as e:
                print(f"Failed to initialize session: {e}")
                return None

    async def list_tools(self):
        """发送请求"""
        async with httpx.AsyncClient() as client:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1
                }

                response = await client.post(
                    f"{self.server_url}",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"请求失败: {e}")
                if hasattr(e, 'response'):
                    print(f"响应内容: {e.response.text}")
                return None
            
    async def call_tool(self, method: str, params: dict = None):
        """发送消息。"""
        if not self.session_id:
            await self.initialize()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.server_url}",
                    headers={"Mcp-Session-Id": self.session_id, **self.headers},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params":
                        {
                          "name":method,
                          "arguments": params
                        }
                    },
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Failed to send message: {e}")
                return None

    async def listen_sse(self):
        if not self.session_id:
            await self.initialize()

        async with httpx.AsyncClient(timeout=None) as client:  # 取消超时限制
            try:
                async with client.stream(
                        "GET",
                        f"{self.server_url}",
                        headers={"Mcp-Session-Id": self.session_id, **self.headers},
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():  # 避免空行
                            print(f"SSE Message: {line}")
            except Exception as e:
                print(f"Failed to listen SSE: {e}")
                await self.reconnect()

    async def reconnect(self):
        """断线重连。"""
        print("Attempting to reconnect...")
        await asyncio.sleep(5)  # 等待5秒后重试
        await self.initialize()
        await self.listen_sse()

async def main():
    #client = HttpMCPClient("http://ec2-35-93-77-218.us-west-2.compute.amazonaws.com:8080/message",
    client = HttpMCPClient("https://wtaaklh1ga.execute-api.us-east-1.amazonaws.com/dev/mcp",
                           region="us-east-1")
    await client.initialize()
    response = await client.list_tools()
    print((str(response['result'])))
    #response = await client.call_tool("add", {"a": 5,"b":10000})
    response = await client.call_tool("search_codes", params={
                          "search_term": "Swish",
                          "repo_url":"https://github.com/qingyuan18/ComfyUI-AnyText.git"})
    print(f"Response: {response}")
    #await client.listen_sse()


