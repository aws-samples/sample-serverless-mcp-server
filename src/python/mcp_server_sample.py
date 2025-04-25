import sys, os
from datetime import datetime
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response,PlainTextResponse,JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route, Mount
import json
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from rag.retrieve.search import search_git_doc_lance
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# 存储会话ID和对应的任务队列，仅用于需要session的功能
sessions: Dict[str, Dict[str, Any]] = {}
SERPAPI_API_KEY = "*******"


SEARCH_WEBSITE_TOOL_SCHEMA ={
    "name": "search_website",
    "description": "Use this tools WHENEVER When querying something which we don't know and need to search the website to get the exactly and latest information",
    "inputSchema": {
        "type": "object",
        "properties": {
            "search_term": {
                "type": "string",
                "description": "The term to search on website"
            }
        },
        "required": [
            "search_term"
        ]
    }
}

CODE_SEARCH_TOOL_SCHEMA = {
        "name": "search_codes",
        "description": "Search for code snippets based on a given search term within a specified repository.",
        "inputSchema": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Search term to look for in code"
                    },
                    "repo_url": {
                        "type": "string",
                        "description": "Repository URL"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number",
                        "default": 1
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Items per page",
                        "default": 10
                    }
                },
                "required": [
                    "search_term",
                    "repo_url"
                ],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
    }



def search_website(search_term):
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

## 模拟SSE消息推送
async def event_generator(session_id):
    loop_count = 0
    max_loops = 10

    while loop_count < max_loops:
        try:
            # Try to get a message from the queue with a timeout
            message = await asyncio.wait_for(sessions[session_id]["task_queue"].get(), timeout=3)
            
        except asyncio.TimeoutError:
            # If timeout occurs, send a heartbeat message
            heartbeat_message = {
                "type": "heartbeat", 
                "server_timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(heartbeat_message)}\n\n"
        loop_count += 1
        yield json.dumps(content)

async def event_generator_simple(session_id:str=""):
            data={
                    "jsonrpc": "2.0",
                    "method": "sse/connection",
                    "params": { "message": "Stream started" }

                }
            yield json.dumps(data)


            for i in range(2):
                await asyncio.sleep(1)  # 模拟延迟
                data={
                    "jsonrpc": "2.0",
                    "method": "sse/message",
                    "params": { "data": "server message "+str(i) }
                }
                yield json.dumps(data)
            
            data={
                    "jsonrpc": "2.0",
                    "method": "sse/complete",
                    "params": { "message": "Stream completed" }
                }
            yield json.dumps(data)


async def log_request(request: Request):
    """记录请求详情的辅助函数"""
    print(f"Request URL: {request.url}")
    print(f"Request Method: {request.method}")
    print(f"Request Headers: {dict(request.headers)}")

    if request.method == "POST":
        try:
            body = await request.body()
            body_text = body.decode()
            print(f"Request Body: {body_text}")
            # 重新设置request.body，因为body只能读取一次
            setattr(request, '_body', body)
        except Exception as e:
            print(f"Error logging request body: {str(e)}")



def handle_options(request: Request):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "86400",
        "Mcp-Session-Id": "preflight-session-id"
    }
    return Response(status_code=204, headers=headers)


async def handle_message_v2(request: Request):
    #await log_request(request)
    ##"CORS"###
    if request.method == "OPTIONS":
        return handle_options(request)
    ##MCP Client initial###
    elif request.method == "POST":
        try:
            data = await request.json()
            # 处理 tools/list 请求
            if data.get("method") == "tools/list":
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": {
                            "tools": [CODE_SEARCH_TOOL_SCHEMA,SEARCH_WEBSITE_TOOL_SCHEMA]
                        }
                    }
                )

            # 处理 tools/call 请求
            elif data.get("method") == "tools/call":
                try:
                    params = data.get("params", {})
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    meta = params.get("_meta", {})

                    # e.g:搜索工具调用
                    if tool_name == "search_website":
                        search_term = arguments.get("search_term")
                        result =  search_website(search_term)
                        return JSONResponse(
                            content={
                                "jsonrpc": "2.0",
                                "id": data.get("id"),
                                "result": result
                            })
                    else:
                        return JSONResponse(
                            content={
                                "jsonrpc": "2.0",
                                "id": data.get("id"),
                                "error": {
                                    "code": -32601,
                                    "message": f"Tool '{tool_name}' not found"
                                }
                            },
                            status_code=404
                        )

                except Exception as e:
                    logger.info("call method error:" + str(e))
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "id": data.get("id"),
                            "error": {
                                "code": -32000,
                                "message": f"Internal error: {str(e)}"
                            }
                        },
                        status_code=500
                    )

            elif data.get("method") == "initialize":
                # 初始化会话
                session_id = str(uuid.uuid4())
                sessions[session_id] = {
                    "initialized": True,
                    "task_queue": asyncio.Queue()
                }
                response = JSONResponse(status_code=200,
                    content={
                      "jsonrpc": "2.0",
                      "result": {
                          "protocolVersion":"2024-11-05",
                          "serverInfo": {"name": "Code Search MCP Server", "version": "1.0"},
                          "capabilities": {
                                "tools": {"listTools": True},
                                "resources": {}
                            }
                      },
                      "id": data.get("id")
                    }
                )
                response.headers["Mcp-Session-Id"] = session_id
                #response.headers["Content-Type"]="text/event-stream"
                #response.headers["Cache-Control"] = "no-cache"
                #response.headers["Connection"]="keep-alive"
                return response
            
            elif data.get("method")=="notifications/initialized" or data.get("method")=="ping":
                session_id = request.headers.get("Mcp-Session-Id") or request.query_params.get("Mcp-Session-Id")
                content={
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": {
                            "serverInfo": {"name": "MCP Server ready", "version": "1.0"},
                            "capabilities": {
                                "tools": {"listTools": True},
                                "resources": {}
                            }
                        },
                    }
                
                response = JSONResponse(status_code=202,content=content) 
                response.headers["Mcp-Session-Id"] = session_id
                return response 

            else:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "error": {
                            "code": -32601,
                            "message": "Method not found"
                        }
                    },
                    status_code=404
                )

        except Exception as e:
            logger.info("mcp server process error:" + str(e))
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32000,
                        "message": f"Internal server error: {str(e)}"
                    }
                },
                status_code=500
            )
    elif request.method == "DELETE":
        session_id = request.headers.get("Mcp-Session-Id")
        if session_id in sessions:
            sessions.pop(session_id)  # 移除会话
            return JSONResponse(content={"info": "session removed!"}, status_code=200)
        else:
            return JSONResponse(content={"error": "session not found"}, status_code=404)
    elif request.method == "GET":   
        #Opt1: 暂时不提供SSE
        return JSONResponse(content={"error": "Method Not Allowed"}, status_code=405) 
        
        #Opt2: 处理SSE流请求
        #session_id = request.headers.get("Mcp-Session-Id")
        #if not session_id or session_id not in sessions:
        #    return JSONResponse(content={"error": "Session not found"}, status_code=404)
        #response = StreamingResponse(event_generator_simple(session_id), media_type="text/event-stream",status_code=200)
        #response.headers["Mcp-Session-Id"] = session_id
        #return response
    else:
        return JSONResponse(content={"error": f"{request.method} Method Not Allowed"}, status_code=405) 


## sample Streamable Http MCP server 初始化
app = Starlette(
    routes=[
        Route("/message", handle_message_v2, methods=["GET", "POST","OPTIONS","DELETE"])  # initiall/list_tools/tools_call
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)