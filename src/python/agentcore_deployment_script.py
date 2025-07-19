#!/usr/bin/env python3
"""
AWS Bedrock AgentCore MCP Server 部署脚本

这个脚本展示了如何使用 AWS Bedrock AgentCore SDK 部署 MCP server。
它包含了从设置认证到部署的完整流程。

使用方法:
    python agentcore_deployment_script.py

前提条件:
    - AWS 凭证已配置
    - 安装了必要的依赖: pip install mcp bedrock-agentcore-starter-toolkit boto3
    - Docker 正在运行
    - agentcore_mcp_server.py 和 requirements.txt 文件存在
"""

import boto3
import json
import time
import os
import sys
from boto3.session import Session
from bedrock_agentcore_starter_toolkit import Runtime

def setup_cognito_user_pool():
    """设置 Cognito 用户池和客户端"""
    cognito_client = boto3.client('cognito-idp')
    
    try:
        print("创建 Cognito 用户池...")
        user_pool_response = cognito_client.create_user_pool(
            PoolName='MCPServerUserPool',
            Policies={
                'PasswordPolicy': {
                    'MinimumLength': 8,
                    'RequireUppercase': False,
                    'RequireLowercase': False,
                    'RequireNumbers': False,
                    'RequireSymbols': False
                }
            }
        )
        
        user_pool_id = user_pool_response['UserPool']['Id']
        print(f"✓ 用户池创建成功: {user_pool_id}")
        
        # 创建应用客户端
        print("创建应用客户端...")
        client_response = cognito_client.create_user_pool_client(
            UserPoolId=user_pool_id,
            ClientName='MCPServerClient',
            GenerateSecret=False,
            ExplicitAuthFlows=[
                'ALLOW_USER_PASSWORD_AUTH',
                'ALLOW_REFRESH_TOKEN_AUTH'
            ]
        )
        
        client_id = client_response['UserPoolClient']['ClientId']
        print(f"✓ 应用客户端创建成功: {client_id}")
        
        # 创建测试用户
        print("创建测试用户...")
        cognito_client.admin_create_user(
            UserPoolId=user_pool_id,
            Username='testuser',
            TemporaryPassword='TempPass123!',
            MessageAction='SUPPRESS'
        )
        
        # 设置永久密码
        cognito_client.admin_set_user_password(
            UserPoolId=user_pool_id,
            Username='testuser',
            Password='MyPassword123!',
            Permanent=True
        )
        print("✓ 测试用户创建成功")
        
        # 获取访问令牌
        print("获取访问令牌...")
        auth_response = cognito_client.initiate_auth(
            ClientId=client_id,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': 'testuser',
                'PASSWORD': 'MyPassword123!'
            }
        )
        
        access_token = auth_response['AuthenticationResult']['AccessToken']
        print("✓ 访问令牌获取成功")
        
        # 获取区域
        session = Session()
        region = session.region_name or 'us-east-1'
        
        discovery_url = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/openid-configuration"
        
        return {
            'user_pool_id': user_pool_id,
            'client_id': client_id,
            'discovery_url': discovery_url,
            'bearer_token': access_token,
            'region': region
        }
        
    except Exception as e:
        print(f"❌ Cognito 设置失败: {e}")
        raise

def create_agentcore_role(agent_name: str):
    """创建 AgentCore IAM 执行角色"""
    iam_client = boto3.client('iam')
    
    role_name = f"AgentCore-{agent_name}-ExecutionRole"
    
    # 信任策略
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock-agentcore.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # 创建角色
        role_response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"Execution role for AgentCore {agent_name}"
        )
        
        # 附加基本执行策略
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "s3:ListAllMyBuckets",
                        "s3:GetBucketLocation"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{agent_name}-ExecutionPolicy",
            PolicyDocument=json.dumps(policy_document)
        )
        
        print(f"✓ IAM 角色创建成功: {role_name}")
        return role_response
        
    except Exception as e:
        print(f"❌ IAM 角色创建失败: {e}")
        raise

def deploy_mcp_server():
    """部署 MCP server 到 AgentCore Runtime"""
    
    print("🚀 开始部署 MCP Server 到 AWS Bedrock AgentCore Runtime")
    print("=" * 60)
    
    # 检查必需文件
    required_files = ['agentcore_mcp_server.py', 'requirements.txt']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 必需文件 {file} 未找到")
            print("请确保您在正确的目录中运行此脚本")
            sys.exit(1)
    print("✓ 所有必需文件已找到")
    
    # 设置 Cognito
    print("\n1. 设置 Amazon Cognito 认证...")
    cognito_config = setup_cognito_user_pool()
    
    # 创建 IAM 角色
    print("\n2. 创建 IAM 执行角色...")
    tool_name = "mcp_server_sample"
    agentcore_iam_role = create_agentcore_role(agent_name=tool_name)
    
    # 配置 AgentCore Runtime
    print("\n3. 配置 AgentCore Runtime...")
    region = cognito_config['region']
    agentcore_runtime = Runtime()
    
    auth_config = {
        "customJWTAuthorizer": {
            "allowedClients": [
                cognito_config['client_id']
            ],
            "discoveryUrl": cognito_config['discovery_url'],
        }
    }
    
    response = agentcore_runtime.configure(
        entrypoint="agentcore_mcp_server.py",
        execution_role=agentcore_iam_role['Role']['Arn'],
        auto_create_ecr=True,
        requirements_file="requirements.txt",
        region=region,
        authorizer_configuration=auth_config,
        protocol="MCP",
        agent_name=tool_name
    )
    print("✓ 配置完成")
    
    # 启动部署
    print("\n4. 启动部署...")
    print("这可能需要几分钟时间...")
    launch_result = agentcore_runtime.launch()
    print("✓ 启动完成")
    print(f"Agent ARN: {launch_result.agent_arn}")
    print(f"Agent ID: {launch_result.agent_id}")
    
    # 等待就绪
    print("\n5. 等待 AgentCore Runtime 就绪...")
    status_response = agentcore_runtime.status()
    status = status_response.endpoint['status']
    print(f"初始状态: {status}")
    
    end_status = ['READY', 'CREATE_FAILED', 'DELETE_FAILED', 'UPDATE_FAILED']
    while status not in end_status:
        print(f"状态: {status} - 等待中...")
        time.sleep(10)
        status_response = agentcore_runtime.status()
        status = status_response.endpoint['status']
    
    if status == 'READY':
        print("✅ AgentCore Runtime 已就绪！")
    else:
        print(f"⚠ AgentCore Runtime 状态: {status}")
        return None
    
    # 存储配置
    print("\n6. 存储配置信息...")
    ssm_client = boto3.client('ssm', region_name=region)
    secrets_client = boto3.client('secretsmanager', region_name=region)
    
    try:
        # 存储 Cognito 凭证
        try:
            secrets_client.create_secret(
                Name='mcp_server_sample/cognito/credentials',
                Description='MCP server 的 Cognito 凭证',
                SecretString=json.dumps(cognito_config)
            )
        except secrets_client.exceptions.ResourceExistsException:
            secrets_client.update_secret(
                SecretId='mcp_server_sample/cognito/credentials',
                SecretString=json.dumps(cognito_config)
            )
        
        # 存储 Agent ARN
        ssm_client.put_parameter(
            Name='/mcp_server_sample/runtime/agent_arn',
            Value=launch_result.agent_arn,
            Type='String',
            Description='MCP server 的 Agent ARN',
            Overwrite=True
        )
        print("✓ 配置信息存储成功")
        
    except Exception as e:
        print(f"❌ 存储配置时出错: {e}")
    
    print("\n🎉 部署完成！")
    print("=" * 60)
    print(f"Agent ARN: {launch_result.agent_arn}")
    print(f"Bearer Token: {cognito_config['bearer_token']}")
    print(f"区域: {region}")
    print("\n您现在可以使用 notebook 中的测试代码测试已部署的 MCP server")
    
    return {
        'agent_arn': launch_result.agent_arn,
        'cognito_config': cognito_config,
        'region': region
    }

if __name__ == "__main__":
    try:
        result = deploy_mcp_server()
        if result:
            print("\n✅ 部署成功完成！")
        else:
            print("\n❌ 部署失败")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ 部署被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 部署过程中出现错误: {e}")
        sys.exit(1)
