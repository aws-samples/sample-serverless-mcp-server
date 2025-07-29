#!/bin/bash

# ComfyUI MCP Server 部署脚本
# 使用方法: ./deploy.sh [ComfyUI服务器URL]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查必要工具
check_prerequisites() {
    print_info "检查部署前置条件..."
    
    if ! command -v sam &> /dev/null; then
        print_error "SAM CLI 未安装。请运行: pip install aws-sam-cli"
        exit 1
    fi
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI 未安装。请运行: pip install awscli"
        exit 1
    fi
    
    # 检查 AWS 凭证
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS 凭证未配置。请运行: aws configure"
        exit 1
    fi
    
    print_success "前置条件检查通过"
}

# 检查配置文件
check_config() {
    print_info "检查配置文件..."
    
    if [ ! -f "samconfig.toml" ]; then
        print_warning "samconfig.toml 不存在，从示例文件创建..."
        if [ -f "samconfig-example.toml" ]; then
            cp samconfig-example.toml samconfig.toml
            print_warning "请编辑 samconfig.toml 文件，更新您的配置参数"
            print_warning "特别是 McpAuthToken 和 ComfyUIServerUrl"
            read -p "按 Enter 键继续，或 Ctrl+C 退出编辑配置..."
        else
            print_error "samconfig-example.toml 文件不存在"
            exit 1
        fi
    fi
    
    print_success "配置文件检查完成"
}

# 更新 ComfyUI 服务器 URL
update_comfyui_url() {
    local comfyui_url="$1"
    
    if [ -n "$comfyui_url" ]; then
        print_info "更新 ComfyUI 服务器 URL 为: $comfyui_url"
        
        # 使用 sed 更新 samconfig.toml 中的 ComfyUIServerUrl
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|ComfyUIServerUrl=[^,]*|ComfyUIServerUrl=$comfyui_url|g" samconfig.toml
        else
            # Linux
            sed -i "s|ComfyUIServerUrl=[^,]*|ComfyUIServerUrl=$comfyui_url|g" samconfig.toml
        fi
        
        print_success "ComfyUI 服务器 URL 已更新"
    fi
}

# 验证工作流文件
validate_workflows() {
    print_info "验证工作流文件..."
    
    if [ ! -f "workflows/flux_t2i.json" ]; then
        print_error "workflows/flux_t2i.json 文件不存在"
        exit 1
    fi
    
    if [ ! -f "workflows/flux_kontext.json" ]; then
        print_error "workflows/flux_kontext.json 文件不存在"
        exit 1
    fi
    
    # 验证 JSON 格式
    if ! python3 -m json.tool workflows/flux_t2i.json > /dev/null 2>&1; then
        print_error "workflows/flux_t2i.json 不是有效的 JSON 文件"
        exit 1
    fi
    
    if ! python3 -m json.tool workflows/flux_kontext.json > /dev/null 2>&1; then
        print_error "workflows/flux_kontext.json 不是有效的 JSON 文件"
        exit 1
    fi
    
    print_success "工作流文件验证通过"
}

# 构建应用
build_app() {
    print_info "构建 SAM 应用..."
    
    if sam build; then
        print_success "应用构建成功"
    else
        print_error "应用构建失败"
        exit 1
    fi
}

# 部署应用
deploy_app() {
    print_info "部署 SAM 应用..."
    
    if sam deploy; then
        print_success "应用部署成功"
    else
        print_error "应用部署失败"
        exit 1
    fi
}

# 显示部署信息
show_deployment_info() {
    print_info "获取部署信息..."
    
    # 获取 stack 名称
    local stack_name=$(grep "stack_name" samconfig.toml | cut -d'"' -f2)
    
    if [ -n "$stack_name" ]; then
        print_info "Stack 名称: $stack_name"
        
        # 获取 API Gateway URL
        local api_url=$(aws cloudformation describe-stacks \
            --stack-name "$stack_name" \
            --query 'Stacks[0].Outputs[?OutputKey==`MCPServerApi`].OutputValue' \
            --output text 2>/dev/null)
        
        if [ -n "$api_url" ] && [ "$api_url" != "None" ]; then
            print_success "MCP Server API URL: $api_url"
        fi
        
        # 获取 DynamoDB 表名
        local table_name=$(aws cloudformation describe-stacks \
            --stack-name "$stack_name" \
            --query 'Stacks[0].Outputs[?OutputKey==`McpSessionsTableName`].OutputValue' \
            --output text 2>/dev/null)
        
        if [ -n "$table_name" ] && [ "$table_name" != "None" ]; then
            print_success "DynamoDB 表名: $table_name"
        fi
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "ComfyUI MCP Server 部署脚本"
    echo "========================================"
    
    # 检查参数
    local comfyui_url="$1"
    
    # 执行部署步骤
    check_prerequisites
    check_config
    update_comfyui_url "$comfyui_url"
    validate_workflows
    build_app
    deploy_app
    show_deployment_info
    
    echo "========================================"
    print_success "部署完成！"
    echo "========================================"
    
    print_info "下一步："
    echo "1. 测试 MCP Server API 连接"
    echo "2. 验证 ComfyUI 集成功能"
    echo "3. 配置监控和告警"
}

# 显示使用帮助
show_help() {
    echo "ComfyUI MCP Server 部署脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [ComfyUI服务器URL]"
    echo ""
    echo "示例:"
    echo "  $0                                          # 使用配置文件中的默认 URL"
    echo "  $0 http://localhost:8188                    # 本地 ComfyUI 服务器"
    echo "  $0 http://ec2-xx-xx-xx-xx.compute-1.amazonaws.com:8188  # EC2 上的 ComfyUI"
    echo ""
    echo "选项:"
    echo "  -h, --help    显示此帮助信息"
    echo ""
}

# 处理命令行参数
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac
