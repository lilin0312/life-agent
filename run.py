"""
生活管家 AI-Agent 启动脚本
"""
import os
import sys
import subprocess

def check_api_key():
    """检查 API Key 是否配置"""
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        print("=" * 50)
        print("⚠️  未检测到 DASHSCOPE_API_KEY 环境变量")
        print()
        print("请先设置 API Key（三选一）：")
        print()
        print("  方式1 - 通义千问 (推荐):")
        print("    set DASHSCOPE_API_KEY=sk-xxxxxxxx")
        print()
        print("  方式2 - 智谱 GLM:")
        print("    set ZHIPU_API_KEY=xxxxxxxx")
        print("    set LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4")
        print("    set LLM_MODEL=glm-4-flash")
        print()
        print("  方式3 - OpenAI 兼容接口:")
        print("    set DASHSCOPE_API_KEY=sk-xxxxxxxx")
        print("    set LLM_BASE_URL=https://your-api-endpoint/v1")
        print("    set LLM_MODEL=gpt-4o-mini")
        print()
        print("获取通义千问 Key: https://dashscope.console.aliyun.com/")
        print("=" * 50)
        return False
    return True


def main():
    print("🏠 生活管家 AI-Agent")
    print("=" * 40)

    # 添加项目根目录到 path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    os.chdir(project_dir)

    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)

    # 检查 API Key
    has_key = check_api_key()
    if has_key:
        print("✅ API Key 已配置")
    else:
        print("⚠️  将以降级模式启动（LLM 功能不可用）")

    # 读取配置
    from app.config import SERVER_HOST, SERVER_PORT, SERVER_WORKERS

    print(f"🚀 启动服务: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"   按Ctrl+C停止服务")
    print("=" * 40)

    # 启动 uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--workers", str(SERVER_WORKERS),
        "--log-level", "info",
        "--access-log",
    ]

    try:
        subprocess.run(cmd, cwd=project_dir)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")


if __name__ == "__main__":
    main()
