"""
Python小学智能教育问答系统
主程序入口
"""
import os
import sys

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_streamlit():
    """运行Streamlit应用"""
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), "src", "ui", "app.py")
    subprocess.run(["streamlit", "run", app_path])


def main():
    """主函数"""
    print("=" * 50)
    print("Python小学智能教育问答系统")
    print("基于RAG框架的智能问答")
    print("=" * 50)
    print("\n启动Web界面...")
    run_streamlit()


if __name__ == "__main__":
    main()
