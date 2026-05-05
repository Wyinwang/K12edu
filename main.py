"""
Python小学智能教育问答系统
主程序入口
"""
# 过滤所有警告信息 - 必须在所有导入之前
import warnings
import logging
import os
import sys
import subprocess
import threading

# 禁用 transformers 库的警告输出 - 必须在导入 transformers 之前设置
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置环境变量，在子进程中也生效
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"

# 设置 logging 级别抑制 transformers 库的信息
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.dynamic_modules").setLevel(logging.ERROR)

# 过滤所有 FutureWarning 和 DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", message="Accessing")

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def filter_stderr_output(process):
    """过滤 stderr 输出，隐藏 transformers 的警告信息"""
    warning_keywords = [
        "Accessing `__path__`",
        "Behavior may be different",
        "will be removed in future versions"
    ]
    for line in process.stderr:
        line = line.decode('utf-8', errors='replace').rstrip()
        # 过滤掉 transformers 的警告行
        if not any(keyword in line for keyword in warning_keywords):
            print(line, file=sys.stderr)


def run_streamlit():
    """运行Streamlit应用"""
    # 设置环境变量以过滤子进程中的警告
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    app_path = os.path.join(os.path.dirname(__file__), "src", "ui", "app.py")

    # 启动 subprocess 并捕获 stderr
    process = subprocess.Popen(
        ["streamlit", "run", app_path],
        env=env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        bufsize=1
    )

    # 启动线程来过滤 stderr 输出
    stderr_thread = threading.Thread(target=filter_stderr_output, args=(process,))
    stderr_thread.daemon = True
    stderr_thread.start()

    # 等待进程结束
    process.wait()


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
