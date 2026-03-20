"""
项目配置文件
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置 - 多LLM提供商支持
# 通义千问 (阿里云)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 字节跳动 (头条/豆包)
TOUTIAO_API_KEY = os.getenv("TOUTIAO_API_KEY", "")

# 腾讯混元
HUNYUAN_API_KEY = os.getenv("HUNYUAN_API_KEY", "")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Gemini (Google)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 当前使用的LLM提供商: qwen, deepseek, toutiao, hunyuan, openai, gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")

# 模型配置
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TOUTIAO_MODEL = os.getenv("TOUTIAO_MODEL", "doubao-pro-32k")
HUNYUAN_MODEL = os.getenv("HUNYUAN_MODEL", "hunyuan-lite")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# Embedding配置
# 当前使用的Embedding提供商: qwen, openai, gemini
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "qwen")

# Embedding模型配置
# 通义千问: text-embedding-v2, text-embedding-v3
# OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
# Gemini: models/embedding-001, models/text-embedding-004
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, os.getenv("UPLOAD_DIR", "uploads"))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "knowledge_base"))

# RAG配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 3))

# 学科配置
SUBJECTS = {
    "chinese": {
        "name": "语文",
        "icon": "📖",
        "description": "语文学习辅导，包括阅读理解、作文、古诗词等"
    },
    "math": {
        "name": "数学",
        "icon": "🔢",
        "description": "数学学习辅导，包括算术、几何、应用题等"
    },
    "english": {
        "name": "英语",
        "icon": "🔤",
        "description": "英语学习辅导，包括单词、语法、阅读等"
    }
}

# 确保必要目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# 为每个学科创建上传目录
for subject_key in SUBJECTS.keys():
    subject_upload_dir = os.path.join(UPLOAD_DIR, subject_key)
    os.makedirs(subject_upload_dir, exist_ok=True)
