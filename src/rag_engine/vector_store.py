"""
向量存储模块
使用ChromaDB进行文档向量化存储和检索
支持多学科分离存储
支持多Embedding提供商
"""
import os
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    DASHSCOPE_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
    EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_PROVIDER,
    CHROMA_PERSIST_DIR, SUBJECTS
)
from src.utils.logger import vectorstore_logger


def create_embeddings(provider: Optional[str] = None, model_name: Optional[str] = None) -> Embeddings:
    """
    创建Embedding模型实例

    Args:
        provider: Embedding提供商 (qwen, openai, gemini)
        model_name: 模型名称

    Returns:
        Embeddings实例
    """
    provider = provider or EMBEDDING_PROVIDER
    model_name = model_name or EMBEDDING_MODEL
    vectorstore_logger.info(f"创建Embedding模型 - 提供商: {provider}, 模型: {model_name}")

    if provider == "qwen":
        from langchain_community.embeddings import DashScopeEmbeddings
        vectorstore_logger.info("使用通义千问 Embedding模型")
        # 使用专门的 Embedding API Key 或默认使用 DASHSCOPE_API_KEY
        api_key = EMBEDDING_API_KEY or DASHSCOPE_API_KEY
        return DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=api_key
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        vectorstore_logger.info("使用OpenAI Embedding模型")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=OPENAI_API_KEY
        )
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        vectorstore_logger.info("使用Gemini Embedding模型")
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=GEMINI_API_KEY
        )
    else:
        vectorstore_logger.error(f"不支持的Embedding提供商: {provider}")
        raise ValueError(f"不支持的Embedding提供商: {provider}，支持: qwen, openai, gemini")


class VectorStore:
    """向量存储管理类，支持多学科"""

    def __init__(self, subject: str = "chinese", persist_directory: Optional[str] = None,
                 embedding_provider: Optional[str] = None):
        """
        初始化向量存储

        Args:
            subject: 学科类型 (chinese/math/english)
            persist_directory: 向量数据库持久化目录
            embedding_provider: Embedding提供商 (qwen, openai, gemini)
        """
        if subject not in SUBJECTS:
            vectorstore_logger.error(f"不支持的学科类型: {subject}")
            raise ValueError(f"不支持的学科类型: {subject}，支持的学科: {list(SUBJECTS.keys())}")

        self.subject = subject
        self.persist_directory = persist_directory or os.path.join(CHROMA_PERSIST_DIR, subject)
        self.embedding_provider = embedding_provider or EMBEDDING_PROVIDER

        vectorstore_logger.info(f"初始化向量存储 - 学科: {subject}, 持久化目录: {self.persist_directory}, Embedding提供商: {self.embedding_provider}")

        # 初始化Embedding模型
        self.embeddings = create_embeddings(self.embedding_provider)

        self.vectorstore: Optional[Chroma] = None

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        从文档创建向量存储

        Args:
            documents: Document对象列表

        Returns:
            Chroma向量存储对象
        """
        vectorstore_logger.info(f"========== 开始创建向量存储 - 学科: {self.subject} ==========")
        vectorstore_logger.info(f"文档数量: {len(documents)}")

        # 为每个文档添加学科标签
        for doc in documents:
            doc.metadata["subject"] = self.subject

        vectorstore_logger.info("正在生成向量嵌入...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        # 持久化存储
        self.vectorstore.persist()
        vectorstore_logger.info(f"向量存储创建完成并已持久化 - 路径: {self.persist_directory}")
        vectorstore_logger.info(f"========== 向量存储创建成功 ==========")
        return self.vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        加载已有的向量存储

        Returns:
            Chroma向量存储对象，如果不存在则返回None
        """
        vectorstore_logger.info(f"尝试加载向量存储 - 学科: {self.subject}, 路径: {self.persist_directory}")

        if not os.path.exists(self.persist_directory):
            vectorstore_logger.info(f"向量存储目录不存在: {self.persist_directory}")
            return None

        # 检查目录是否为空
        if not os.listdir(self.persist_directory):
            vectorstore_logger.info(f"向量存储目录为空: {self.persist_directory}")
            return None

        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            count = self.vectorstore._collection.count()
            vectorstore_logger.info(f"向量存储加载成功 - 学科: {self.subject}, 文档数: {count}")
            return self.vectorstore
        except Exception as e:
            vectorstore_logger.error(f"加载向量存储失败: {str(e)}")
            return None

    def add_documents(self, documents: List[Document]) -> None:
        """
        向现有向量存储添加文档

        Args:
            documents: 要添加的Document对象列表
        """
        vectorstore_logger.info(f"========== 开始添加文档到向量存储 - 学科: {self.subject} ==========")
        vectorstore_logger.info(f"待添加文档数量: {len(documents)}")

        # 为每个文档添加学科标签
        for doc in documents:
            doc.metadata["subject"] = self.subject

        if self.vectorstore is None:
            vectorstore_logger.info("向量存储不存在，创建新的向量存储")
            self.create_vectorstore(documents)
        else:
            vectorstore_logger.info("正在向现有向量存储添加文档...")
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            vectorstore_logger.info("文档添加完成并已持久化")

        total_count = self.vectorstore._collection.count()
        vectorstore_logger.info(f"当前向量存储总文档数: {total_count}")
        vectorstore_logger.info(f"========== 文档添加成功 ==========")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似的Document对象列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化，请先加载或创建向量存储")

        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        带分数的相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            (Document, score) 元组列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化，请先加载或创建向量存储")

        return self.vectorstore.similarity_search_with_score(query, k=k)

    def delete_collection(self) -> None:
        """删除整个向量存储集合"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None

    def get_document_count(self) -> int:
        """
        获取向量存储中的文档数量

        Returns:
            文档数量
        """
        if self.vectorstore is None:
            return 0
        return self.vectorstore._collection.count()


class MultiSubjectVectorStore:
    """多学科向量存储管理器"""

    def __init__(self):
        """初始化多学科向量存储管理器"""
        self._stores: Dict[str, VectorStore] = {}

    def get_store(self, subject: str) -> VectorStore:
        """
        获取指定学科的向量存储

        Args:
            subject: 学科类型

        Returns:
            VectorStore实例
        """
        if subject not in self._stores:
            self._stores[subject] = VectorStore(subject)
            self._stores[subject].load_vectorstore()
        return self._stores[subject]

    def get_all_subjects_info(self) -> Dict[str, Dict]:
        """
        获取所有学科的信息

        Returns:
            学科信息字典
        """
        info = {}
        for subject_key, subject_info in SUBJECTS.items():
            store = self.get_store(subject_key)
            info[subject_key] = {
                "name": subject_info["name"],
                "icon": subject_info["icon"],
                "description": subject_info["description"],
                "document_count": store.get_document_count()
            }
        return info
