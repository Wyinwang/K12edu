"""
文档处理模块
支持PDF和DOCX文件的解析和分块
"""
import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import document_logger


class DocumentProcessor:
    """文档处理器，负责解析和分块文档"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文档处理器

        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        document_logger.info(f"文档处理器初始化完成 - chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        加载PDF文件

        Args:
            file_path: PDF文件路径

        Returns:
            Document对象列表
        """
        from langchain_community.document_loaders import PyPDFLoader

        document_logger.info(f"开始加载PDF文件: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 添加文件名作为元数据
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["file_type"] = "pdf"

        document_logger.info(f"PDF文件加载完成 - 文件: {filename}, 页数: {len(documents)}")
        return documents

    def load_docx(self, file_path: str) -> List[Document]:
        """
        加载DOCX文件

        Args:
            file_path: DOCX文件路径

        Returns:
            Document对象列表
        """
        from langchain_community.document_loaders import Docx2txtLoader

        document_logger.info(f"开始加载DOCX文件: {file_path}")
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

        # 添加文件名作为元数据
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["file_type"] = "docx"

        document_logger.info(f"DOCX文件加载完成 - 文件: {filename}, 文档数: {len(documents)}")
        return documents

    def load_document(self, file_path: str) -> List[Document]:
        """
        根据文件类型加载文档

        Args:
            file_path: 文件路径

        Returns:
            Document对象列表

        Raises:
            ValueError: 不支持的文件类型
        """
        ext = os.path.splitext(file_path)[1].lower()
        document_logger.info(f"识别文件类型: {ext}, 文件: {file_path}")

        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return self.load_docx(file_path)
        else:
            document_logger.error(f"不支持的文件类型: {ext}")
            raise ValueError(f"不支持的文件类型: {ext}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成较小的块

        Args:
            documents: Document对象列表

        Returns:
            分割后的Document对象列表
        """
        document_logger.info(f"开始分割文档 - 原始文档数: {len(documents)}")
        split_docs = self.text_splitter.split_documents(documents)
        document_logger.info(f"文档分割完成 - 生成文本块数: {len(split_docs)}")
        return split_docs

    def process_file(self, file_path: str) -> List[Document]:
        """
        处理单个文件：加载并分块

        Args:
            file_path: 文件路径

        Returns:
            处理后的Document对象列表
        """
        document_logger.info(f"========== 开始处理文件: {file_path} ==========")
        filename = os.path.basename(file_path)

        try:
            # 加载文档
            document_logger.info(f"正在加载文档: {filename}")
            documents = self.load_document(file_path)
            document_logger.info(f"文档加载成功 - 页数/文档数: {len(documents)}")

            # 分块
            document_logger.info(f"正在分割文档: {filename}")
            split_docs = self.split_documents(documents)
            document_logger.info(f"文档分割成功 - 文本块数: {len(split_docs)}")

            document_logger.info(f"========== 文件处理完成: {filename}, 共生成 {len(split_docs)} 个文本块 ==========")
            return split_docs

        except Exception as e:
            document_logger.error(f"处理文件失败: {filename}, 错误: {str(e)}")
            raise

    def process_directory(self, directory: str) -> List[Document]:
        """
        处理目录下的所有支持文件

        Args:
            directory: 目录路径

        Returns:
            所有处理后的Document对象列表
        """
        all_documents = []
        supported_extensions = [".pdf", ".docx", ".doc"]
        document_logger.info(f"========== 开始处理目录: {directory} ==========")

        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                file_path = os.path.join(directory, filename)
                try:
                    docs = self.process_file(file_path)
                    all_documents.extend(docs)
                    document_logger.info(f"成功处理文件: {filename}, 生成 {len(docs)} 个文本块")
                except Exception as e:
                    document_logger.error(f"处理文件 {filename} 时出错: {str(e)}")

        document_logger.info(f"========== 目录处理完成 - 共处理 {len(all_documents)} 个文本块 ==========")
        return all_documents
