"""
RAG核心引擎
整合文档处理、向量存储和大语言模型
支持多学科知识库
"""
import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.document_processor.processor import DocumentProcessor
from src.rag_engine.vector_store import VectorStore, MultiSubjectVectorStore
from src.llm.qwen_llm import QwenLLM
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SUBJECTS, UPLOAD_DIR
from src.utils.logger import rag_logger


class RAGEngine:
    """RAG引擎主类，支持多学科"""

    # 学科相关的系统提示词
    SUBJECT_PROMPTS = {
        "chinese": """你是一个专业的小学语文教育助手，专门为小学生提供语文学习辅导。
你的回答应该：
1. 使用简单易懂的语言，适合小学生理解
2. 多用生动的比喻和例子来解释语文知识
3. 对汉字、词语、句子进行细致讲解
4. 保持耐心和鼓励的态度，激发学习兴趣
5. 当提供的参考材料中有相关内容时，优先基于参考材料回答
6. 如果参考材料中没有相关内容，可以基于你的知识给出回答，但要说明这是额外补充
7. 可以回答拼音、汉字、词语、句子、阅读理解、作文、古诗词等语文相关问题""",

        "math": """你是一个专业的小学数学教育助手，专门为小学生提供数学学习辅导。
你的回答应该：
1. 使用简单易懂的语言，适合小学生理解
2. 多用生活中的例子来解释数学概念
3. 逐步讲解解题思路，帮助学生理解
4. 保持耐心和鼓励的态度，激发学习兴趣
5. 当提供的参考材料中有相关内容时，优先基于参考材料回答
6. 如果参考材料中没有相关内容，可以基于你的知识给出回答，但要说明这是额外补充
7. 可以回答加减乘除、分数、小数、几何、应用题等数学问题""",

        "english": """你是一个专业的小学英语教育助手，专门为小学生提供英语学习辅导。
你的回答应该：
1. 使用简单易懂的语言，适合小学生理解
2. 多用生动的例子和图片描述来解释英语知识
3. 发音要标注音标或用中文近似音帮助记忆
4. 保持耐心和鼓励的态度，激发学习兴趣
5. 当提供的参考材料中有相关内容时，优先基于参考材料回答
6. 如果参考材料中没有相关内容，可以基于你的知识给出回答，但要说明这是额外补充
7. 可以回答单词、语法、句型、阅读理解等英语相关问题"""
    }

    def __init__(
        self,
        subject: str = "chinese",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K
    ):
        """
        初始化RAG引擎

        Args:
            subject: 学科类型 (chinese/math/english)
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            top_k: 检索时返回的文档数量
        """
        if subject not in SUBJECTS:
            rag_logger.error(f"不支持的学科类型: {subject}")
            raise ValueError(f"不支持的学科类型: {subject}，支持的学科: {list(SUBJECTS.keys())}")

        self.subject = subject
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        rag_logger.info(f"========== 初始化RAG引擎 ==========")
        rag_logger.info(f"学科: {subject}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, top_k: {top_k}")

        # 初始化各组件
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStore(subject)
        self.llm = QwenLLM()

        # 设置学科相关的系统提示词
        self.llm.system_prompt = self.SUBJECT_PROMPTS.get(subject, self.SUBJECT_PROMPTS["chinese"])

        # 尝试加载已有的向量存储
        self._load_existing_vectorstore()
        rag_logger.info(f"RAG引擎初始化完成 - 学科: {subject}")

    def _load_existing_vectorstore(self) -> bool:
        """
        尝试加载已有的向量存储

        Returns:
            是否成功加载
        """
        vectorstore = self.vector_store.load_vectorstore()
        loaded = vectorstore is not None
        if loaded:
            rag_logger.info(f"成功加载已有向量存储")
        else:
            rag_logger.info(f"未找到已有向量存储，将创建新的向量存储")
        return loaded

    def switch_subject(self, subject: str) -> None:
        """
        切换学科

        Args:
            subject: 新的学科类型
        """
        if subject not in SUBJECTS:
            rag_logger.error(f"不支持的学科类型: {subject}")
            raise ValueError(f"不支持的学科类型: {subject}")

        rag_logger.info(f"切换学科: {self.subject} -> {subject}")
        self.subject = subject
        self.vector_store = VectorStore(subject)
        self.llm.system_prompt = self.SUBJECT_PROMPTS.get(subject, self.SUBJECT_PROMPTS["chinese"])
        self._load_existing_vectorstore()
        rag_logger.info(f"学科切换完成: {subject}")

    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        添加文档到知识库

        Args:
            file_path: 文档路径

        Returns:
            包含处理结果的字典
        """
        rag_logger.info(f"========== 开始添加文档到知识库 ==========")
        rag_logger.info(f"文件路径: {file_path}")
        rag_logger.info(f"学科: {self.subject}")

        try:
            # 处理文档
            rag_logger.info("开始处理文档...")
            documents = self.doc_processor.process_file(file_path)
            rag_logger.info(f"文档处理完成 - 生成 {len(documents)} 个文本块")

            # 添加到向量存储
            rag_logger.info("开始添加到向量存储...")
            self.vector_store.add_documents(documents)
            rag_logger.info("文档成功添加到向量存储")

            result = {
                "success": True,
                "message": f"成功添加文档，生成了 {len(documents)} 个文本块",
                "chunk_count": len(documents)
            }
            rag_logger.info(f"========== 文档添加成功 ==========")
            return result
        except Exception as e:
            rag_logger.error(f"添加文档失败: {str(e)}")
            return {
                "success": False,
                "message": f"添加文档失败: {str(e)}",
                "chunk_count": 0
            }

    def add_documents_from_directory(self, directory: str) -> Dict[str, Any]:
        """
        从目录批量添加文档

        Args:
            directory: 目录路径

        Returns:
            包含处理结果的字典
        """
        rag_logger.info(f"========== 开始批量添加文档 ==========")
        rag_logger.info(f"目录路径: {directory}")
        rag_logger.info(f"学科: {self.subject}")

        try:
            documents = self.doc_processor.process_directory(directory)
            rag_logger.info(f"目录处理完成 - 共生成 {len(documents)} 个文本块")

            if not documents:
                rag_logger.warning(f"目录中没有找到支持的文档文件: {directory}")
                return {
                    "success": False,
                    "message": "目录中没有找到支持的文档文件",
                    "chunk_count": 0
                }

            rag_logger.info("开始添加到向量存储...")
            self.vector_store.add_documents(documents)
            rag_logger.info("文档成功添加到向量存储")

            result = {
                "success": True,
                "message": f"成功添加文档，共生成 {len(documents)} 个文本块",
                "chunk_count": len(documents)
            }
            rag_logger.info(f"========== 批量添加文档成功 ==========")
            return result
        except Exception as e:
            rag_logger.error(f"批量添加文档失败: {str(e)}")
            return {
                "success": False,
                "message": f"批量添加文档失败: {str(e)}",
                "chunk_count": 0
            }

    def query(
        self,
        question: str,
        use_rag: bool = True,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        查询问题

        Args:
            question: 用户问题
            use_rag: 是否使用RAG检索
            show_sources: 是否返回来源信息

        Returns:
            包含回答和相关信息的字典
        """
        rag_logger.info(f"========== 收到查询请求 ==========")
        rag_logger.info(f"问题: {question}")
        rag_logger.info(f"使用RAG: {use_rag}, 学科: {self.subject}")

        context = None
        sources = []

        if use_rag:
            # 检索相关文档
            try:
                rag_logger.info("开始检索相关文档...")
                results = self.vector_store.similarity_search_with_score(
                    question, k=self.top_k
                )
                rag_logger.info(f"检索完成 - 找到 {len(results)} 个相关文档")

                if results:
                    # 构建上下文
                    context_parts = []
                    for doc, score in results:
                        context_parts.append(doc.page_content)
                        if show_sources:
                            sources.append({
                                "content": doc.page_content[:200] + "...",
                                "source": doc.metadata.get("source", "未知"),
                                "page": doc.metadata.get("page", "未知"),
                                "score": round(1 - score, 4)  # 转换为相似度
                            })
                    context = "\n\n---\n\n".join(context_parts)
                    rag_logger.info(f"上下文构建完成 - 长度: {len(context)} 字符")
            except Exception as e:
                rag_logger.error(f"检索失败: {str(e)}")

        # 生成回答
        rag_logger.info("开始生成回答...")
        answer = self.llm.chat(question, context)
        rag_logger.info(f"回答生成完成 - 长度: {len(answer)} 字符")
        rag_logger.info(f"========== 查询完成 ==========")

        return {
            "answer": answer,
            "sources": sources if show_sources else [],
            "used_rag": use_rag and context is not None
        }

    def query_with_history(
        self,
        question: str,
        history: List[dict],
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        带历史记录的查询

        Args:
            question: 用户问题
            history: 历史对话记录
            use_rag: 是否使用RAG检索

        Returns:
            包含回答和相关信息的字典
        """
        rag_logger.info(f"========== 收到带历史的查询请求 ==========")
        rag_logger.info(f"问题: {question}")
        rag_logger.info(f"历史记录数: {len(history)}, 使用RAG: {use_rag}")

        context = None
        sources = []

        if use_rag:
            try:
                rag_logger.info("开始检索相关文档...")
                results = self.vector_store.similarity_search_with_score(
                    question, k=self.top_k
                )
                rag_logger.info(f"检索完成 - 找到 {len(results)} 个相关文档")

                if results:
                    context_parts = []
                    for doc, score in results:
                        context_parts.append(doc.page_content)
                        sources.append({
                            "content": doc.page_content[:200] + "...",
                            "source": doc.metadata.get("source", "未知"),
                            "score": round(1 - score, 4)
                        })
                    context = "\n\n---\n\n".join(context_parts)
                    rag_logger.info(f"上下文构建完成 - 长度: {len(context)} 字符")
            except Exception as e:
                rag_logger.error(f"检索失败: {str(e)}")

        rag_logger.info("开始生成回答...")
        answer = self.llm.chat_with_history(question, history, context)
        rag_logger.info(f"回答生成完成 - 长度: {len(answer)} 字符")
        rag_logger.info(f"========== 查询完成 ==========")

        return {
            "answer": answer,
            "sources": sources,
            "used_rag": use_rag and context is not None
        }

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        获取知识库信息

        Returns:
            知识库信息字典
        """
        count = self.vector_store.get_document_count()
        subject_info = SUBJECTS.get(self.subject, {})
        return {
            "subject": self.subject,
            "subject_name": subject_info.get("name", "未知"),
            "subject_icon": subject_info.get("icon", "📚"),
            "document_count": count,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k
        }

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        清空知识库

        Returns:
            操作结果
        """
        try:
            self.vector_store.delete_collection()
            return {
                "success": True,
                "message": "知识库已清空"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"清空知识库失败: {str(e)}"
            }


class MultiSubjectRAGEngine:
    """多学科RAG引擎管理器"""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K
    ):
        """
        初始化多学科RAG引擎管理器

        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            top_k: 检索时返回的文档数量
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._engines: Dict[str, RAGEngine] = {}

    def get_engine(self, subject: str) -> RAGEngine:
        """
        获取指定学科的RAG引擎

        Args:
            subject: 学科类型

        Returns:
            RAGEngine实例
        """
        if subject not in self._engines:
            self._engines[subject] = RAGEngine(
                subject=subject,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                top_k=self.top_k
            )
        return self._engines[subject]

    def get_all_subjects_info(self) -> Dict[str, Dict]:
        """
        获取所有学科的信息

        Returns:
            学科信息字典
        """
        info = {}
        for subject_key in SUBJECTS.keys():
            engine = self.get_engine(subject_key)
            info[subject_key] = engine.get_knowledge_base_info()
        return info
