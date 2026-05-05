"""
统一LLM接口模块 - 支持多个LLM提供商
"""
import os
from typing import Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    DASHSCOPE_API_KEY, DASHSCOPE_API_URL, DASHSCOPE_MODEL, QWEN_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
    TOUTIAO_API_KEY, TOUTIAO_MODEL,
    HUNYUAN_API_KEY, HUNYUAN_SECRET_ID, HUNYUAN_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    LLM_PROVIDER
)


class LLMFactory:
    """LLM工厂类 - 根据配置创建对应的LLM实例"""

    @staticmethod
    def create_chat_model(provider: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.7) -> BaseChatModel:
        """
        创建聊天模型实例

        Args:
            provider: LLM提供商 (qwen, deepseek, toutiao, hunyuan, openai, ollama)
            model_name: 模型名称
            temperature: 温度参数

        Returns:
            对应的Chat模型实例
        """
        provider = provider or LLM_PROVIDER

        if provider == "qwen":
            return LLMFactory._create_qwen(model_name, temperature)
        elif provider == "deepseek":
            return LLMFactory._create_deepseek(model_name, temperature)
        elif provider == "toutiao":
            return LLMFactory._create_toutiao(model_name, temperature)
        elif provider == "hunyuan":
            return LLMFactory._create_hunyuan(model_name, temperature)
        elif provider == "openai":
            return LLMFactory._create_openai(model_name, temperature)
        elif provider == "gemini":
            return LLMFactory._create_gemini(model_name, temperature)
        elif provider == "ollama":
            return LLMFactory._create_ollama(model_name, temperature)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")

    @staticmethod
    def _create_qwen(model_name: Optional[str], temperature: float) -> BaseChatModel:
        """创建通义千问模型"""
        from langchain_community.chat_models import ChatTongyi
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or DASHSCOPE_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_API_URL,
            temperature=temperature
        )

    @staticmethod
    def _create_deepseek(model_name: Optional[str], temperature: float) -> BaseChatModel:
        """创建DeepSeek模型"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_URL,
            temperature=temperature
        )

    @staticmethod
    def _create_toutiao(model_name: Optional[str], temperature: float) -> BaseChatModel:
        """创建字节跳动(豆包)模型"""
        from langchain_openai import ChatOpenAI
        # 字节跳动豆包API使用OpenAI兼容接口
        # 需要在环境变量中设置 TOUTIAO_BASE_URL 或使用默认值
        toutiao_base_url = os.getenv("TOUTIAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        return ChatOpenAI(
            model=model_name or TOUTIAO_MODEL,
            api_key=TOUTIAO_API_KEY,
            base_url=toutiao_base_url,
            temperature=temperature
        )

    @staticmethod
    def _create_hunyuan(model_name: Optional[str], temperature: float) -> BaseChatModel:
        """创建腾讯混元模型"""
        from langchain_community.chat_models import ChatTencentHunyuan
        return ChatTencentHunyuan(
            model=model_name or HUNYUAN_MODEL,
            tencent_secret_key=HUNYUAN_API_KEY,
            tencent_secret_id=HUNYUAN_SECRET_ID,
            temperature=temperature
        )

    @staticmethod
    def _create_openai(model_name: Optional[str], temperature: float) -> BaseChatModel:
        """创建OpenAI模型"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature
        )

    @staticmethod
    def _create_gemini(model_name: Optional[str], temperature: float) -> BaseChatModel:
        """创建Google Gemini模型"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature
        )

    @staticmethod
    def _create_ollama(model_name: Optional[str], temperature: float) -> BaseChatModel:
        """创建Ollama本地模型"""
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature
        )


class UnifiedLLM:
    """统一LLM接口 - 支持多个提供商"""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.7):
        """
        初始化LLM

        Args:
            provider: LLM提供商 (qwen, deepseek, toutiao, hunyuan, openai)
            model_name: 模型名称
            temperature: 温度参数
        """
        self.provider = provider or LLM_PROVIDER
        self.model_name = model_name
        self.temperature = temperature
        self.chat_model = LLMFactory.create_chat_model(provider, model_name, temperature)

        # 系统提示词 - 针对小学教育的角色设定
        self.system_prompt = """你是一个专业的小学教育助手，专门为小学生提供学习辅导。
你的回答应该：
1. 使用简单易懂的语言，适合小学生理解
2. 多用生动的比喻和例子来解释各种知识
3. 对复杂概念进行拆解，循序渐进地讲解
4. 保持耐心和鼓励的态度，激发学习兴趣
5. 当提供的参考材料中有相关内容时，优先基于参考材料回答
6. 如果参考材料中没有相关内容，可以基于你的知识给出回答，但要说明这是额外补充
7. 可以回答语文、数学、英语、科学、常识等小学各学科的问题"""

    def chat(self, message: str, context: Optional[str] = None) -> str:
        """
        进行对话

        Args:
            message: 用户消息
            context: 可选的上下文信息

        Returns:
            模型回复
        """
        messages = [SystemMessage(content=self.system_prompt)]

        if context:
            context_message = f"""以下是相关的参考材料，请在回答时参考：

{context}

请基于以上参考材料回答用户的问题。"""
            messages.append(SystemMessage(content=context_message))

        messages.append(HumanMessage(content=message))

        response = self.chat_model.invoke(messages)
        return response.content

    def chat_with_history(
        self,
        message: str,
        history: List[dict],
        context: Optional[str] = None
    ) -> str:
        """
        带历史记录的对话

        Args:
            message: 用户消息
            history: 历史对话记录
            context: 可选的上下文信息

        Returns:
            模型回复
        """
        messages = [SystemMessage(content=self.system_prompt)]

        if context:
            context_message = f"""以下是相关的参考材料，请在回答时参考：

{context}

请基于以上参考材料回答用户的问题。"""
            messages.append(SystemMessage(content=context_message))

        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        response = self.chat_model.invoke(messages)
        return response.content

    def generate_stream(self, message: str, context: Optional[str] = None):
        """
        流式生成回复

        Args:
            message: 用户消息
            context: 可选的上下文信息

        Yields:
            生成的文本片段
        """
        messages = [SystemMessage(content=self.system_prompt)]

        if context:
            context_message = f"""以下是相关的参考材料，请在回答时参考：

{context}

请基于以上参考材料回答用户的问题。"""
            messages.append(SystemMessage(content=context_message))

        messages.append(HumanMessage(content=message))

        for chunk in self.chat_model.stream(messages):
            yield chunk.content


# 为了向后兼容，保留 QwenLLM 作为别名
QwenLLM = UnifiedLLM
