"""
通义千问LLM接口模块
注意: 此模块已迁移到 unified_llm.py，保留此文件用于向后兼容
新代码请使用 unified_llm.py 中的 UnifiedLLM 或 LLMFactory
"""
from .unified_llm import UnifiedLLM, LLMFactory, QwenLLM

# 向后兼容：导出原来的类
__all__ = ['QwenLLM', 'UnifiedLLM', 'LLMFactory']
