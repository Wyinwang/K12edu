"""
练习题生成模块
基于大语言模型生成各学科练习题
"""
import json
import re
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.qwen_llm import QwenLLM
from config.settings import SUBJECTS


class ExerciseGenerator:
    """练习题生成器"""

    # 各学科的练习题生成提示词模板
    SUBJECT_PROMPTS = {
        "chinese": """你是一个小学语文老师，请根据以下内容生成练习题。

参考内容：
{context}

请生成 {count} 道练习题，类型包括：{question_types}。

要求：
1. 题目难度适合小学生
2. 题目要有教育意义，能帮助学生巩固知识
3. 每道题都要有标准答案和解析

请严格按照以下JSON格式输出，不要包含其他文字：
{{
    "questions": [
        {{
            "type": "选择题|填空题|判断题",
            "question": "题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "answer": "正确答案",
            "explanation": "答案解析"
        }}
    ]
}}

注意：
- 选择题必须有4个选项
- 填空题options字段为空列表
- 判断题options为["A. 正确", "B. 错误"]""",

        "math": """你是一个小学数学老师，请根据以下内容生成练习题。

参考内容：
{context}

请生成 {count} 道练习题，类型包括：{question_types}。

要求：
1. 题目难度适合小学生
2. 计算题要有明确的解题步骤
3. 应用题要贴近生活实际
4. 每道题都要有标准答案和解析

请严格按照以下JSON格式输出，不要包含其他文字：
{{
    "questions": [
        {{
            "type": "选择题|填空题|判断题|计算题",
            "question": "题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "answer": "正确答案",
            "explanation": "答案解析和解题步骤"
        }}
    ]
}}

注意：
- 选择题必须有4个选项
- 填空题options字段为空列表
- 计算题需要给出详细的解题步骤""",

        "english": """你是一个小学英语老师，请根据以下内容生成练习题。

参考内容：
{context}

请生成 {count} 道练习题，类型包括：{question_types}。

要求：
1. 题目难度适合小学生
2. 可以考查单词、语法、句型等
3. 每道题都要有标准答案和解析
4. 解析用中文说明

请严格按照以下JSON格式输出，不要包含其他文字：
{{
    "questions": [
        {{
            "type": "选择题|填空题|判断题",
            "question": "题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "answer": "正确答案",
            "explanation": "答案解析（中文）"
        }}
    ]
}}

注意：
- 选择题必须有4个选项
- 填空题options字段为空列表
- 判断题options为["A. 正确", "B. 错误"]"""
    }

    # 默认提示词（当没有上下文时使用）
    DEFAULT_PROMPTS = {
        "chinese": """你是一个小学语文老师，请生成 {count} 道小学语文练习题。

题目类型包括：{question_types}。

要求：
1. 题目难度适合小学生（1-6年级）
2. 可以涉及拼音、汉字、词语、句子、阅读理解等
3. 每道题都要有标准答案和解析

请严格按照以下JSON格式输出，不要包含其他文字：
{{
    "questions": [
        {{
            "type": "选择题|填空题|判断题",
            "question": "题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "answer": "正确答案",
            "explanation": "答案解析"
        }}
    ]
}}""",

        "math": """你是一个小学数学老师，请生成 {count} 道小学数学练习题。

题目类型包括：{question_types}。

要求：
1. 题目难度适合小学生（1-6年级）
2. 可以涉及加减乘除、分数、小数、几何、应用题等
3. 每道题都要有标准答案和解析

请严格按照以下JSON格式输出，不要包含其他文字：
{{
    "questions": [
        {{
            "type": "选择题|填空题|判断题|计算题",
            "question": "题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "answer": "正确答案",
            "explanation": "答案解析和解题步骤"
        }}
    ]
}}""",

        "english": """你是一个小学英语老师，请生成 {count} 道小学英语练习题。

题目类型包括：{question_types}。

要求：
1. 题目难度适合小学生（1-6年级）
2. 可以涉及单词、语法、句型等
3. 每道题都要有标准答案和解析（解析用中文）

请严格按照以下JSON格式输出，不要包含其他文字：
{{
    "questions": [
        {{
            "type": "选择题|填空题|判断题",
            "question": "题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "answer": "正确答案",
            "explanation": "答案解析（中文）"
        }}
    ]
}}"""
    }

    # 各学科支持的题型
    QUESTION_TYPES = {
        "chinese": ["选择题", "填空题", "判断题"],
        "math": ["选择题", "填空题", "判断题", "计算题"],
        "english": ["选择题", "填空题", "判断题"]
    }

    def __init__(self, subject: str = "chinese"):
        """
        初始化练习题生成器

        Args:
            subject: 学科类型 (chinese/math/english)
        """
        if subject not in SUBJECTS:
            raise ValueError(f"不支持的学科类型: {subject}")

        self.subject = subject
        self.llm = QwenLLM()

    def generate_exercises(
        self,
        context: Optional[str] = None,
        question_types: Optional[List[str]] = None,
        count: int = 5
    ) -> Dict[str, Any]:
        """
        生成练习题

        Args:
            context: 参考内容（来自知识库检索）
            question_types: 题型列表，默认为该学科支持的所有题型
            count: 生成题目数量

        Returns:
            包含练习题的字典
        """
        # 设置默认题型
        if question_types is None:
            question_types = self.QUESTION_TYPES.get(self.subject, ["选择题", "填空题"])

        # 验证题型
        valid_types = self.QUESTION_TYPES.get(self.subject, [])
        question_types = [qt for qt in question_types if qt in valid_types]
        if not question_types:
            question_types = valid_types[:2] if len(valid_types) >= 2 else valid_types

        question_types_str = "、".join(question_types)

        # 选择提示词模板
        if context:
            prompt_template = self.SUBJECT_PROMPTS.get(self.subject, self.SUBJECT_PROMPTS["chinese"])
            prompt = prompt_template.format(
                context=context,
                count=count,
                question_types=question_types_str
            )
        else:
            prompt_template = self.DEFAULT_PROMPTS.get(self.subject, self.DEFAULT_PROMPTS["chinese"])
            prompt = prompt_template.format(
                count=count,
                question_types=question_types_str
            )

        # 调用LLM生成练习题
        try:
            response = self.llm.chat(prompt)

            # 解析JSON响应
            exercises = self._parse_response(response)

            return {
                "success": True,
                "subject": self.subject,
                "subject_name": SUBJECTS[self.subject]["name"],
                "question_count": len(exercises.get("questions", [])),
                "questions": exercises.get("questions", []),
                "message": f"成功生成 {len(exercises.get('questions', []))} 道练习题"
            }
        except Exception as e:
            return {
                "success": False,
                "subject": self.subject,
                "subject_name": SUBJECTS[self.subject]["name"],
                "question_count": 0,
                "questions": [],
                "message": f"生成练习题失败: {str(e)}"
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应，提取JSON

        Args:
            response: LLM响应文本

        Returns:
            解析后的字典
        """
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON块
        json_pattern = r'\{[\s\S]*"questions"[\s\S]*\}'
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 尝试提取```json```块
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        code_matches = re.findall(code_block_pattern, response)

        for match in code_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 解析失败，返回空结果
        return {"questions": []}

    def check_answer(
        self,
        question: str,
        user_answer: str,
        correct_answer: str,
        question_type: str
    ) -> Dict[str, Any]:
        """
        检查用户答案

        Args:
            question: 题目内容
            user_answer: 用户答案
            correct_answer: 正确答案
            question_type: 题目类型

        Returns:
            检查结果
        """
        # 标准化答案
        user_answer = user_answer.strip().upper()
        correct_answer = correct_answer.strip().upper()

        # 判断题特殊处理
        if question_type == "判断题":
            # 将各种表示方式统一
            correct_map = {
                "正确": "A", "对": "A", "√": "A", "T": "A", "TRUE": "A",
                "错误": "B", "错": "B", "×": "B", "F": "B", "FALSE": "B"
            }
            user_normalized = correct_map.get(user_answer, user_answer)
            correct_normalized = correct_map.get(correct_answer, correct_answer)
            is_correct = user_normalized == correct_normalized
        else:
            # 选择题：比较选项字母
            if question_type == "选择题":
                # 提取选项字母
                user_letter = user_answer[0] if user_answer else ""
                correct_letter = correct_answer[0] if correct_answer else ""
                is_correct = user_letter == correct_letter
            else:
                # 填空题、计算题：直接比较
                is_correct = user_answer == correct_answer

        return {
            "is_correct": is_correct,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "question_type": question_type
        }

    def evaluate_exercises(
        self,
        questions: List[Dict],
        user_answers: List[str]
    ) -> Dict[str, Any]:
        """
        批量评估练习题答案

        Args:
            questions: 题目列表
            user_answers: 用户答案列表

        Returns:
            评估结果
        """
        results = []
        correct_count = 0

        for i, (question, user_answer) in enumerate(zip(questions, user_answers)):
            check_result = self.check_answer(
                question["question"],
                user_answer,
                question["answer"],
                question["type"]
            )
            check_result["index"] = i + 1
            check_result["question"] = question["question"]
            check_result["explanation"] = question.get("explanation", "")

            if check_result["is_correct"]:
                correct_count += 1

            results.append(check_result)

        total_count = len(questions)
        score = round(correct_count / total_count * 100, 1) if total_count > 0 else 0

        return {
            "total_count": total_count,
            "correct_count": correct_count,
            "wrong_count": total_count - correct_count,
            "score": score,
            "results": results
        }
