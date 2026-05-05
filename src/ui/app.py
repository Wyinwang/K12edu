"""
Streamlit Web界面
小学智能教育问答系统
支持多学科知识库、练习题生成、学习进度跟踪、错题本、用户管理
"""
# 过滤所有警告信息 - 必须在所有导入之前
import warnings
import logging
import os
import sys

# 禁用 transformers 库的警告输出 - 必须在导入 transformers 之前设置
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

import streamlit as st

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_engine.engine import MultiSubjectRAGEngine
from src.core.exercise_generator import ExerciseGenerator
from src.auth import is_staff, is_admin, verify_login, init_database_users
from src.tracking.learning_tracker import LearningTracker
from src.tracking.weak_point_analyzer import WrongQuestionManager, WeakPointAnalyzer
from src.user.user_manager import UserManager
from config.settings import (
    ROLE_ADMIN,
    ROLE_STUDENT,
    ROLE_TEACHER,
    UPLOAD_DIR,
    SUBJECTS,
    LLM_PROVIDER,
    DASHSCOPE_MODEL,
    OLLAMA_MODEL
)

# 页面配置
st.set_page_config(
    page_title="小学智能教育问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    .source-box {
        background-color: #FFF8E1;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .subject-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .exercise-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .correct-answer {
        background-color: #E8F5E9;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #4CAF50;
    }
    .wrong-answer {
        background-color: #FFEBEE;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #F44336;
    }
    .score-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .wrong-question-card {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #FF9800;
    }
    .knowledge-point-tag {
        background-color: #E1F5FE;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .stat-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .user-table {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    # 初始化数据库用户
    init_database_users()

    if "rag_engines" not in st.session_state:
        st.session_state.rag_engines = MultiSubjectRAGEngine()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
        for subject in SUBJECTS.keys():
            st.session_state.chat_history[subject] = []
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = "chinese"
    if "exercises" not in st.session_state:
        st.session_state.exercises = None
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "exercise_submitted" not in st.session_state:
        st.session_state.exercise_submitted = False
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None
    if "auth_username" not in st.session_state:
        st.session_state.auth_username = None
    if "auth_user_id" not in st.session_state:
        st.session_state.auth_user_id = None
    # LLM模型选择状态
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = LLM_PROVIDER
    if "llm_model" not in st.session_state:
        # 根据默认 provider 设置默认 model
        if LLM_PROVIDER == "qwen":
            st.session_state.llm_model = DASHSCOPE_MODEL
        elif LLM_PROVIDER == "ollama":
            st.session_state.llm_model = OLLAMA_MODEL
        else:
            st.session_state.llm_model = None

    # 学习跟踪器实例
    if "learning_tracker" not in st.session_state:
        st.session_state.learning_tracker = LearningTracker()
    if "wrong_question_manager" not in st.session_state:
        st.session_state.wrong_question_manager = WrongQuestionManager()
    if "user_manager" not in st.session_state:
        st.session_state.user_manager = UserManager()


def _role_label(role: str) -> str:
    return {
        ROLE_STUDENT: "学生",
        ROLE_TEACHER: "教师",
        ROLE_ADMIN: "管理员",
    }.get(role, role)


def render_login():
    """未登录时显示登录表单"""
    st.markdown(
        '<h1 class="main-header">📚 小学智能教育问答系统</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align: center; color: #757575; margin-bottom: 1.5rem;">
            请先登录。学生账号仅可使用「学科问答」与「出题」；教材上传与知识库管理需教师或管理员账号。
        </div>
        """,
        unsafe_allow_html=True,
    )
    _c1, c2, _c3 = st.columns([1, 1.2, 1])
    with c2:
        username = st.text_input("用户名", key="login_username")
        password = st.text_input("密码", type="password", key="login_password")
        if st.button("登录", type="primary", use_container_width=True):
            # 先尝试数据库验证获取用户ID
            user_manager = st.session_state.user_manager
            user_info = user_manager.verify_user_login(username, password)
            if user_info:
                st.session_state.auth_role = user_info['role']
                st.session_state.auth_username = username.strip()
                st.session_state.auth_user_id = user_info['id']
                st.rerun()
            else:
                # 兜底配置文件验证
                role = verify_login(username, password)
                if role:
                    st.session_state.auth_role = role
                    st.session_state.auth_username = username.strip()
                    st.rerun()
                else:
                    st.error("用户名或密码错误")


def get_rag_engine(subject: str):
    """获取指定学科的RAG引擎"""
    return st.session_state.rag_engines.get_engine(subject)


def render_sidebar(staff: bool, admin: bool):
    """渲染侧边栏"""
    with st.sidebar:
        user = st.session_state.auth_username or ""
        role = st.session_state.auth_role or ""
        user_id = st.session_state.auth_user_id
        st.markdown(f"**已登录：** {user}（{_role_label(role)}）")
        if st.button("退出登录", use_container_width=True):
            st.session_state.auth_role = None
            st.session_state.auth_username = None
            st.session_state.auth_user_id = None
            st.rerun()

        st.markdown("---")

        # 学科选择
        st.markdown("### 📚 选择学科")

        cols = st.columns(3)
        for i, (subject_key, subject_info) in enumerate(SUBJECTS.items()):
            with cols[i]:
                is_current = st.session_state.current_subject == subject_key
                btn_type = "primary" if is_current else "secondary"
                if st.button(
                    f"{subject_info['icon']}\n{subject_info['name']}",
                    key=f"subject_{subject_key}",
                    use_container_width=True,
                    type=btn_type
                ):
                    st.session_state.current_subject = subject_key
                    st.session_state.exercises = None
                    st.session_state.exercise_submitted = False
                    st.rerun()

        st.markdown("---")

        # 模型选择
        st.markdown("### 🤖 选择模型")

        llm_options = {
            "阿里云 Qwen3.5-Plus": {"provider": "qwen", "model": "qwen3.5-plus"},
            "本地 Ollama Qwen2.5-14B": {"provider": "ollama", "model": "qwen2.5:14b"},
        }

        current_model_key = None
        for key, val in llm_options.items():
            if st.session_state.llm_provider == val["provider"] and st.session_state.llm_model == val["model"]:
                current_model_key = key
                break

        if current_model_key is None:
            current_model_key = list(llm_options.keys())[0]

        selected_model = st.selectbox(
            "选择问答模型",
            list(llm_options.keys()),
            index=list(llm_options.keys()).index(current_model_key),
            help="选择用于问答和出题的大语言模型"
        )

        new_provider = llm_options[selected_model]["provider"]
        new_model = llm_options[selected_model]["model"]
        if new_provider != st.session_state.llm_provider or new_model != st.session_state.llm_model:
            st.session_state.llm_provider = new_provider
            st.session_state.llm_model = new_model
            st.session_state.rag_engines.switch_llm(new_provider, new_model)
            st.rerun()

        st.markdown("---")

        # 当前学科知识库状态
        current_subject = st.session_state.current_subject
        subject_info = SUBJECTS[current_subject]
        engine = get_rag_engine(current_subject)
        kb_info = engine.get_knowledge_base_info()

        st.markdown(f"### {subject_info['icon']} {subject_info['name']}知识库")
        st.markdown(f"""
        <div class="info-box">
            <strong>知识库状态</strong><br>
            文档块数量: {kb_info['document_count']}<br>
            当前学科: {kb_info['subject_name']}
        </div>
        """, unsafe_allow_html=True)

        # 显示学习进度摘要
        if user_id:
            tracker = st.session_state.learning_tracker
            learning_days = tracker.get_user_learning_days(user_id, days=30)
            accuracy = tracker.get_user_accuracy(user_id, days=30)
            avg_accuracy = sum(accuracy.values()) / len(accuracy) if accuracy else 0

            st.markdown(f"""
            <div class="stat-card">
                <strong>📊 近30天学习</strong><br>
                学习天数: {learning_days}天<br>
                平均正确率: {avg_accuracy:.1f}%
            </div>
            """, unsafe_allow_html=True)

        if staff:
            with st.expander("📊 全部学科概览"):
                all_info = st.session_state.rag_engines.get_all_subjects_info()
                for sk, si in all_info.items():
                    st.markdown(f"""
                    <div class="subject-card">
                        {si['subject_icon']} <strong>{si['subject_name']}</strong><br>
                        文档块: {si['document_count']}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # 文档上传区域
            st.markdown("### 📤 上传教材")
            uploaded_files = st.file_uploader(
                f"上传{subject_info['name']}教材",
                type=["pdf", "docx", "doc"],
                accept_multiple_files=True,
                help="支持PDF和Word格式的教材文件",
            )

            if uploaded_files:
                if st.button("添加到知识库", type="primary"):
                    for uploaded_file in uploaded_files:
                        subject_upload_dir = os.path.join(UPLOAD_DIR, current_subject)
                        os.makedirs(subject_upload_dir, exist_ok=True)
                        file_path = os.path.join(subject_upload_dir, uploaded_file.name)

                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        with st.spinner(f"正在处理: {uploaded_file.name}"):
                            result = engine.add_document(file_path)

                        if result["success"]:
                            st.success(f"✅ {uploaded_file.name}: {result['message']}")
                        else:
                            st.error(f"❌ {uploaded_file.name}: {result['message']}")

                    st.rerun()

            st.markdown("---")
        else:
            st.caption(
                "教材上传与知识库管理由教师/管理员操作；你可使用学科问答与出题。"
            )
            st.markdown("---")

        # 设置选项
        st.markdown("### ⚙️ 设置")
        use_rag = st.checkbox("启用RAG检索", value=True, help="使用知识库中的教材辅助回答")
        show_sources = st.checkbox("显示参考来源", value=True, help="显示回答参考的教材内容")

        st.markdown("---")

        # 清空操作
        st.markdown("### 🗑️ 管理")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("清空对话", use_container_width=True):
                st.session_state.chat_history[current_subject] = []
                st.rerun()

        with col2:
            if staff:
                if st.button("清空知识库", use_container_width=True):
                    if st.session_state.get("confirm_clear"):
                        result = engine.clear_knowledge_base()
                        if result["success"]:
                            st.success(result["message"])
                            st.session_state.confirm_clear = False
                            st.rerun()
                        else:
                            st.error(result["message"])
                    else:
                        st.session_state.confirm_clear = True
                        st.warning("再次点击确认清空")

        return use_rag, show_sources


def render_chat_message(role: str, content: str, sources: list = None):
    """渲染聊天消息"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 你:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 助手:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

        if sources:
            with st.expander("📖 查看参考来源"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>来源 {i}</strong> (相似度: {source['score']})<br>
                        文件: {source['source']}<br>
                        内容: {source['content']}
                    </div>
                    """, unsafe_allow_html=True)


def render_chat_history(subject: str):
    """渲染聊天历史"""
    for message in st.session_state.chat_history[subject]:
        render_chat_message(
            message["role"],
            message["content"],
            message.get("sources")
        )


def render_exercise_page(use_rag: bool):
    """渲染练习题页面"""
    current_subject = st.session_state.current_subject
    subject_info = SUBJECTS[current_subject]
    user_id = st.session_state.auth_user_id

    st.markdown(f"### 📝 {subject_info['name']} · 出题")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        question_count = st.slider("题目数量", min_value=3, max_value=10, value=5)

    with col2:
        available_types = {
            "chinese": ["选择题", "填空题", "判断题"],
            "math": ["选择题", "填空题", "判断题", "计算题"],
            "english": ["选择题", "填空题", "判断题"]
        }
        selected_types = st.multiselect(
            "题型选择",
            available_types.get(current_subject, ["选择题"]),
            default=available_types.get(current_subject, ["选择题"])[:2]
        )

    with col3:
        use_knowledge = st.checkbox("基于知识库", value=use_rag, help="使用知识库内容生成练习题")

    if st.button("🎯 生成练习题", type="primary", use_container_width=True):
        if not selected_types:
            st.warning("请至少选择一种题型")
        else:
            with st.spinner("正在生成练习题..."):
                generator = ExerciseGenerator(current_subject)

                context = None
                if use_knowledge:
                    engine = get_rag_engine(current_subject)
                    kb_info = engine.get_knowledge_base_info()
                    if kb_info["document_count"] > 0:
                        try:
                            results = engine.vector_store.similarity_search_with_score(
                                "知识点 概念 定义", k=5
                            )
                            if results:
                                context = "\n\n---\n\n".join([doc[0].page_content for doc in results])
                        except Exception:
                            pass

                result = generator.generate_exercises(
                    context=context,
                    question_types=selected_types,
                    count=question_count
                )

                if result["success"]:
                    st.session_state.exercises = result
                    st.session_state.user_answers = {}
                    st.session_state.exercise_submitted = False
                    st.success(result["message"])
                else:
                    st.error(result["message"])

    # 显示练习题
    if st.session_state.exercises:
        st.markdown("---")

        questions = st.session_state.exercises.get("questions", [])

        for i, q in enumerate(questions):
            with st.container():
                st.markdown(f"""
                <div class="exercise-card">
                    <strong>第 {i+1} 题</strong> [{q['type']}]<br><br>
                    {q['question']}
                </div>
                """, unsafe_allow_html=True)

                if q["type"] == "选择题":
                    options = q.get("options", [])
                    selected = st.radio(
                        "选择答案",
                        options,
                        key=f"q_{i}",
                        index=None,
                        disabled=st.session_state.exercise_submitted
                    )
                    if selected:
                        st.session_state.user_answers[i] = selected[0]

                elif q["type"] == "判断题":
                    options = ["A. 正确", "B. 错误"]
                    selected = st.radio(
                        "选择答案",
                        options,
                        key=f"q_{i}",
                        index=None,
                        disabled=st.session_state.exercise_submitted
                    )
                    if selected:
                        st.session_state.user_answers[i] = selected[0]

                elif q["type"] == "填空题":
                    answer = st.text_input(
                        "填写答案",
                        key=f"q_{i}",
                        disabled=st.session_state.exercise_submitted
                    )
                    if answer:
                        st.session_state.user_answers[i] = answer

                elif q["type"] == "计算题":
                    answer = st.text_area(
                        "填写答案（可写解题步骤）",
                        key=f"q_{i}",
                        disabled=st.session_state.exercise_submitted
                    )
                    if answer:
                        st.session_state.user_answers[i] = answer

                # 显示答案和解析（提交后）
                if st.session_state.exercise_submitted:
                    user_ans = st.session_state.user_answers.get(i, "未作答")
                    correct_ans = q.get("answer", "")
                    explanation = q.get("explanation", "")

                    is_correct = user_ans.strip().upper() == correct_ans.strip().upper()

                    # 记录练习结果
                    if user_id:
                        tracker = st.session_state.learning_tracker
                        tracker.record_exercise_result(
                            user_id, current_subject, q['question'], q['type'],
                            user_ans, correct_ans, is_correct
                        )

                        # 如果答错，添加到错题本
                        if not is_correct:
                            wrong_manager = st.session_state.wrong_question_manager
                            wrong_manager.add_wrong_question(
                                user_id, current_subject, q['question'], q['type'],
                                correct_ans, user_ans, explanation
                            )

                    if is_correct:
                        st.markdown(f"""
                        <div class="correct-answer">
                            ✅ <strong>正确!</strong> 你的答案: {user_ans}<br>
                            解析: {explanation}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="wrong-answer">
                            ❌ <strong>错误</strong><br>
                            你的答案: {user_ans}<br>
                            正确答案: {correct_ans}<br>
                            解析: {explanation}
                        </div>
                        """, unsafe_allow_html=True)

        # 提交按钮
        if not st.session_state.exercise_submitted:
            if st.button("📤 提交答案", type="primary", use_container_width=True):
                if len(st.session_state.user_answers) < len(questions):
                    st.warning("请完成所有题目后再提交")
                else:
                    st.session_state.exercise_submitted = True
                    st.rerun()

        # 显示成绩
        if st.session_state.exercise_submitted:
            correct = 0
            for i, q in enumerate(questions):
                user_ans = st.session_state.user_answers.get(i, "")
                correct_ans = q.get("answer", "")
                if user_ans.strip().upper() == correct_ans.strip().upper():
                    correct += 1

            total = len(questions)
            score = round(correct / total * 100, 1)

            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if score >= 80:
                    color = "#4CAF50"
                    emoji = "🎉"
                elif score >= 60:
                    color = "#FF9800"
                    emoji = "💪"
                else:
                    color = "#F44336"
                    emoji = "📚"

                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background-color: #FAFAFA; border-radius: 1rem;">
                    <div style="font-size: 3rem;">{emoji}</div>
                    <div class="score-display" style="color: {color};">{score}分</div>
                    <div style="color: #757575;">答对 {correct}/{total} 题</div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🔄 再做一组", use_container_width=True):
                st.session_state.exercises = None
                st.session_state.user_answers = {}
                st.session_state.exercise_submitted = False
                st.rerun()


def render_learning_progress_page():
    """渲染学习进度页面"""
    user_id = st.session_state.auth_user_id
    tracker = st.session_state.learning_tracker
    current_subject = st.session_state.current_subject
    subject_info = SUBJECTS[current_subject]

    st.markdown(f"### 📊 {subject_info['name']} · 学习进度")

    # 获取用户学习概况
    summary = tracker.get_user_summary(user_id, days=30)

    # 显示三项核心统计
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem; color: #1E88E5;">📅</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{summary['learning_days']}</div>
            <div style="color: #757575;">累计学习天数</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        qa_count = summary['qa_count_by_subject'].get(current_subject, 0)
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem; color: #43A047;">💬</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{qa_count}</div>
            <div style="color: #757575;">问答次数</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        accuracy = summary['accuracy_by_subject'].get(current_subject, 0)
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem; color: #FB8C00;">🎯</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{accuracy}%</div>
            <div style="color: #757575;">练习正确率</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 各学科问答次数分布（柱状图）
    st.markdown("#### 📈 各学科问答次数")
    qa_data = summary['qa_count_by_subject']
    if any(qa_data.values()):
        st.bar_chart({
            SUBJECTS[k]['name']: v for k, v in qa_data.items()
        })
    else:
        st.info("暂无问答数据")

    st.markdown("---")

    # 正确率趋势（折线图）
    st.markdown("#### 📉 正确率趋势")
    trend_data = summary['accuracy_trend']
    if trend_data:
        import pandas as pd
        df = pd.DataFrame(trend_data)
        df = df.set_index('date')
        st.line_chart(df['accuracy'])
    else:
        st.info("暂无练习数据")


def render_wrong_questions_page():
    """渲染错题本页面"""
    user_id = st.session_state.auth_user_id
    wrong_manager = st.session_state.wrong_question_manager
    weak_analyzer = WeakPointAnalyzer()
    current_subject = st.session_state.current_subject
    subject_info = SUBJECTS[current_subject]

    st.markdown(f"### 📒 {subject_info['name']} · 错题本")

    # 获取错题列表
    wrong_questions = wrong_manager.get_wrong_questions(user_id, current_subject)

    if not wrong_questions:
        st.info("暂无错题记录，继续保持吧！")
        return

    # 显示薄弱知识点
    weak_points = weak_analyzer.get_weak_points(user_id, current_subject)
    if weak_points:
        st.markdown("#### 🔍 薄弱知识点分析")
        tags = ""
        for wp in weak_points[:5]:
            tags += f'<span class="knowledge-point-tag">{wp["knowledge_point"]} (错{wp["wrong_count"]}次)</span>'
        st.markdown(tags, unsafe_allow_html=True)
        st.markdown("---")

    # 显示错题列表
    st.markdown(f"#### 📝 错题列表 (共{len(wrong_questions)}题)")

    for i, wq in enumerate(wrong_questions):
        with st.container():
            st.markdown(f"""
            <div class="wrong-question-card">
                <strong>第 {i+1} 题</strong> [{wq['question_type']}]<br>
                <span class="knowledge-point-tag">{wq['knowledge_point']}</span><br><br>
                {wq['question']}<br><br>
                <span style="color: #F44336;">你的答案: {wq['user_answer']}</span><br>
                <span style="color: #4CAF50;">正确答案: {wq['correct_answer']}</span><br>
                <span style="color: #757575;">解析: {wq['explanation']}</span>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ 已掌握", key=f"master_{wq['id']}", use_container_width=True):
                    wrong_manager.mark_mastered(wq['id'])
                    st.success("已标记为掌握")
                    st.rerun()
            with col2:
                if st.button(f"🗑️ 删除", key=f"delete_{wq['id']}", use_container_width=True):
                    wrong_manager.delete_wrong_question(wq['id'])
                    st.success("已删除")
                    st.rerun()


def render_user_management_page():
    """渲染用户管理页面（管理员）"""
    user_manager = st.session_state.user_manager

    st.markdown("### 👥 用户管理")

    # 用户列表
    st.markdown("#### 用户列表")
    users = user_manager.get_all_users()

    if users:
        # 显示用户表格
        for user in users:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
                with col1:
                    st.write(f"**{user['username']}**")
                with col2:
                    st.write(_role_label(user['role']))
                with col3:
                    created = user['created_at'][:10] if user['created_at'] else "-"
                    st.write(f"注册: {created}")
                with col4:
                    last_login = user['last_login_at'][:10] if user['last_login_at'] else "-"
                    st.write(f"登录: {last_login}")
                with col5:
                    st.write("✅" if user['is_active'] else "❌")

        st.markdown("---")

    # 新增用户
    st.markdown("#### ➕ 新增用户")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_username = st.text_input("用户名", key="new_username")
    with col2:
        new_password = st.text_input("密码", key="new_password")
    with col3:
        new_role = st.selectbox(
            "角色",
            [ROLE_STUDENT, ROLE_TEACHER, ROLE_ADMIN],
            format_func=_role_label,
            key="new_role"
        )

    if st.button("创建用户", type="primary"):
        if new_username and new_password:
            result = user_manager.create_user(new_username, new_password, new_role)
            if result['success']:
                st.success(result['message'])
                st.rerun()
            else:
                st.error(result['message'])
        else:
            st.warning("请填写用户名和密码")

    st.markdown("---")

    # 修改用户
    st.markdown("#### 🔧 修改用户")
    if users:
        user_options = {u['id']: u['username'] for u in users}
        selected_user_id = st.selectbox(
            "选择用户",
            list(user_options.keys()),
            format_func=lambda x: user_options[x],
            key="edit_user_select"
        )

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            edit_role = st.selectbox(
                "新角色",
                [ROLE_STUDENT, ROLE_TEACHER, ROLE_ADMIN],
                format_func=_role_label,
                key="edit_role"
            )
        with col2:
            edit_password = st.text_input("新密码（留空不修改）", key="edit_password")
        with col3:
            st.write("")  # 占位

        col1, col2 = st.columns(2)
        with col1:
            if st.button("更新用户", type="primary"):
                result = user_manager.update_user(
                    selected_user_id,
                    role=edit_role,
                    password=edit_password if edit_password else None
                )
                if result['success']:
                    st.success(result['message'])
                    st.rerun()
                else:
                    st.error(result['message'])

        with col2:
            if st.button("删除用户", type="secondary"):
                if st.session_state.get("confirm_delete_user") == selected_user_id:
                    result = user_manager.delete_user(selected_user_id)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.confirm_delete_user = None
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.session_state.confirm_delete_user = selected_user_id
                    st.warning("再次点击确认删除")


def render_class_summary_page():
    """渲染班级学情页面（教师）"""
    tracker = st.session_state.learning_tracker

    st.markdown("### 📊 班级学情概览")

    # 获取班级汇总数据
    class_summary = tracker.get_class_summary(days=30)

    if not class_summary:
        st.info("暂无班级学习数据")
        return

    # 显示学生列表
    st.markdown("#### 学生学习情况排名")

    for i, student in enumerate(class_summary, 1):
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
            with col1:
                st.write(f"#{i}")
            with col2:
                st.write(f"**{student['username']}**")
            with col3:
                st.write(f"问答: {student['qa_count']}次")
            with col4:
                accuracy_color = "#4CAF50" if student['accuracy'] >= 80 else "#FF9800" if student['accuracy'] >= 60 else "#F44336"
                st.markdown(f'<span style="color: {accuracy_color};">{student["accuracy"]}%</span>', unsafe_allow_html=True)

    st.markdown("---")

    # 班级总体统计
    total_students = len(class_summary)
    avg_accuracy = sum(s['accuracy'] for s in class_summary) / total_students if total_students > 0 else 0
    total_qa = sum(s['qa_count'] for s in class_summary)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("学生人数", total_students)
    with col2:
        st.metric("班级平均正确率", f"{avg_accuracy:.1f}%")
    with col3:
        st.metric("总问答次数", total_qa)


def main():
    """主函数"""
    init_session_state()

    if st.session_state.auth_role is None:
        render_login()
        st.stop()

    current_subject = st.session_state.current_subject
    subject_info = SUBJECTS[current_subject]

    st.markdown(f'<h1 class="main-header">📚 小学智能教育问答系统</h1>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: center; color: #757575; margin-bottom: 2rem;">
        当前学科: {subject_info['icon']} {subject_info['name']} - {subject_info['description']}
    </div>
    """, unsafe_allow_html=True)

    staff = is_staff(st.session_state.auth_role)
    admin = is_admin(st.session_state.auth_role)
    use_rag, show_sources = render_sidebar(staff, admin)

    # 主内容区 - 标签页
    tabs = ["💬 学科问答", "📝 出题", "📊 学习进度", "📒 错题本"]
    if staff:
        tabs.append("👥 班级学情")
    if admin:
        tabs.append("⚙️ 用户管理")

    tab_objs = st.tabs(tabs)

    with tab_objs[0]:  # 学科问答
        st.markdown("### 💬 学科问答")
        render_chat_history(current_subject)

        user_question = st.chat_input("输入你的问题...")
        user_id = st.session_state.auth_user_id

        if user_question:
            engine = get_rag_engine(current_subject)

            render_chat_message("user", user_question)

            with st.spinner("正在思考..."):
                result = engine.query(
                    user_question,
                    use_rag=use_rag,
                    show_sources=show_sources
                )

            render_chat_message("assistant", result["answer"], result["sources"])

            # 记录问答交互
            if user_id:
                tracker = st.session_state.learning_tracker
                tracker.record_qa_interaction(
                    user_id, current_subject, user_question, result["answer"]
                )

            # 保存到历史记录
            st.session_state.chat_history[current_subject].append({
                "role": "user",
                "content": user_question
            })
            st.session_state.chat_history[current_subject].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"] if show_sources else []
            })

            st.rerun()

    with tab_objs[1]:  # 出题
        render_exercise_page(use_rag)

    with tab_objs[2]:  # 学习进度
        render_learning_progress_page()

    with tab_objs[3]:  # 错题本
        render_wrong_questions_page()

    if staff and len(tab_objs) > 4:
        with tab_objs[4]:  # 班级学情
            render_class_summary_page()

    if admin and len(tab_objs) > 5:
        with tab_objs[5]:  # 用户管理
            render_user_management_page()

    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9E9E9E; font-size: 0.9rem;">
        💡 提示：上传教材后，系统将基于教材内容回答问题和生成练习题，提高准确性
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()