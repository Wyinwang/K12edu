"""
Streamlit Web界面
小学智能教育问答系统
支持多学科知识库和练习题生成
"""
import os
import sys
import streamlit as st

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_engine.engine import MultiSubjectRAGEngine
from src.core.exercise_generator import ExerciseGenerator
from src.auth import is_staff, verify_login
from config.settings import (
    ROLE_ADMIN,
    ROLE_STUDENT,
    ROLE_TEACHER,
    UPLOAD_DIR,
    SUBJECTS,
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
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
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


def render_sidebar(staff: bool):
    """渲染侧边栏。staff 为 False（学生）时不展示教材上传与知识库危险操作。"""
    with st.sidebar:
        user = st.session_state.auth_username or ""
        role = st.session_state.auth_role or ""
        st.markdown(f"**已登录：** {user}（{_role_label(role)}）")
        if st.button("退出登录", use_container_width=True):
            st.session_state.auth_role = None
            st.session_state.auth_username = None
            st.rerun()

        st.markdown("---")

        # 学科选择
        st.markdown("### 📚 选择学科")

        # 学科选择按钮
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

        if staff:
            # 显示所有学科知识库概览
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
            # 学生无「清空知识库」权限，第二列留空以保持布局

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

        # 显示来源
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

    st.markdown(f"### 📝 {subject_info['name']} · 出题")

    # 练习题设置
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

    # 生成按钮
    if st.button("🎯 生成练习题", type="primary", use_container_width=True):
        if not selected_types:
            st.warning("请至少选择一种题型")
        else:
            with st.spinner("正在生成练习题..."):
                generator = ExerciseGenerator(current_subject)

                # 获取知识库上下文
                context = None
                if use_knowledge:
                    engine = get_rag_engine(current_subject)
                    kb_info = engine.get_knowledge_base_info()
                    if kb_info["document_count"] > 0:
                        # 用一个通用查询获取相关内容
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

                # 根据题型显示不同的答题方式
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
                        st.session_state.user_answers[i] = selected[0]  # 存储选项字母

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

            # 重新开始按钮
            if st.button("🔄 再做一组", use_container_width=True):
                st.session_state.exercises = None
                st.session_state.user_answers = {}
                st.session_state.exercise_submitted = False
                st.rerun()


def main():
    """主函数"""
    init_session_state()

    if st.session_state.auth_role is None:
        render_login()
        st.stop()

    # 标题
    current_subject = st.session_state.current_subject
    subject_info = SUBJECTS[current_subject]

    st.markdown(f'<h1 class="main-header">📚 小学智能教育问答系统</h1>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: center; color: #757575; margin-bottom: 2rem;">
        当前学科: {subject_info['icon']} {subject_info['name']} - {subject_info['description']}
    </div>
    """, unsafe_allow_html=True)

    staff = is_staff(st.session_state.auth_role)
    use_rag, show_sources = render_sidebar(staff)

    # 主内容区 - 标签页（学生仅可使用学科问答与出题）
    tab1, tab2 = st.tabs(["💬 学科问答", "📝 出题"])

    with tab1:
        st.markdown("### 💬 学科问答")

        # 显示聊天历史
        render_chat_history(current_subject)

        # 输入框
        user_question = st.chat_input("输入你的问题...")

        if user_question:
            engine = get_rag_engine(current_subject)

            # 显示用户消息
            render_chat_message("user", user_question)

            # 获取回答
            with st.spinner("正在思考..."):
                result = engine.query(
                    user_question,
                    use_rag=use_rag,
                    show_sources=show_sources
                )

            # 显示助手消息
            render_chat_message("assistant", result["answer"], result["sources"])

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

    with tab2:
        # 练习题页面
        render_exercise_page(use_rag)

    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9E9E9E; font-size: 0.9rem;">
        💡 提示：上传教材后，系统将基于教材内容回答问题和生成练习题，提高准确性
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
