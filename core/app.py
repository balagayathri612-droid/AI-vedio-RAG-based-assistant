"""
Streamlit UI for the AI Video/Meeting Assistant.

Wraps the existing CLI pipeline (run_pipeline) with:
- Source input (YouTube URL or local file upload)
- Language selection
- Tabbed results view (Summary, Transcript, Action Items, Decisions, Questions)
- A chat interface backed by the RAG chain

Run with:  streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# --------------------------------------------------------------------------- #
# Pipeline (same logic as the CLI script, reused as-is)
# --------------------------------------------------------------------------- #
def run_pipeline(source: str, language: str = "english", progress_cb=None) -> dict:
    def step(msg):
        if progress_cb:
            progress_cb(msg)

    step("Processing audio into chunks...")
    chunks = process_input(source)

    step("Transcribing audio...")
    transcript = transcribe_all(chunks, language)

    step("Generating title...")
    title = generate_title(transcript)

    step("Summarizing...")
    summary = summarize(transcript)

    step("Extracting action items...")
    action_items = extract_action_items(transcript)

    step("Extracting key decisions...")
    decisions = extract_key_decisions(transcript)

    step("Extracting open questions...")
    questions = extract_questions(transcript)

    step("Building chat index (RAG)...")
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# --------------------------------------------------------------------------- #
# Streamlit page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎙️",
    layout="wide",
)

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)
if "processing" not in st.session_state:
    st.session_state.processing = False
if "runs_cache" not in st.session_state:
    # Manual cache keyed by (source, language) -> result dict.
    # Avoids re-transcribing / rebuilding the RAG chain for a source you've
    # already processed this session. Cleared when you click "Clear results"
    # only removes the active view, not the cache — see "Clear cache" button.
    st.session_state.runs_cache = {}
if "run_history" not in st.session_state:
    # Ordered list of (title, source, language) for past runs this session.
    st.session_state.run_history = []


# --------------------------------------------------------------------------- #
# Sidebar — input controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎙️ AI Video Assistant")
    st.caption("Transcribe, summarize, and chat with any meeting or video.")

    st.divider()

    input_mode = st.radio("Source type", ["YouTube URL", "Upload file"], horizontal=True)

    source = None
    tmp_path_to_cleanup = None

    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video file",
            type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"],
        )
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix
            tmp_dir = Path(tempfile.gettempdir())
            tmp_path = tmp_dir / f"upload_{uploaded_file.name}"
            tmp_path.write_bytes(uploaded_file.getbuffer())
            source = str(tmp_path)
            st.success(f"Ready: {uploaded_file.name}")

    language = st.selectbox("Language", ["english", "Tanglish"], index=0)

    st.divider()

    run_clicked = st.button(
        "🚀 Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing or not source,
    )

    if st.session_state.result:
        st.divider()
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.run_history:
        st.divider()
        st.caption("Past runs this session")
        for i, (hist_title, hist_source, hist_language) in enumerate(reversed(st.session_state.run_history)):
            if st.button(f"📄 {hist_title}", key=f"hist_{i}", use_container_width=True):
                cache_key = (hist_source, hist_language)
                st.session_state.result = st.session_state.runs_cache.get(cache_key)
                st.session_state.chat_history = []
                st.rerun()
        if st.button("🧹 Clear cache", use_container_width=True):
            st.session_state.runs_cache = {}
            st.session_state.run_history = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #
if run_clicked and source:
    st.session_state.processing = True
    st.session_state.chat_history = []

    cache_key = (source, language)

    if cache_key in st.session_state.runs_cache:
        st.session_state.result = st.session_state.runs_cache[cache_key]
        st.toast("Loaded from this session's cache — no re-processing needed.", icon="⚡")
        st.session_state.processing = False
    else:
        status_box = st.empty()
        progress_bar = st.progress(0, text="Starting...")

        steps_seen = {"count": 0}
        total_steps = 7

        def progress_cb(msg):
            steps_seen["count"] += 1
            pct = min(int(steps_seen["count"] / total_steps * 100), 100)
            progress_bar.progress(pct, text=msg)

        try:
            with st.spinner("Working on it..."):
                result = run_pipeline(source, language, progress_cb=progress_cb)
            st.session_state.result = result
            st.session_state.runs_cache[cache_key] = result
            st.session_state.run_history.append((result["title"], source, language))
            progress_bar.progress(100, text="Done!")
            status_box.success("Pipeline completed successfully ✅")
        except Exception as e:
            status_box.error(f"Pipeline failed: {e}")
        finally:
            st.session_state.processing = False


# --------------------------------------------------------------------------- #
# Results view
# --------------------------------------------------------------------------- #
result = st.session_state.result

if result is None:
    st.info("👈 Add a YouTube URL or upload a file, then click **Run Pipeline** to get started.")
else:
    st.header(f"📌 {result['title']}")

    tabs = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "💬 Chat"]
    )

    with tabs[0]:
        st.markdown(result["summary"])

    with tabs[1]:
        st.text_area("Full transcript", result["transcript"], height=500)
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

    with tabs[2]:
        st.markdown(result["action_items"])

    with tabs[3]:
        st.markdown(result["key_decisions"])

    with tabs[4]:
        st.markdown(result["open_questions"])

    with tabs[5]:
        st.caption("Ask questions about the transcript — answered via the RAG chain.")

        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        question = st.chat_input("Ask something about this meeting...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"⚠️ Error answering question: {e}"
                    st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))