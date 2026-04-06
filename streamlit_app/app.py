"""
Streamlit frontend for the Enterprise Agentic RAG Platform.
"""

import os
import uuid
import requests
import streamlit as st
from requests.exceptions import RequestException

# --- Page Configuration & Setup ---
st.set_page_config(
    page_title="Enterprise RAG Platform",
    page_icon="🤖",
    layout="wide",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Session State Management ---
# Persist session variables across user interactions.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

short_session_id = st.session_state.session_id[:8]

# --- Helper Functions ---
def render_routing_badge(routing_decision: str):
    """Render a styled badge based on the routing decision."""
    decision_lower = routing_decision.lower()
    if decision_lower == "rag":
        st.markdown(":green[**RAG**]")
    elif decision_lower == "direct":
        st.markdown(":blue[**Direct**]")
    elif decision_lower == "out_of_scope":
        st.markdown(":gray[**Out of Scope**]")
    else:
        st.markdown(f"**{routing_decision}**")

def get_content_icon(content_type: str) -> str:
    """Return an icon based on content type."""
    ctype = (content_type or "text").lower()
    if ctype == "table":
        return "📊"
    elif ctype == "image":
        return "🖼️"
    return "📄"

# --- Sidebar Components ---
with st.sidebar:
    st.header("Session Info")
    st.write(f"**Session ID:** `{short_session_id}`")
    
    st.divider()
    
    st.header("Settings")
    show_eval_scores = st.toggle("Show Evaluation Scores", value=False)
    
    if st.button("Clear Conversation", type="primary"):
        try:
            # Call backend to clear server-side history
            response = requests.delete(f"{BACKEND_URL}/api/v1/chat/history/{st.session_state.session_id}")
            response.raise_for_status()
            # Clear local chat UI messages
            st.session_state.messages = []
            st.success("Conversation cleared.")
        except RequestException as e:
            st.error(f"Failed to clear conversation. Backend might be unreachable: {e}")
            
    st.divider()
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file and st.button("Upload Document"):
        with st.spinner("Uploading and processing..."):
            try:
                # Backend expects form-data with the file under the 'file' key
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                upload_response = requests.post(f"{BACKEND_URL}/api/v1/documents/upload", files=files)
                upload_response.raise_for_status()
                
                result = upload_response.json()
                
                # Show success message and chunk breakdowns
                st.success(f"Successfully processed {uploaded_file.name}")
                st.write(f"📄 Text Chunks: {result.get('text_chunks', 0)}")
                st.write(f"📊 Table Chunks: {result.get('table_chunks', 0)}")
                st.write(f"🖼️ Image Chunks: {result.get('image_chunks', 0)}")
                
                # Add to local state list
                if uploaded_file.name not in st.session_state.uploaded_documents:
                    st.session_state.uploaded_documents.append(uploaded_file.name)
            
            except requests.exceptions.HTTPError as he:
                st.error(f"Upload failed: {he.response.text}")
            except RequestException:
                st.error("Backend is unreachable. Please verify BACKEND_URL is correct and the server is running.")
                
    if st.session_state.uploaded_documents:
        st.subheader("Uploaded Documents")
        for doc in st.session_state.uploaded_documents:
            st.caption(f"- {doc}")

# --- Main App Interface ---
st.title("🤖 Enterprise RAG Platform")

# Render previous messages from the conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Re-render assistant metadata properties (badges, metrics, sources)
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            
            # Show routing decision
            if routing := meta.get("routing_decision"):
                render_routing_badge(routing)

            # Show cache hit status
            if meta.get("cache_hit"):
                st.caption("⚡ Cache hit")

            # Show RAGAS evaluation scores if toggled enabled
            eval_scores = meta.get("evaluation_scores", {})
            if show_eval_scores and eval_scores:
                st.caption("Evaluation Scores")
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness", round(eval_scores.get("faithfulness", 0), 2))
                col2.metric("Answer Relevancy", round(eval_scores.get("answer_relevancy", 0), 2))
                col3.metric("Context Precision", round(eval_scores.get("context_precision", 0), 2))

            # Show context sources breakdown
            sources = meta.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for idx, src in enumerate(sources):
                        icon = get_content_icon(src.get("content_type", "text"))
                        page = src.get("page", src.get("page_number", "?"))
                        source_name = src.get("source", src.get("source_filename", "Unknown file"))
                        st.write(f"**{idx + 1}.** {icon} {source_name} (Page {page})")

# User chat input area
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Spinner while waiting for backend answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/api/v1/chat",
                    json={"question": prompt, "session_id": st.session_state.session_id},
                    params={"evaluate": show_eval_scores}
                )
                res.raise_for_status()
                data = res.json()
                
                answer = data.get("answer", "")
                st.write(answer)
                
                # Routing badge
                routing = data.get("routing_decision", "")
                if routing:
                    render_routing_badge(routing)
                
                # Cache indicator
                cache_hit = data.get("cache_hit", False)
                if cache_hit:
                    st.caption("⚡ Cache hit")
                
                # Render incoming RAGAS scores metrics conditionally
                eval_scores = data.get("evaluation_scores", {})
                if show_eval_scores and eval_scores:
                    st.caption("Evaluation Scores")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Faithfulness", round(eval_scores.get("faithfulness", 0), 2))
                    col2.metric("Answer Relevancy", round(eval_scores.get("answer_relevancy", 0), 2))
                    col3.metric("Context Precision", round(eval_scores.get("context_precision", 0), 2))
                        
                # Sources Dropdown
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for idx, src in enumerate(sources):
                            icon = get_content_icon(src.get("content_type", "text"))
                            page = src.get("page", src.get("page_number", "?"))
                            source_name = src.get("source", src.get("source_filename", "Unknown file"))
                            st.write(f"**{idx + 1}.** {icon} {source_name} (Page {page})")
                            
                # Save assistant output metadata so they persist correctly upon reruns
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "routing_decision": routing,
                        "cache_hit": cache_hit,
                        "evaluation_scores": eval_scores,
                        "sources": sources
                    }
                })
                
            except requests.exceptions.HTTPError as he:
                error_msg = f"Backend error: {he.response.text}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except RequestException:
                error_msg = "Backend is unreachable. Please verify BACKEND_URL is correct and the server is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
