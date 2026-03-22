import os
import tempfile
import streamlit as st
from rag_engine import build_index, load_index, build_rag_chain, ask
from langchain_core.messages import HumanMessage, AIMessage

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

st.set_page_config(page_title="PDF Chat", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions about its content.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload document")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded and st.button("Process PDF", type="primary"):
        with st.spinner("Reading and indexing your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                tmp_path = f.name
            n_chunks = build_index(tmp_path)
            os.unlink(tmp_path)
        st.success(f"Done! Indexed {n_chunks} chunks.")
        st.session_state["indexed"] = True
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about your PDF..."):
    if not st.session_state.indexed:
        st.warning("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                vectorstore = load_index()
                chain = build_rag_chain(vectorstore)
                answer = ask(chain, question)
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))
