import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

# --- Custom Streamlit-friendly callback handler ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

# --- API Key from Streamlit secrets ---
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Streamlit App UI setup ---
st.set_page_config(page_title="📄 PDF Chatbot", layout="wide")
st.title("📚 PDF Question Answering Chatbot")

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.rerun()

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- Upload PDF ---
pdf = st.file_uploader("Upload a PDF file", type="pdf")

if pdf and st.session_state.qa_chain is None:
    # --- Extract text from PDF ---
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    # --- Chunk the text ---
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    st.write(f"✅ Total chunks created: {len(chunks)}")

    # --- Embed chunks ---
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # --- Setup retriever with limited chunks to avoid token overflow ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # --- Setup memory and chain ---
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    llm = ChatOpenAI(temperature=0, streaming=True)

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# --- Chat Interface ---
if st.session_state.qa_chain:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask something about the PDF...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            stream_output = st.empty()
            handler = StreamHandler(container=stream_output)

            result = st.session_state.qa_chain(
                {"question": user_input},
                callbacks=[handler],
                return_only_outputs=True
            )

            response_text = result.get("answer", "No response.")
            sources = result.get("source_documents", [])

            stream_output.markdown(response_text)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            if sources:
                with st.expander("📄 Sources used in this answer"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.code(doc.page_content[:1000], language="text")

# --- Download Chat Transcript ---
if st.session_state.chat_history:
    transcript = ""
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        transcript += f"{role}: {msg['content']}\n\n"

    st.download_button(
        label="📥 Download Chat Transcript",
        data=transcript,
        file_name="chat_transcript.txt",
        mime="text/plain"
    )
