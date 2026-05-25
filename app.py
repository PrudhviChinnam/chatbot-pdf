import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
}

h1 {
    text-align: center;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 Chat With Your PDF")

with st.sidebar:

    st.header("Settings")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    chunk_size = st.slider(
        "Chunk Size",
        200,
        2000,
        500,
        100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        0,
        500,
        50,
        10
    )

    k_value = st.slider(
        "Chunks to Retrieve",
        1,
        10,
        3
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.0
    )

    clear_chat = st.button("Clear Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if clear_chat:
    st.session_state.messages = []
    st.rerun()

if uploaded_file is not None:

    if st.session_state.vector_store is None:

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )

            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory="./chromadb"
            )

            st.session_state.vector_store = vector_store

        st.success("PDF processed successfully")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask question from PDF...")

if user_input:

    if st.session_state.vector_store is None:
        st.warning("Upload PDF first")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        chat_history = ""

        for msg in st.session_state.messages:

            chat_history += (
                f'{msg["role"]}: {msg["content"]}\n'
            )

        with st.spinner("Thinking..."):

            retriever = (
                st.session_state.vector_store
                .as_retriever(
                    search_kwargs={"k": k_value}
                )
            )

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                streaming=True
            )

            prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI assistant.

Use previous conversation history and context
to answer the question.

If answer is not present in context,
answer using your own knowledge.

Chat History:
{chat_history}

<context>
{context}
</context>

Question:
{input}
""")

            document_chain = (
                create_stuff_documents_chain(
                    llm,
                    prompt
                )
            )

            retrieval_chain = (
                create_retrieval_chain(
                    retriever,
                    document_chain
                )
            )

            response = retrieval_chain.stream({
                "input": user_input,
                "chat_history": chat_history
            })

            full_response = ""

            placeholder = st.empty()

            for chunk in response:

                if "answer" in chunk:

                    full_response += chunk["answer"]

                    placeholder.markdown(
                        full_response + "▌"
                    )

            placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })