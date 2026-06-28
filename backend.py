import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF documents."""
    text = ""
    skipped_pages = 0

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                skipped_pages += 1

    if skipped_pages:
        st.warning(
            f"⚠️ Skipped {skipped_pages} page(s) with no extractable text "
            "(likely scanned images without OCR)."
        )

    return text


def get_text_chunks(text):
    """Splits raw text into manageable chunks. """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def build_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks and embeddings."""
    if not text_chunks:
        return None

    embeddings = get_embedding_model()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


@st.cache_resource
def get_embedding_model():
    """Loads the HuggingFace embedding model once and shares it across all sessions. """

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_conversation_chain(vector_store):
    """Creates the conversational retrieval chain (RAG) using LCEL. """
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        model_kwargs={"reasoning_effort": "low", "include_reasoning": False},
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 1. Contextualize (rephrase) question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answering prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know. Do not make up information that isn't "
        "in theAC provided context."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 3. Create the document "stuffing" chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 4. Combine into the final chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain