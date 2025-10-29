import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Splits raw text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# This decorator tells Streamlit to cache the output of this function.
# The function will only be re-run if the input (text_chunks) changes.
@st.cache_resource
def get_vector_store(_text_chunks):
    """Creates a FAISS vector store from text chunks and embeddings."""
    if not _text_chunks:
        return None
    
    # This part is slow (downloading and running on CPU)
    embeddings = HuggingFaceEmbeddings( model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # This part is also slow (indexing)
    vectorStore = FAISS.from_texts(texts=_text_chunks, embedding=embeddings)
    return vectorStore


# We also cache the conversation chain creation
@st.cache_resource
def get_conversation_chain(_vector_store):
    """Creates the modern conversational retrieval chain (RAG) using LCEL."""
    # This function depends on the vector_store, so we pass it in
    # The underscore (_) is a convention to show it's an input to a cached func
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    retriever = _vector_store.as_retriever()

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

    history_aware_retriever = create_history_aware_retriever( llm, retriever, contextualize_q_prompt )

    # 2. Answering prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know."
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
