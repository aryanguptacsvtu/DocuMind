import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import backend functions
from backend import (
    get_pdf_text,
    get_text_chunks,
    build_vector_store,
    build_conversation_chain,
)


def main():
    load_dotenv()
    st.set_page_config(page_title="DocuMind", page_icon="📚")
    st.header("Chat with DocuMind 📚🧠")

    # --- Session State Initialization ---
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # HumanMessage/AIMessage objects

    # --- Display existing chat messages ---
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! 👋 I'm **DocuMind**. Upload your documents in the "
                "sidebar, and I'll become your personal expert on them."
            )

    for message in st.session_state.chat_history:
        with st.chat_message(message.type):
            st.markdown(message.content)
        
            sources = getattr(message, "additional_kwargs", {}).get("sources")
           

    # --- Sidebar for PDF processing ---
    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader(
            "Choose PDF files and click on 'Process'",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)

                    if not text_chunks:
                        st.error(
                            "Could not extract any text from the PDFs. "
                            "If these are scanned/image-only PDFs, they "
                            "need OCR before they can be processed."
                        )
                    else:
                        vector_store = build_vector_store(text_chunks)

                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.conversation = build_conversation_chain(
                                vector_store
                            )

                            st.success("Documents processed!")

                            st.session_state.chat_history = [
                                AIMessage(
                                    content=(
                                        "Your documents are processed! I've "
                                        "analyzed the content and am ready "
                                        "to answer your questions."
                                    )
                                )
                            ]
                            st.rerun()
                        else:
                            st.error("Failed to create vector store.")

        st.divider()

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # --- Chat Input and Response Logic ---
    if user_question := st.chat_input(
        "Ask DocuMind a question about your documents:",
        disabled=not st.session_state.conversation,
    ):
        if st.session_state.conversation:
            with st.chat_message("user"):
                st.markdown(user_question)

            history_for_chain = list(st.session_state.chat_history)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.invoke(
                        {
                            "input": user_question,
                            "chat_history": history_for_chain,
                        }
                    )

                    answer = response.get(
                        "answer", "I'm sorry, I couldn't find an answer."
                    )
                    st.markdown(answer)

                    sources = [
                        doc.page_content for doc in response.get("context", [])
                    ]

            # Now append both messages to history, in order, once.
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            ai_message = AIMessage(content=answer)
            ai_message.additional_kwargs["sources"] = sources
            st.session_state.chat_history.append(ai_message)

        else:
            st.warning("Please upload and process your documents first.")


if __name__ == "__main__":
    main()