import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import backend functions
from backend import get_pdf_text, get_text_chunks, get_vector_store, get_conversation_chain

def main():
    load_dotenv()
    # 1. Update Page Config
    st.set_page_config(page_title="DocuMind", page_icon="📚")

    # 2. Update Header
    st.header("Chat with DocuMind 📚🧠")

    # --- Session State Initialization ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Will store HumanMessage/AIMessage objects

    # --- Display existing chat messages ---
    
    # 3. Update Default Welcome Message
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown("Hello! 👋 I'm **DocuMind**. Upload your documents in the sidebar, and I'll become your personal expert on them.")

    for message in st.session_state.chat_history:
        # message.type will be "human" or "ai"
        with st.chat_message(message.type):
            st.markdown(message.content)

    # --- Sidebar for PDF processing ---
    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader(
            "Choose PDF files and click on 'Process'", 
            type="pdf",
            accept_multiple_files=True
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")

            else:
                with st.spinner("Processing..."):
                    # Get PDF Text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get the Text Chunks
                    text_chunks = get_text_chunks(raw_text)

                    if not text_chunks:
                        st.error("Could not extract text from the PDFs.")
                    else:
                        # Create Vector Store
                        vector_store = get_vector_store(text_chunks)
                        
                        if vector_store:
                            # Create Conversation Chain and save to session state
                            st.session_state.conversation = get_conversation_chain(vector_store)
                            
                            st.success("Documents processed!")
                            
                            # 4. Update Post-Processing Message
                            # Clear chat history and add a success message
                            st.session_state.chat_history = [
                                AIMessage(content="Your documents are processed! I've analyzed the content and am ready to answer your questions.")
                            ]
                            
                            # st.rerun() # Rerun to show the new message and enable chat

                        else:
                            st.error("Failed to create vector store.")

        st.divider()
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun() # Rerun to clear the chat display

    # --- Chat Input and Response Logic ---
    # 5. Update Chat Input Prompt
    if user_question := st.chat_input("Ask DocuMind a question about your documents:", disabled=not st.session_state.conversation):
        
        # Check if conversation chain is ready
        if st.session_state.conversation:
            # Add user message to UI
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Add user message to history
            st.session_state.chat_history.append(HumanMessage(content=user_question))

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    
                    # Invoke the chain with the new input and existing history
                    response = st.session_state.conversation.invoke({
                        "input": user_question,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response.get("answer", "I'm sorry, I couldn't find an answer.")
                    st.markdown(answer)
            
            # Add AI response to history
            st.session_state.chat_history.append(AIMessage(content=answer))
            
        else:
            st.warning("Please upload and process your documents first.")


if __name__ == '__main__':
    main()
