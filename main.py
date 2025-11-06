
hi
qjvga3eMWOyvFUa3CvIIqnsKZg6snWucbyG05FPKsYlszPk7TUu-mbXdjzhFZKsM3t2rGJLK-QT3BlbkFJ46YRPcZ39EQPEwjJad1TwopKg6aDObanx7tJY604Gi5dFZghqjX70CiDfiVeRpCD1Y-4VYEuwA


"""
PDF Q&A Web Application using Streamlit
Requirements:
pip install streamlit openai langchain langchain-openai pypdf faiss-cpu
"""

import streamlit as st
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_pdf(uploaded_file, api_key):
    """Process uploaded PDF and create vector store"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Create LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4",
            temperature=0
        )
        
        # Create prompt
        prompt_template = """Use the following pieces of context to answer the question. 
        If you don't know the answer, say so - don't make up information.

        Context: {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return vector_store, qa_chain, len(documents), len(texts)
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def main():
    st.title("📄 PDF Question & Answer System")
    st.markdown("Upload a PDF and ask questions about its content using AI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload a PDF file to analyze"
        )
        
        if uploaded_file and api_key:
            if st.button("🔄 Process PDF", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    try:
                        vector_store, qa_chain, num_pages, num_chunks = process_pdf(
                            uploaded_file, api_key
                        )
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = qa_chain
                        st.session_state.chat_history = []
                        
                        st.success(f"✅ PDF processed successfully!")
                        st.info(f"📄 Pages: {num_pages}\n\n📦 Chunks: {num_chunks}")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
        
        if st.session_state.qa_chain:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 💡 How to use:
        1. Enter your OpenAI API key
        2. Upload a PDF document
        3. Click "Process PDF"
        4. Ask questions about the document
        """)
    
    # Main content area
    if not api_key:
        st.info("👈 Please enter your OpenAI API key in the sidebar to get started")
        return
    
    if not uploaded_file:
        st.info("👈 Please upload a PDF document in the sidebar")
        return
    
    if not st.session_state.qa_chain:
        st.info("👈 Click 'Process PDF' in the sidebar to analyze your document")
        return
    
    # Chat interface
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("sources"):
                with st.expander("📚 View Sources"):
                    for j, source in enumerate(chat["sources"][:2]):
                        st.markdown(f"**Source {j+1}:**")
                        st.text(source[:300] + "...")
    
    # Question input
    question = st.chat_input("Ask a question about your document...")
    
    if question:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": question})
                    answer = result["result"]
                    sources = [doc.page_content for doc in result["source_documents"]]
                    
                    st.write(answer)
                    
                    # Show sources
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources[:2]):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(source[:300] + "...")
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
