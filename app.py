import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. App Configuration
st.set_page_config(page_title="Knowledge Support Agent", page_icon="🤖")
st.title("🤖 AI Support Agent")

# 2. Sidebar for Setup
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google API Key", type="password", help="Get this from aistudio.google.com")
    website_url = st.text_input("Enter Knowledge Base URL", placeholder="https://example.com/support")
    
    if st.button("Load Knowledge Base"):
        if not api_key or not website_url:
            st.error("Please provide both an API Key and a URL.")
        else:
            with st.spinner("Reading website..."):
                try:
                    # Load Data
                    loader = WebBaseLoader(website_url)
                    docs = loader.load()
                    
                    # Split Data into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
                    
                    # Create Vector Store (Memory)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                    
                    # Save to session state
                    st.session_state.vectorstore = vectorstore
                    st.success("Knowledge Base Loaded!")
                except Exception as e:
                    st.error(f"Error loading URL: {e}")

# 3. Chat Interface
if "vectorstore" in st.session_state:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    # Create Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based ONLY on the following context. 
    If the answer is not in the context, say "I don't have that information in my knowledge base."
    
    Context:
    {context}
    
    Question:
    {input}
    """)
    
    # Create Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Chat Input
    user_query = st.chat_input("Ask a question about the documentation...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_query})
            
        with st.chat_message("assistant"):
            st.write(response['answer'])

else:
    st.info("👈 Please enter your Google API Key and a URL in the sidebar to start.")
