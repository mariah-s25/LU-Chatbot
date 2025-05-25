#1. Import Required Libraries
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import chromadb

#2. Set Up API Key for Google Gemini
os.environ["GOOGLE_API_KEY"] = "......."
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY is not set! Please set it in environment variables or Streamlit secrets.")
    st.stop()

#3. Initialize Google Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

#4. Load a Predefined RAG Prompt from LangChain Hub
try:
    prompt = hub.pull("rlm/rag-prompt") 
except Exception as e:
    st.error(f"Error loading prompt: {e}")
    st.stop()

#5. Function to Format Retrieved Documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#6. Function to Build the RAG Chain
def build_the_chain():
    # List of faculty branch URLs
    urls = [
        "https://www.ul.edu.lb/faculte/branches.aspx?facultyId=6",
        "https://www.ul.edu.lb/faculte/branches.aspx?facultyId=14",
        "https://www.ul.edu.lb/faculte/branches.aspx?facultyId=13",
        "https://www.ul.edu.lb/faculte/branches.aspx?facultyId=16",
        "https://www.ul.edu.lb/faculte/branches.aspx?facultyId=15",
        "https://www.ul.edu.lb/faculte/branches.aspx?facultyId=12",
        "https://www.ul.edu.lb/faculte/branches.aspx?facultyId=35"
    ]

    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading faculty data: {e}")
        return None
     
    #7. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    #8. Generate Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    #9. Store Embeddings
    client = chromadb.Client()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, client=client)

    #10. Create Retriever
    retriever = vectorstore.as_retriever()

    #11. Define RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

#12. Streamlit UI Setup
st.title("🏛️ LU Faculties Chatbot")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

#13. User Input Form
with st.form("my_form", clear_on_submit=True):
    user_input = st.text_area("Enter your query about LU faculties:", "")
    submitted = st.form_submit_button("Send")

    #14. Process Input
    if submitted and user_input:
        st.session_state.chat_history.append({"user": user_input})
        rag_chain = build_the_chain()

        if rag_chain:
            with st.spinner("Analyzing faculty information..."):
                try:
                    response = rag_chain.invoke(user_input)
                    st.session_state.chat_history.append({"BOT": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.error("Failed to initialize chatbot. Please try again.")

#15. Display Chat
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        if "user" in chat:
            st.markdown(f"**You:** {chat['user']}")
        if "BOT" in chat:
            st.markdown(f"**BOT:** {chat['BOT']}")
