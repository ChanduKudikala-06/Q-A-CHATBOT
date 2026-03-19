import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
#os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant")
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate reponse based on the question

    <context>
    {context}
    <context>
    Question:{input}
    
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFLoader("Attention.pdf")#dataingestion
        st.session_state.docs=st.session_state.loader.load()#Document loader
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("RAG document Q&A with Groq and Lama3")
user_prompt=st.text_input("Enter your query related to attention document")


if st.button("Document Embedding"):
    #for first time when we clicj button it will load
    #second time vectors is in session state so it will not load
    create_vector_embeddings()
    st.write("Vector Database is ready")


import time
if user_prompt:#user_prompt!=""
    #Prepares a pipeline to feed documents into the model.
    #talks to llm
    document_chain=create_stuff_documents_chain(llm,prompt)

    #Allows you to search the vector database for relevant chunks
    # Covert FAISS DB into search tool
    #retriever finds relevant chunks from FIASS and context is injected automatically and prompt gets filled
    #finds relevant data
    retriever=st.session_state.vectors.as_retriever()

    #Connects search + LLM.
    #connects both
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    #final 
    #When we invoke(),the retrievel chain uses the retriver to find relvant context,then passes that context
    #along with the userinput to the llm which generates the final answer based on that context
    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time is {time.process_time()-start}")

    st.write(response['answer'])

    #With a streamlit expander

    with st.expander("Document similarity search"):
        #enumerate it will give index and content
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('--------------------------')

