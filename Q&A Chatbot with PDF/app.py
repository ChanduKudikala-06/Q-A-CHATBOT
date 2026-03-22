#RAG Q&A Conversation with PDF Including Chat History
import os
from urllib import response
from langchain_core import retrievers
from langchain_core.runnables import Runnable
import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG With PDF")
st.write("Upload any PDF and ask questions related to that PDF")

groq_api_key=os.getenv("GROQ_API_KEY")

api_key=st.text_input("Enter your Groq API Key:",type="password")

if api_key:
    llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant")

    session_id=st.text_input("Session ID",value="Session1")


    if 'store' not in st.session_state:
        #Initalizing store with empty dict if its not in session_state
        st.session_state.store={}  

    uploaded_files=st.file_uploader("Choose any PDF file",type="PDF",accept_multiple_files=True)

    #If uploaded file is not blank
    if uploaded_files:
        documents=[]
        for upload_file in uploaded_files:
            temppdf=f"./temp.pdf"

            with open(temppdf,"wb") as file:
                file.write(upload_file.getvalue())
                file_name=upload_file.name

            
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)#adding docs list to documnets
        

        #Splitting creating embeddings for the documents

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        #It converts your vector database (vectorstore) into a retriever object that can search and fetch relevant documents.
        #This creates a search interface over your vector DB
        retriever=vectorstore.as_retriever()

      #This is used to rewrite the user’s question using chat history so that the retriever can understand it better.
        context_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history"
            "With out chat history if needed otherwise return as it is."
        )

        #system → instruction to LLM
        #chat_history → past conversation
        #input → current user question

        context_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system", context_q_system_prompt),
                #MessagesPlaceholder is used to insert a list of messages (like chat history) dynamically into a prompt template.
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        
        #creates a smart retriever that understands conversation context before searching.
        history_aware_retriever=create_history_aware_retriever(llm,retriever,context_q_prompt)


        #Answer question

        system_prompt=(

            "You are an assistant for question-answering tasks"
            "Use the following pieces of retrieved context to answer"
            "The question.if you don't know the answer,say that you"
            "dont know.Use three sentences maximum and keep the answer concise"
            "\n\n"
            "{context}"

        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        
       # RunnableWithMessageHistory → adds chat_history
       # history_aware_retriever → uses chat_history + input        ↓
       #rag_chain → uses retriever output + input

       

        #context_q_prompt → for rewriting->for searching better results
        #qa_prompt → for answering
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
       #RunnableWithMessageHistory is used to add memory (chat history) to your LangChain pipeline.
       #It passes rag pipeline,ChatHistory and input
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"#output will store in answer
        )

        user_input=st.text_input("your question:")

        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )

            st.write("Session Store:",st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter the API Key")
        




