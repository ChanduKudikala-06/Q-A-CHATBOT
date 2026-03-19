import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv
load_dotenv()

##langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot With OpenAI"


#Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
         ("system","Your are helpful assistant please answer my queries"),
         ("user","Question:{question}")
    ]  
)

#temperature means how random or creativity the response is.

#low temp - 0.0-0.3(same question same identical answer for multiple times)
def generate_answer(question,api_key,model,temperature,max_tokens):
   # openai.api_key=api_key,
    llm=ChatOpenAI(model=model,api_key=api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


#Title of the app

st.title("🦜 Q&A ChatBot with OpenAI")

#Sidebarfor settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your OpenAI API Key:",type="password")


#Dropdown to select various OpenAI models

model=st.sidebar.selectbox("Select an OpenAI Model",["gpt-4o","gpt-4-turbo","gpt-4"])


#Adjust reponse parameters
#value means default value
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


#Main interface for user input
st.write("Ask any Question")
user_input=st.text_input("You:")


if user_input!="":
    response=generate_answer(user_input,api_key,model,temperature,max_tokens)
    st.write(response)

else:
    st.warning("Please provide the query")