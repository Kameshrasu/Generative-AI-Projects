

# importing the necessary modules
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


#----------------------------------------++++++--------------------------------------------------------

# load the langchain api key. for monitoring the model 
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] =groq_api_key

#----------------------------------------++++++--------------------------------------------------------


# embedding model
embeddings=OllamaEmbeddings()


#----------------------------------------++++++--------------------------------------------------------

# loading the content fron the website
#loader=WebBaseLoader("https://www.vsbcetc.com/")
#docs=loader.load()


# loadindg the content from the document
from langchain_community.document_loaders import TextLoader
loader=TextLoader("path to your file")
docs=loader.load()


# loading the content using the pdf 
# from langchain.document_loaders import PyPDFLoader
# pdf_file = "Circular_Whistle Blower Policy_12.02.2019-english.pdf"
# loader = PyPDFLoader(pdf_file)
# docs = loader.load()


#----------------------------------------++++++--------------------------------------------------------


# splitting the docs and load the final docs in the CHORMO VECTOR DATABASE
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=text_splitter.split_documents(docs)
vectors=Chroma.from_documents(final_documents,embeddings)


#----------------------------------------++++++--------------------------------------------------------

#llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
llm =Ollama(model="phi3")
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)



response=retrieval_chain.invoke({"input":prompt})
