from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
import openai
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import QAGenerationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
import os
import glob

_ = load_dotenv(find_dotenv())
openai.api_key ="sk-zCEsrfgT7iOrpcPYUOBdLBiicuy0C_tGEvsmara84yT3BlbkFJeoXrghKGn5btGFPvQwtyOoY94-Ta1YoLFbCdin_wYA"
os.environ["OPENAI_API_KEY"] ="sk-zCEsrfgT7iOrpcPYUOBdLBiicuy0C_tGEvsmara84yT3BlbkFJeoXrghKGn5btGFPvQwtyOoY94-Ta1YoLFbCdin_wYA"
os.environ['HUGGINGFACEHUB_API_TOKEN'] ="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"# Charger les documents

path= "./cuisine1.pdf"
loader = PyPDFLoader(path)
pages = loader.load()
print(f'This document have {len(pages)} pages')
#print(pages[0].page_content)
print(pages[0].metadata)

r_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 10)
docs = r_splitter.split_documents(pages)
print(len(docs))


# Générer les embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#doc_embeddings = [embeddings.embed_query(doc) for doc in documents]
#print(doc_embeddings)

#embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(docs, embeddings)
#relevant information
query = "Qu'est ce la cuisine "
relevant_docs = vectordb.similarity_search(query=query, k=3)
print(relevant_docs)

