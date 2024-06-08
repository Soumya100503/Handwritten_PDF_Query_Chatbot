# This is a test code which does not have any interface, just use for getting quick answers for your prompt in the shell itself
# For interface run Assignment_final_Soumya
# pip install panda(if not available)
# pip install json
# pip install ocr-nanonets-wrapper
# pip install langchain
# pip install langchain_community
# pip install ctransformers
# pip install sentence-transformers
# pip install faiss-cpu
import streamlit as st
import sys
import os
import pandas as pd
import json
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

pdf_input = "C:/Users/beher/Downloads/Handwritten_notes_Andrew_NG.pdf" #Change file path to your file pathW
DB_FAISS_PATH = "vectorstore/vector_db" # Creating a local vector database 
# model = NANONETSOCR() #Creating a nanonetOCR model
# model.set_token('b81658dc-bf67-11ee-b732-16925d0dbdfd') #Create your own token or use the same
# print("Please wait for some time")
# print()
# model.convert_to_csv(pdf_input, output_file_name = 'Output3.csv')
# data=pd.read_csv("Output3.csv").to_json(orient='values')
# list_of_dicts = json.loads(data)
# docs = []
# for row in list_of_dicts:  # Creating documents from Data
#     row_without_none = [item if item is not None else '' for item in row]
#     content = ",".join(row_without_none)
#     doc = Document(page_content=content)
#     docs.append(doc)
loader = DirectoryLoader('C:/Users/beher/Downloads/Hanwritten pdf query/data', glob="**/*.pdf", show_progress=True, loader_cls= UnstructuredFileLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70) #Splitting texts to chunks
text_chunks = text_splitter.split_documents(documents)
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings") #generating embeddings

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(documents, embeddings) # performing similarity search using FAISS (Facebook AI Similarity Search)
docsearch.save_local(DB_FAISS_PATH)
bm25_retriever = BM25Retriever.from_documents(documents) #Implementing bm25 retriever to ensemble with docsearch
bm25_retriever.k = 4

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, docsearch.as_retriever(search_kwargs={"k": 2})], weights=[0.6, 0.4])


llm = CTransformers(model="ggml-model-Q4_K_M.gguf", 
                model_type="llama",
                max_new_tokens= 1024,
                temperature=0)

qa = ConversationalRetrievalChain.from_llm(llm,ensemble_retriever) # Variable for retreival chain 
while True:
    chat_history = []
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])
    print("-"*135)
    print()
