# pip install streamlit==1.24.0
# pip install panda(if not available)
# pip install json
# pip install ocr-nanonets-wrapper
# pip install langchain
# pip install langchain_community
# pip install ctransformers
# pip install sentence-transformers
# pip install faiss-cpu
# Before running this code create a folder models and copy the llm model from model folder in Assignment in your current VS code directory 
# To run the code type: streamlit run {full_file_path}
import streamlit as st
import sys
import os
import pandas as pd
import json
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from nanonets import NANONETSOCR
from streamlit_chat import message

st.set_page_config(layout = 'wide') # Setting the configuration of Streamlit
st.title ("Chat with your Handwritten/Text PDFs") 
pdf_input = st.file_uploader("Upload pdf file",type=['pdf']) #Uploading File via Streamlit

DB_FAISS_PATH = "./vector_db" # Creating a local vector database 
os.makedirs(DB_FAISS_PATH, exist_ok=True)
upload_dir = "./uploaded_files"
model = NANONETSOCR() #Creating a nanonetOCR model
model.set_token('b81658dc-bf67-11ee-b732-16925d0dbdfd') #Create your own token or use the same
os.makedirs(upload_dir, exist_ok=True)
if pdf_input is not None:
    file_path = os.path.join(upload_dir, pdf_input.name)
    with open(file_path, "wb") as f:
        f.write(pdf_input.read())  #saving the uploaded file into a local file
    try:
       model.convert_to_csv(file_path, output_file_name = 'Output3.csv') # Converting the handwritten pdf to csv using ocr
       print("Status : Converted pdf")
    except:
        pass
        print("Status : PDF converted")
    data=pd.read_csv("Output3.csv").to_json(orient='values')
    list_of_dicts = json.loads(data)
    docs = []
    for row in list_of_dicts:  # Creating documents from Data
        row_without_none = [item if item is not None else '' for item in row]
        content = ",".join(row_without_none)
        doc = Document(page_content=content)
        docs.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20) #Splitting texts to chunks
    text_chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = 'NeuML/pubmedbert-base-embeddings') #generating embeddings

    # COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
    docsearch = FAISS.from_documents(docs, embeddings) # performing similarity search using FAISS (Facebook AI Similarity Search)
    docsearch.save_local(DB_FAISS_PATH)
    bm25_retriever = BM25Retriever.from_documents(docs) #Implementing bm25 retriever to ensemble with docsearch
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, docsearch.as_retriever(search_kwargs={"k": 2})], weights=[0.6, 0.4])
    

    llm = CTransformers(model="ggml-model-Q4_K_M.gguf", #Large Language Model : LLama 2.0 taken from Hugging face
                    model_type="llama",
                    max_new_tokens= 1024,
                    temperature=0)

    qa = ConversationalRetrievalChain.from_llm(llm,ensemble_retriever) # Variable for retreival chain 
    def conversation(query): #Function for retreiving response for queries
        chat_history = []
        result = qa.invoke({"question":query, 'chat_history': chat_history})
        return result["answer"]

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + pdf_input.name]
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    
    response_container = st.container() # Response container
    container = st.container() 

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            print("Status : Taking query")
            query = st.text_input("Query:", placeholder="Talk to your pdf here ", key='input') #Taking query from the user
            submit = st.form_submit_button(label='Send') # Submit button
            
        if submit and query:
            output = conversation(query)
            # Appending the current query and response to the past list
            st.session_state['past'].append(query) 
            st.session_state['generated'].append(output)
        
        if st.session_state['generated']:
         with response_container:
            print("Status : Loading result")
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user') #Printing current chat with all previous chats
                message(st.session_state["generated"][i], key=str(i))
