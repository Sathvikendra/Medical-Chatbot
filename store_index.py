from src.helper import load_pdf,text_split,download_huggingface_embedding
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)
extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings=download_huggingface_embedding()
vectorstore=FAISS.from_documents(text_chunks,embeddings)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="medical-chatbot"
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name,
    # namespace="default", 
    pinecone_api_key=PINECONE_API_KEY
)

