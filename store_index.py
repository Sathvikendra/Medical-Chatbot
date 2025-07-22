from src.helper import load_pdf,text_split,download_huggingface_embedding
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings=download_huggingface_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="medical-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        )
    )

docsearch=PineconeVectorStore.from_documents(documents=text_chunks,embedding=embeddings,index_name=index_name)

print("ALL OK")