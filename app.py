from flask import Flask,render_template,jsonify,request
from src.helper import load_pdf,text_split,download_huggingface_embedding
from langchain.vectorstores import Pinecone
from pinecone import Pinecone,ServerlessSpec
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings=download_huggingface_embedding()
vectorstore=FAISS.from_documents(text_chunks,embeddings)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="medical-chatbot"
index = pc.Index(index_name)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # or the dimension of your embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        )
    )

docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name,
    # namespace="default", 
    pinecone_api_key=PINECONE_API_KEY
)

# docsearch=Pinecone.from_existing_index(index_name,embeddings)


# docsearch = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings)

prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"prompt":prompt}

llm=CTransformers(model="model/llama-2-13b-chat.ggmlv3.q6_K.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
chain_type_kwargs = {} 
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,    
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    result=qa({"query":input})
    print("Response:",result["result"])
    return str(result["result"])


if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
