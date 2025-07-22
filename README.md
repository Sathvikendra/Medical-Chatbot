# Medical-Chatbot
Medical Chatbot using Llama

### Steps to run the project

Clone the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

```bash
pip install -r requirements.txt
```

Download Llama model

https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main

Create a .env file in root directory and add Pinecone credentials

```ini
PINECONE_API_KEY="xxxxxxxxxxxxxxxx"
PINECONE_API_ENV="xxxxxxx"
```

```bash
python store_index.py
```

```bash
python app.py
```

Now, 

```bash
open up localhost:
```

### Techstack Used:

- Python
- Langchain
- Flask
- Meta Llama
- Pinecone