from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec,Pinecone as pt
import os
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = pt(
    api_key=PINECONE_API_KEY,
)


index_name="medical-bot"

index = pc.Index(index_name)
docsearch = PineconeVectorStore(embedding = embeddings,index=index)
docsearch.add_documents(documents=text_chunks, ids=uuids)
#Creating Embeddings for Each of The Text Chunks & storing