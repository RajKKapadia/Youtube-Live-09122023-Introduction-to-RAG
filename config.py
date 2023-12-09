import os

from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

'''Reading environment variables
'''
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

ERROR_MESSAGE = 'We are facing technical issue at this moment.'

'''Qdrant setting
'''
COLLECTION_NAME = 'rag_database'
QDRANT_URL = 'http://localhost:6333'
DOC_TYPE = 'pdf'

client = QdrantClient(
    host='localhost',
    port=6333
)

'''Openai setting
'''
embedding_function = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

chat_model = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4'
)
