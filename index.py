from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores.qdrant import Qdrant

import config


def create_index(document_dir: str) -> str:
    try:
        loader = DirectoryLoader(
            document_dir,
            glob='**/*.pdf',
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=16
        )
        texts = text_splitter.split_documents(documents)
        Qdrant.from_documents(
            texts,
            config.embedding_function,
            collection_name=config.COLLECTION_NAME,
            url=config.QDRANT_URL
        )
        return 'Documents uploaded and index created successfully. You can chat now.'
    except Exception as e:
        return e
