from QApdf import data_ingestion

import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

def load_embedding(llm,documents):
    print("inside load_embedding")
    pc = Pinecone(
            #api_key=os.environ.get("PINECONE_API_KEY")
            api_key="pcsk_6eAkFm_9Qm79pHBej35G8deq6Hovsns839czXgKjGr9sL9aKGQgdKCqPcLdEXLiL7PL9p2"
        )
    if 'myindex02' not in pc.list_indexes().names():
        pc.create_index(
            name='myindex02',
            dimension=768,  # Gemini embeddings output size
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        print("index exists")
        pinecone_index = pc.Index("myindex02")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        # Load documents and build index
       # documents = SimpleDirectoryReader("H:\\Github\\GenAI\\Frameworks\\LlamaIndex\\Information-Retriever-Using-llamaIndex\\data\\").load_data()

        # construct vector store and customize storage context
        storage_context = StorageContext.from_defaults(
        vector_store=vector_store
        )

    
        embed_model = GoogleGenAIEmbedding(
            model_name="text-embedding-004",
            embed_batch_size=100,
            api_key="AIzaSyCMOlGFDkaDowork0eqdr-FN9nXrQtRYWw"
        )


        index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        )

        print("begin query")
        # Query
        query_engine = index.as_query_engine(llm=llm)
        
        # response = query_engine.query("what are the few of the guidelines also referred to last date of data collection to train the system?")
        # print(response)


        return query_engine

    

