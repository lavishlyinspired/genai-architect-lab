from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

def loader_data():
    loader = PyPDFLoader("H:\\Github\\GenAI\\RAG\\RAG using Hybrid Search\data\\2005.11401v4.pdf")
    docs = loader.load()
    return docs
    
def split_data(docs):
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return texts



def create_embeddings(texts):
    # Create a vector store
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    return embeddings

def create_vectorstore(embeddings):
    # Create a vector store
    #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
    )
    return vector_store

def create_retriever(vector_store):
    # Create a retriever
    vectorstore_retreiver = vector_store.as_retriever(search_type="similarity", search_k=2)
    return vectorstore_retreiver

def ensemble_retriever(vectorstore_retreiver, chunks):
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k =  3
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.3, 0.7])
    return ensemble_retriever


def load_llm():
    #LLM
    import os
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

  

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5, max_tokens=1000)
    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm.invoke("Sing a ballad of LangChain.")
    return llm

def normal_hybrid_chain(llm, vectorstore_retreiver,ensemble_retriever):
     
    normal_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore_retreiver)
    hybrid_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ensemble_retriever)
    return normal_chain, hybrid_chain

if __name__== "__main__":
    docs = loader_data()
    texts = split_data(docs)
    embeddings =  create_embeddings(texts)
    vector_store = create_vectorstore(embeddings)
    vectorstore_retreiver = create_retriever(vector_store)
    ensemble_retriever = ensemble_retriever(vectorstore_retreiver,texts)

        #LLM
   
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5, max_tokens=1000)
    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    llm.invoke("Sing a ballad of LangChain.")
    
   # llm = load_llm()
    normal_chain, hybrid_chain = normal_hybrid_chain(llm, vectorstore_retreiver,ensemble_retriever)
    print(normal_chain.invoke("what is Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks?"))
    print(hybrid_chain.invoke("what is Retrieval-Augmented Generation forKnowledge-Intensive NLP Tasks?"))




