from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Initialize the LLM and embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

# Sample documents for demonstration (replace with your own document loading)
sample_documents = [
    """Machine Learning is a subset of artificial intelligence that enables computers 
    to learn and make decisions from data without being explicitly programmed. 
    It involves algorithms that can identify patterns, make predictions, and improve 
    their performance over time through experience.""",
    
    """The main types of machine learning are: 
    1. Supervised Learning - uses labeled data to train models
    2. Unsupervised Learning - finds patterns in unlabeled data  
    3. Reinforcement Learning - learns through interaction with environment
    4. Semi-supervised Learning - combines labeled and unlabeled data""",
    
    """Deep Learning is a subset of machine learning that uses artificial neural networks 
    with multiple layers to model and understand complex patterns in data. It's particularly 
    effective for tasks like image recognition, natural language processing, and speech recognition.""",
    
    """Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
    interpret, and generate human language. It combines computational linguistics with 
    machine learning and deep learning models."""
]

# Create documents and split them
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Convert sample documents to Document objects
from langchain.schema import Document
docs = [Document(page_content=text) for text in sample_documents]
splits = text_splitter.split_documents(docs)

# Create vector store and retriever
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create the contextualization prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

print("History-aware retriever created!")

# Create the QA system prompt
qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the complete conversational RAG chain
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)

print("Conversational RAG chain created!")

# Initialize chat history
chat_history = []

def ask_question(question, chat_history):
    """Ask a question and update chat history"""
    result = conversational_rag_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    
    # Update chat history
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result['answer'])
    ])
    
    return result, chat_history

# Example conversation
print("\n" + "="*60)
print("CONVERSATIONAL RAG DEMO")
print("="*60)

# First question
print("\n1. First Question:")
result1, chat_history = ask_question("What is machine learning?", chat_history)
print(f"Q: What is machine learning?")
print(f"A: {result1['answer']}")
print(f"Sources used: {len(result1['context'])} documents")

# Follow-up question that references the previous context
print("\n2. Follow-up Question:")
result2, chat_history = ask_question("What are its main types?", chat_history)
print(f"Q: What are its main types?")
print(f"A: {result2['answer']}")
print(f"Sources used: {len(result2['context'])} documents")

# Another follow-up question
print("\n3. Another Follow-up:")
result3, chat_history = ask_question("Tell me more about the first type", chat_history)
print(f"Q: Tell me more about the first type")
print(f"A: {result3['answer']}")
print(f"Sources used: {len(result3['context'])} documents")

# New topic question
print("\n4. New Topic:")
result4, chat_history = ask_question("What is deep learning?", chat_history)
print(f"Q: What is deep learning?")
print(f"A: {result4['answer']}")
print(f"Sources used: {len(result4['context'])} documents")

print("\n" + "="*60)
print("CHAT HISTORY SUMMARY")
print("="*60)
for i, message in enumerate(chat_history):
    role = "Human" if isinstance(message, HumanMessage) else "AI"
    print(f"{i+1}. {role}: {message.content[:100]}{'...' if len(message.content) > 100 else ''}")

# Function to continue the conversation
def continue_conversation():
    """Interactive function to continue the conversation"""
    global chat_history
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE - Type 'quit' to exit")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        if question:
            result, chat_history = ask_question(question, chat_history)
            print(f"Assistant: {result['answer']}")
            print(f"(Used {len(result['context'])} source documents)")

# Uncomment the line below to start interactive mode
# continue_conversation()

# Advanced: Show how the contextualization works
print("\n" + "="*60)
print("CONTEXTUALIZATION EXAMPLE")
print("="*60)

# Example of how the system contextualizes follow-up questions
contextualize_chain = contextualize_q_prompt | llm

sample_history = [
    HumanMessage(content="What is machine learning?"),
    AIMessage(content="Machine learning is a subset of AI that enables computers to learn from data.")
]

contextualized = contextualize_chain.invoke({
    "chat_history": sample_history,
    "input": "What are its main types?"
})

print("Original question: 'What are its main types?'")
print(f"Contextualized question: '{contextualized.content}'")
print("\nThis shows how the system converts ambiguous follow-up questions into standalone questions!")