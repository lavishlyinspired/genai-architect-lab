
print("Hello World how are You")


from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from datasets import load_dataset
from haystack import Document
document_store = InMemoryDocumentStore()


dataset = load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', 'en', split ='train')



dataset


from datasets import Dataset

# Sample data (10 rows)
data = {
    "Question": [
        "What is AI?", "Define machine learning.", "What is deep learning?",
        "Explain reinforcement learning.", "What is NLP?",
    ],
    "Complex_CoT": [
        "AI is the broader concept -> ML is a subset -> DL is a subset of ML.",
        "ML is teaching machines from data -> supervised vs unsupervised.",
        "DL uses neural networks -> multi-layer perceptrons -> CNNs, RNNs.",
        "RL is based on agents -> environment -> rewards & actions.",
        "NLP deals with text -> tokenization -> embeddings -> LLMs.",
    ],
    "Response": [
        "AI enables machines to mimic human intelligence.",
        "ML allows computers to learn from data.",
        "Deep learning is a type of ML with neural nets.",
        "Reinforcement learning is learning via rewards.",
        "NLP is AI applied to human language.",
    ]
}

# Create dataset
dataset = Dataset.from_dict(data)

# Show summary
print(dataset)
print(dataset[0])  # print first row



# # limit dataset to top 1000 rows
# subset = dataset.select(range(1000))

# build docs list
#docs = [Document(content=doc["Question"], meta=doc["Response"]) for doc in subset]
docs = [
    Document(
        content=doc["Question"], 
        meta={
            "response": doc["Response"],
            "complex_cot": doc["Complex_CoT"]
        }
    )
    for doc in dataset
]





doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")


doc_embedder.warm_up()


docs_with_embeddings = doc_embedder.run(docs)


#docs_with_embeddings


from haystack.document_stores.in_memory import InMemoryDocumentStore


document_store = InMemoryDocumentStore()  # fresh, empty store


document_store.write_documents(docs_with_embeddings["documents"])


from haystack.components.embedders import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")


from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store)


retriever





from haystack.components.builders import PromptBuilder

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)


prompt_builder


import os
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from dotenv import load_dotenv
load_dotenv
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")





gemini = GoogleAIGeminiGenerator(model="gemini-1.5-flash")


from haystack import Pipeline

basic_rag_pipeline = Pipeline()


prompt_builder


# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", gemini)


# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")



question = "hat is the most likely diagnosis for a 2-year-old 70 kg child?"


#response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

response = basic_rag_pipeline.run(
    {"text_embedder": {"text": question},"prompt_builder": {"question": question}},
    include_outputs_from={"retriever", "prompt_builder"})


print(response.keys())
print(response)



#response


# # Example query
# query = "What is generative AI?"

# # Run the pipeline
# result = basic_rag_pipeline.run({
#     "text_embedder": {"text": query}, "prompt_builder": {"question": question}
# })

# # 🔎 Check retrieved documents
# retrieved_docs = response["retriever"]["documents"]
# # print("\n=== Retrieved Documents ===")
# # for i, doc in enumerate(retrieved_docs, 1):
# #     print(f"\nDocument {i}:")
# #     print("Content:", doc.content)
# #     print("Meta:", doc.meta)

# # 🔎 Check the built prompt
# prompt_text = response["prompt_builder"]["prompt"]
# print("\n=== Built Prompt ===")
# print(prompt_text)

# # ✅ Final LLM answer
# print("\n=== LLM Answer ===")
# print(response["llm"]["replies"][0])



response['llm']['replies'][0]


question="What is the most likely diagnosis for a 2-year-old 70 kg child?"


response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})


response["llm"]["replies"][0]


