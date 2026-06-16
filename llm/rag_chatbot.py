from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==========================================
# KNOWLEDGE BASE
# ==========================================

documents = [
    "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
    "Neural networks are computational models inspired by the structure of the human brain.",
    "Artificial intelligence is the simulation of human intelligence by machines."
]

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

doc_embeddings = embedding_model.encode(documents)

# ==========================================
# LOAD LLM
# ==========================================

print("Loading TinyLlama...")

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# ==========================================
# USER QUESTION
# ==========================================

question = input("Ask a question: ")

print("\nSearching documents...")

question_embedding = embedding_model.encode([question])

similarities = cosine_similarity(
    question_embedding,
    doc_embeddings
)[0]

best_index = np.argmax(similarities)

best_score = similarities[best_index]

print(f"\nSimilarity Score: {best_score:.3f}")

# ==========================================
# FALLBACK
# ==========================================

THRESHOLD = 0.50

if best_score < THRESHOLD:

    print("\nI couldn't find information about that topic.")

    print("\nTry asking about:")
    print("- Machine Learning")
    print("- Neural Networks")
    print("- Artificial Intelligence")

    exit()

# ==========================================
# RETRIEVED CONTEXT
# ==========================================

retrieved_text = documents[best_index]

print("\nRetrieved Context:")
print(retrieved_text)

# ==========================================
# PROMPT
# ==========================================

prompt = f"""
You are an Academic Assistant Chatbot.

Answer ONLY using the information provided in the context.

Keep the answer short and easy to understand.

Context:
{retrieved_text}

Question:
{question}

Answer:
"""

print("\nGenerating answer...")

response = generator(
    prompt,
    max_new_tokens=75,
    do_sample=False
)

generated_text = response[0]["generated_text"]

# ==========================================
# CLEAN OUTPUT
# ==========================================

answer = generated_text.split("Answer:")[-1]

if "Context:" in answer:
    answer = answer.split("Context:")[0]

if "Question:" in answer:
    answer = answer.split("Question:")[0]

answer = answer.strip()

# ==========================================
# DISPLAY ANSWER
# ==========================================

print("\nGenerated Answer:\n")
print(answer)