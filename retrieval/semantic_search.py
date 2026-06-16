from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
    "Neural networks are computational models inspired by the human brain.",
    "Artificial intelligence is the simulation of human intelligence by machines."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

question = "What is machine learning?"

question_embedding = model.encode([question])

scores = cosine_similarity(
    question_embedding,
    doc_embeddings
)

best_match_index = scores.argmax()

print("\nQuestion:")
print(question)

print("\nBest Match:")
print(documents[best_match_index])

print("\nSimilarity Score:")
print(scores[0][best_match_index])