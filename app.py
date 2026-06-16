import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =====================================================
# KNOWLEDGE BASE
# =====================================================

documents = [
    "Human-Computer Interaction (HCI) studies how people interact with computer systems and how technology can be designed to be useful, usable, and accessible.",
    "User-centered design is a design approach that focuses on user needs, goals, and feedback throughout the development process.",
    "Usability heuristics are general interface design principles developed by Jakob Nielsen to improve usability and user experience.",
    "Conversational agents are software systems that communicate with users through natural language using text or speech.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models to improve the accuracy and relevance of generated responses.",
    "Semantic search uses embeddings and similarity matching to retrieve information based on meaning rather than exact keyword matches.",
    "Artificial intelligence is the simulation of human intelligence by machines that can learn, reason, and make decisions.",
    "AI chatbots can improve user support, automate responses, provide information, and enhance user engagement."
]

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

doc_embeddings = embedding_model.encode(documents)

# =====================================================
# LOAD LLM
# =====================================================

print("Loading TinyLlama...")

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# =====================================================
# CHAT FUNCTION
# =====================================================

def chatbot_response(message, history):

    if not message.strip():
        return "Please enter a question."

    # Semantic Search
    question_embedding = embedding_model.encode([message])

    similarities = cosine_similarity(
        question_embedding,
        doc_embeddings
    )[0]

    best_index = np.argmax(similarities)
    best_score = similarities[best_index]

    THRESHOLD = 0.50

    # =================================================
    # FALLBACK
    # =================================================

    if best_score < THRESHOLD:

        return """
I couldn't find information about that topic.

Try asking:
• What is Human-Computer Interaction (HCI)?
• What is user-centered design?
• What are usability heuristics?
• What is a conversational agent?
• What is Retrieval-Augmented Generation (RAG)?
• How does semantic search work?
• What is artificial intelligence?
• What are the benefits of AI chatbots?
"""

    # =================================================
    # RETRIEVE CONTEXT
    # =================================================

    retrieved_text = documents[best_index]

    prompt = f"""
You are an Academic Assistant Chatbot.

Use ONLY the provided context to answer.

Keep your response concise, clear, and suitable for a college student.

Context:
{retrieved_text}

Question:
{message}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=75,
        do_sample=False
    )

    generated_text = response[0]["generated_text"]

    answer = generated_text.split("Answer:")[-1]

    if "Context:" in answer:
        answer = answer.split("Context:")[0]

    if "Question:" in answer:
        answer = answer.split("Question:")[0]

    return answer.strip()

# =====================================================
# EXAMPLE QUESTIONS
# =====================================================

examples = [
    "What is Human-Computer Interaction (HCI)?",
    "What is user-centered design?",
    "What are usability heuristics?",
    "What is a conversational agent?",
    "What is Retrieval-Augmented Generation (RAG)?",
    "How does semantic search work?",
    "What is artificial intelligence?",
    "What are the benefits of AI chatbots?"
]

# =====================================================
# GRADIO UI
# =====================================================

demo = gr.ChatInterface(
    fn=chatbot_response,

    title="🎓 Academic Assistant Chatbot",

    description="""
Ask questions about: Human-Computer Interaction (HCI), User-Centered Design, Usability Heuristics, Conversational Agents, Artificial Intelligence, Retrieval-Augmented Generation (RAG)
This chatbot demonstrates a Retrieval-Augmented Generation (RAG) architecture
using semantic search and a large language model.
""",

    examples=examples
)

# =====================================================
# LAUNCH
# =====================================================

demo.launch()