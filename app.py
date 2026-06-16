from dotenv import load_dotenv
load_dotenv()
import os
import gradio as gr
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# =====================================================
# AZURE OPENAI CONFIGURATION
# =====================================================

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")

if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")

if not api_key:
    raise ValueError("AZURE_OPENAI_KEY is not configured")

client = OpenAI(
    base_url=f"{endpoint}/openai/v1",
    api_key=api_key
)

deployment_name = "Phi-4-mini-instruct"

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

print("Embedding model loaded.")

# =====================================================
# CHATBOT FUNCTION
# =====================================================

def chatbot_response(message, history):

    if not message.strip():
        return "Please enter a question."

    try:

        # Semantic Search
        question_embedding = embedding_model.encode([message])

        similarities = cosine_similarity(
            question_embedding,
            doc_embeddings
        )[0]

        best_index = np.argmax(similarities)
        best_score = similarities[best_index]

        THRESHOLD = 0.50

        # =============================================
        # FALLBACK
        # =============================================

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

        # =============================================
        # RETRIEVE CONTEXT
        # =============================================

        context = documents[best_index]

        prompt = f"""
Use ONLY the provided context to answer.

Context:
{context}

Question:
{message}
"""

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an Academic Assistant Chatbot.

Answer using only the provided context.

Keep responses concise, accurate, and suitable for college students.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=200
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"""
Azure Error:

{str(e)}
"""

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
    title="🎓 Academic Assistant Chatbot (Azure Phi-4)",
    description="""
Ask questions about:

• Human-Computer Interaction (HCI)
• User-Centered Design
• Usability Heuristics
• Conversational Agents
• Artificial Intelligence
• Retrieval-Augmented Generation (RAG)

This chatbot uses:
- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Microsoft Azure Phi-4 Mini
""",
    examples=examples
)

# =====================================================
# LAUNCH
# =====================================================

if __name__ == "__main__":
    demo.launch()