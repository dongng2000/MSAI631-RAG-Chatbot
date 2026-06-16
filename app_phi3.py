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
# LOAD MICROSOFT PHI-3 MINI
# =====================================================

print("Loading Microsoft Phi-3 Mini...")

generator = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True
)

# =====================================================
# CHATBOT FUNCTION
# =====================================================

def chatbot_response(message, history):

    if not message.strip():
        return "Please enter a question."

    # -------------------------------------------------
    # Semantic Search
    # -------------------------------------------------

    question_embedding = embedding_model.encode([message])

    similarities = cosine_similarity(
        question_embedding,
        doc_embeddings
    )[0]

    best_index = np.argmax(similarities)

    best_score = similarities[best_index]

    THRESHOLD = 0.50

    # -------------------------------------------------
    # FALLBACK
    # -------------------------------------------------

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

    # -------------------------------------------------
    # RETRIEVED CONTEXT
    # -------------------------------------------------

    retrieved_text = documents[best_index]

    # -------------------------------------------------
    # PHI-3 PROMPT
    # -------------------------------------------------

    prompt = f"""
<|system|>
You are an Academic Assistant Chatbot.
Answer ONLY using the information provided in the context.
If the answer is not available in the context, say:
"I could not find that information in the available documents."
Keep answers concise, educational, and easy to understand.
<|end|>

<|user|>
Context:
{retrieved_text}

Question:
{message}
<|end|>

<|assistant|>
"""

    # -------------------------------------------------
    # GENERATE RESPONSE
    # -------------------------------------------------

    response = generator(
        prompt,
        max_new_tokens=100,
        temperature=0.2,
        do_sample=True
    )

    generated_text = response[0]["generated_text"]

    answer = generated_text.replace(
        prompt,
        ""
    ).strip()

    return answer

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
# CHAT INTERFACE
# =====================================================

demo = gr.ChatInterface(
    fn=chatbot_response,

    chatbot=gr.Chatbot(
        height=300,
        show_copy_button=True
    ),

    title="🎓 Academic Assistant Chatbot",

    description="""
### Topics You Can Ask About
• Human-Computer Interaction (HCI)
• User-Centered Design
• Usability Heuristics
• Conversational Agents
• Artificial Intelligence
• Retrieval-Augmented Generation (RAG)
Select a suggested question below or type your own.
""",

    examples=examples
)

# =====================================================
# LAUNCH
# =====================================================

demo.launch()