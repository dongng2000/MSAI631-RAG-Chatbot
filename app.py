from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai

# =====================================================
# GEMINI CONFIGURATION
# =====================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not configured"
    )

client = genai.Client(
    api_key=api_key
)

# =====================================================
# KNOWLEDGE BASE
# =====================================================

documents = []
document_sources = []


def chunk_text(text):

    paragraphs = text.split("\n\n")

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) < 1000:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


data_folder = "./data"

if not os.path.exists(data_folder):
    raise ValueError(f"Data folder not found: {data_folder}")

print("Loading course materials...")

for filename in os.listdir(data_folder):
    skip_phrases = [
        "Course Website",
        "Instructor Contact",
        "ISBN",
        "Grading Scale",
        "Course Format",
        "Instructor",
        "Required Book",
        "Required Book ISBN"
    ]
    if filename.lower().endswith(".txt"):

        filepath = os.path.join(data_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            if not any(
                phrase.lower() in chunk.lower()
                for phrase in skip_phrases
            ):

                documents.append(
                    f"Source: {filename}\n\n{chunk}"
                )

                document_sources.append(filename)

print(
    f"Loaded {len(documents)} chunks from "
    f"{len(set(document_sources))} files"
)

print("\nLoaded files:")

for file in sorted(set(document_sources)):
    print(f" - {file}")


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

doc_embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True
)

print("Embedding model loaded.")

# =====================================================
# CHATBOT FUNCTION
# =====================================================

def chatbot_response(message, history):

    if not message.strip():
        return "Please enter a question."

    try:

        # Create embedding for question
        question_embedding = embedding_model.encode([message])

        similarities = cosine_similarity(
            question_embedding,
            doc_embeddings
        )[0]

        top_indices = similarities.argsort()[-3:][::-1]

        best_score = similarities[top_indices[0]]

# ==========================================
# RELEVANCE CHECK
# ==========================================

        THRESHOLD = 0.35
        if best_score < THRESHOLD:
            suggested_questions = "\n".join(
                [f"• {q}" for q in examples[:5]]
            )

            return (
                "I could not find information about that topic "
                "in the uploaded course materials.\n\n"
                "Try asking one of these questions:\n\n"
                f"{suggested_questions}"
            )

        # =================================================
        # BUILD CONTEXT
        # =================================================

        context = "\n\n".join(
            [documents[i] for i in top_indices]
        )

        sources = list(
            set(document_sources[i] for i in top_indices)
        )

        # =================================================
        # PROMPT
        # =================================================

        prompt = f"""
        You are an Academic Assistant chatbot.

        Use the provided course materials to answer the question.

        Rules:
        - Provide clear educational explanations.
        - Summarize when appropriate.
        - Compare concepts when requested.
        - Explain relationships between concepts.
        - Use only the provided materials.
        - If the answer is not contained in the materials, say:
        "I could not find that information in the course materials."

        Course Materials:
        {context}

        Question:
        {message}

        Answer:
        """        
        # =================================================
        # Gemini-2.5-flash
        # =================================================

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text

        return f"""
            {answer}

            📚 Sources:
            {", ".join(sources)}
            """
    except Exception as e:

        return f"""
        Sorry, an error occurred while generating a response.

        Details:
        {str(e)}
        """
# =====================================================
# DYNAMIC EXAMPLES FROM CONTENT
# =====================================================

examples = []

seen = set()

for doc in documents:

    lines = doc.split("\n")

    for line in lines:

        line = line.strip()

        if len(line) < 20:
            continue

        # AI
        if "artificial intelligence" in line.lower():

            question = "What is Artificial Intelligence?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

        # ML
        elif "machine learning" in line.lower():

            question = "What is Machine Learning?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

        # Neural Networks
        elif "neural network" in line.lower():

            question = "What are Neural Networks?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

        # HCI
        elif "human-computer interaction" in line.lower():

            question = "What is Human-Computer Interaction (HCI)?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

        # UCD
        elif "user-centered design" in line.lower():

            question = "What is User-Centered Design?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

        # RAG
        elif "retrieval-augmented generation" in line.lower():

            question = "What is Retrieval-Augmented Generation (RAG)?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

        # Semantic Search
        elif "semantic search" in line.lower():

            question = "How does Semantic Search work?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

        # Chatbots
        elif "chatbot" in line.lower():

            question = "What are AI Chatbots?"

            if question not in seen:
                examples.append(question)
                seen.add(question)

examples = examples[:8]

if not examples:
    examples = [
        "Summarize the uploaded course materials",
        "What are the main concepts discussed?",
        "Create a study guide",
        "What topics are covered?"
    ]
# =====================================================
# GRADIO UI
# =====================================================

demo = gr.ChatInterface(
    fn=chatbot_response,
    title="🎓 Academic Assistant Chatbot (Gemini-2.5-flash)",
    description="""
Ask questions about the uploaded course materials.

Features:
• Semantic Search
• Retrieval-Augmented Generation (RAG)
• Google Gemini 2.5 Flash
• Multi-document support
""",
    examples=examples
)

# =====================================================
# LAUNCH
# =====================================================

if __name__ == "__main__":
    demo.launch()

