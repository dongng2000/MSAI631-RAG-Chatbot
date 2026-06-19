from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
import numpy as np
import pickle
import hashlib
import re


from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# ==========================
# Cache Functions
# ==========================

def get_data_folder_hash(data_folder):

    md5 = hashlib.md5()

    for filename in sorted(os.listdir(data_folder)):

        filepath = os.path.join(
            data_folder,
            filename
        )

        if os.path.isfile(filepath):

            md5.update(
                filename.encode()
            )

            with open(filepath, "rb") as f:

                md5.update(
                    f.read()
                )

    return md5.hexdigest()
CACHE_FILE = "embeddings.pkl"
HASH_FILE = "embeddings.hash"

def save_cache(
    documents,
    document_sources,
    doc_embeddings,
    file_hash
):

    with open(
        CACHE_FILE,
        "wb"
    ) as f:

        pickle.dump(
            {
                "documents": documents,
                "document_sources": document_sources,
                "doc_embeddings": doc_embeddings
            },
            f
        )

    with open(
        HASH_FILE,
        "w"
    ) as f:

        f.write(file_hash)


def load_cache():

    with open(
        CACHE_FILE,
        "rb"
    ) as f:

        data = pickle.load(f)

    return (
        data["documents"],
        data["document_sources"],
        data["doc_embeddings"]
    )


def cache_is_valid(
    current_hash
):

    if not os.path.exists(CACHE_FILE):
        return False

    if not os.path.exists(HASH_FILE):
        return False

    with open(
        HASH_FILE,
        "r"
    ) as f:

        saved_hash = f.read()

    return saved_hash == current_hash
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
# PDF READER
# =====================================================

def extract_pdf_text(filepath):

    text = ""

    try:

        reader = PdfReader(filepath)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:

        print(f"Error reading PDF {filepath}: {e}")

    return text

# =====================================================
# KNOWLEDGE BASE
# =====================================================

documents = []
document_sources = []

available_chapters = set()
available_weeks = set()

def chunk_text(text):

    paragraphs = text.split("\n\n")

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) < 1500:
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

print("Loading course materials...")

for filename in os.listdir(data_folder):


    chapter_match = re.search(
        r"chapter(\d+)",
        filename.lower()
    )

    if chapter_match:
        available_chapters.add(
            chapter_match.group(1)
        )

    week_match = re.search(
        r"week(\d+)",
        filename.lower()
    )

    if week_match:
        available_weeks.add(
            week_match.group(1)
        )
    filepath = os.path.join(
        data_folder,
        filename
    )

    text = ""

    # ============================
    # TXT FILES
    # ============================

    if filename.lower().endswith(".txt"):

        print(f"Loading TXT: {filename}")

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

    # ============================
    # PDF FILES
    # ============================

    elif filename.lower().endswith(".pdf"):

        print(f"Loading PDF: {filename}")

        text = extract_pdf_text(filepath)

        print(
            f"{filename}: "
            f"{len(text)} characters"
        )

    else:
        continue

    chunks = chunk_text(text)

    print(
        f"{filename}: "
        f"{len(chunks)} chunks"
    )

    for chunk in chunks:

        if not any(
            phrase.lower() in chunk.lower()
            for phrase in skip_phrases
        ):

            documents.append(
                f"Source: {filename}\n\n{chunk}"
            )

            document_sources.append(
                filename
            )

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

file_hash = get_data_folder_hash(
    data_folder
)

if cache_is_valid(
    file_hash
):

    print(
        "Loading cached embeddings..."
    )

    (
        documents,
        document_sources,
        doc_embeddings
    ) = load_cache()

else:

    print(
        "Generating embeddings..."
    )

    doc_embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    )

    save_cache(
        documents,
        document_sources,
        doc_embeddings,
        file_hash
    )

print(
    "Embedding model loaded."
)

# =====================================================
# CHATBOT FUNCTION
# =====================================================

def chatbot_response(message, history):

    if not message.strip():
        return "Please enter a question."

    try:

        message_lower = message.lower()

        # ==========================================
        # CREATE EMBEDDING
        # ==========================================

        question_embedding = embedding_model.encode([message])

        similarities = cosine_similarity(
            question_embedding,
            doc_embeddings
        )[0]

        # ==========================================
        # REQUEST TYPE DETECTION
        # ==========================================

        study_guide = (
            "study guide" in message_lower
        )

        summary_request = (
            "summarize" in message_lower
        )

        comparison_request = (
            "compare" in message_lower
        )

        quiz_request = (
            "quiz" in message_lower
        )

        flashcards_request = (
            "flashcard" in message_lower
        )

        # ==========================================
        # DYNAMIC CHAPTER DETECTION
        # ==========================================

        chapter_match = re.search(
            r"chapter\s*(\d+)",
            message_lower
        )

        week_match = re.search(
            r"week\s*(\d+)",
            message_lower
        )

        if chapter_match:

            chapter_num = chapter_match.group(1).zfill(2)

            top_indices = [
                i
                for i, source in enumerate(document_sources)
                if f"chapter{chapter_num}" in source.lower()
            ][:5]

            if not top_indices:
                chapter_list = "\n".join(
                    [f"• Chapter {c}" for c in sorted(available_chapters)]
                )

                return (
                    f"I could not find Chapter {chapter_num} "
                    "in the uploaded course materials.\n\n"
                    "Available chapters:\n\n"
                    f"{chapter_list}"
                )

        elif week_match:

            week_num = week_match.group(1).zfill(2)

            top_indices = [
                i
                for i, source in enumerate(document_sources)
                if f"week{week_num}" in source.lower()
            ][:5]

            if not top_indices:

                week_list = "\n".join(
                    [f"• Week {w}" for w in sorted(available_weeks)]
                )

                return (
                    f"I could not find Week {week_num} "
                    "in the uploaded course materials.\n\n"
                    "Available weeks:\n\n"
                    f"{week_list}"
                )

        else:

            if (
                study_guide
                or summary_request
                or quiz_request
                or flashcards_request
            ):
                top_indices = similarities.argsort()[-5:][::-1]

            else:
                top_indices = similarities.argsort()[-3:][::-1]

# ==========================================
# CALCULATE BEST SCORE
# ==========================================

        best_score = similarities[top_indices[0]]

# ==========================================
# DEBUGGING
# ==========================================

        print("\n================================")
        print(f"Question: {message}")
        print(f"Best Score: {best_score:.3f}")

        for i in top_indices:

            print(
                f"Score: {similarities[i]:.3f} | "
                f"Source: {document_sources[i]}"
            )

        print("================================\n")


        print("\nRetrieved Sources:")

        for i in top_indices:

            print(
                f"- {document_sources[i]}"
            )

        print()

# ==========================================
# RELEVANCE CHECK
# ==========================================

        
        THRESHOLD = 0.35
        if (
            best_score < THRESHOLD
            and not study_guide
            and not summary_request
            and not comparison_request
            and not quiz_request
            and not flashcards_request
        ):
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

        context = context[:4000]

        sources = list(
            set(document_sources[i] for i in top_indices)
        )

        # =================================================
        # PROMPT TYPE DETECTION
        # =================================================

        if study_guide:

            prompt = f"""
            Create a detailed study guide.

            Organize the study guide using these sections:

            1. Main Concepts
            2. Important Definitions
            3. Key Topics
            4. Exam Review Points
            5. Important Takeaways
            6. Practice Questions

            Use bullet points where appropriate.

            Course Materials:
            {context}

            Question:
            {message}
            """

        elif comparison_request:

            prompt = f"""
            You are an Academic Assistant chatbot.

            Compare the requested concepts using ONLY the course materials.

            Organize your answer as:

            Overview

            Concept 1

            Concept 2

            Key Differences

            Relationship

            Course Materials:
            {context}

            Question:
            {message}
            """

        elif summary_request:

            prompt = f"""
            You are an Academic Assistant chatbot.

            Create a concise summary using ONLY the course materials.

            Include:

            - Main ideas
            - Key concepts
            - Important takeaways

            Course Materials:
            {context}

            Question:
            {message}
            """
        elif quiz_request:

            prompt = f"""
            Create a 10-question quiz using ONLY the course materials.

            Include:
            - Question
            - Answer

            Course Materials:
            {context}

            Question:
            {message}
            """
        else:

            prompt = f"""
            You are an Academic Assistant chatbot.

            Use the provided course materials to answer the question.

            Rules:
            - Use headings and bullet points when helpful.
            - Explain concepts clearly.
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
        # Gemini-2.x-flash
        # =================================================

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text.strip()

        answer = answer.replace("```", "")

        return (
            f"{answer}\n"
            f"---\n"
            f"📚 Sources:\n"
            f"{chr(10).join(['• ' + s for s in sources])}"
        )
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
    description=f"""
    Ask questions about the uploaded course materials.

    Loaded:
    • {len(set(document_sources))} files
    • {len(documents)} document chunks

    Features:
    • Semantic Search
    • Retrieval-Augmented Generation (RAG)
    • Google Gemini 2.5 Flash
    • TXT and PDF support
    • Multi-document support

    Try asking:
    • What is Artificial Intelligence?
    • Compare AI and HCI
    • Create a study guide for Week 7
    • Explain Ethical AI
    • Summarize Chapter 1
    """
)

# =====================================================
# LAUNCH
# =====================================================

if __name__ == "__main__":
    demo.launch()

