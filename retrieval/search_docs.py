from pathlib import Path

def search_documents(question):

    file_path = Path("data/AI_Basics.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content


if __name__ == "__main__":

    result = search_documents(
        "What is machine learning?"
    )

    print(result)