from dotenv import load_dotenv
import os
import requests

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


load_dotenv()

PERSIST_DIR = "chroma_db"

SYSTEM_PROMPT = """
Eres un asistente virtual de atención al cliente de TechHogar S.A.
Responde solo con información recuperada de los documentos.
Si no encuentras información suficiente, dilo claramente.
Responde siempre en español.
"""


class GitHubEmbeddings:
    def __init__(self, model: str, token: str, url: str):
        self.model = model
        self.token = token
        self.url = url

    def _embed(self, texts):
        clean_texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not clean_texts:
            return []

        response = requests.post(
            self.url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2026-03-10",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": clean_texts,
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            raise ValueError(f"Respuesta de embeddings vacía: {data}")

        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        embeddings = self._embed([text])
        if not embeddings:
            raise ValueError("No se pudo generar embedding para la consulta.")
        return embeddings[0]


def build_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "desconocido")
        parts.append(f"[Fuente {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def main():
    embeddings = GitHubEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        token=os.getenv("GITHUB_TOKEN"),
        url=os.getenv("GITHUB_EMBEDDINGS_URL"),
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model=os.getenv("CHAT_MODEL"),
        temperature=0,
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url=os.getenv("GITHUB_CHAT_BASE_URL"),
    )

    print("TechHogar RAG listo. Escribe tu pregunta.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        question = input("Cliente: ").strip()
        if question.lower() == "salir":
            break

        docs = retriever.invoke(question)
        context = build_context(docs)

        prompt = f"""
{SYSTEM_PROMPT}

Pregunta del cliente:
{question}

Contexto recuperado:
{context}

Instrucciones:
- Responde de manera clara y breve.
- No inventes datos.
- Si falta información, indica que debe derivarse a un agente humano.
"""

        response = llm.invoke(prompt)

        print("\nAsistente:")
        print(response.content)

        print("\nFuentes usadas:")
        for doc in docs:
            print("-", doc.metadata.get("source", "desconocido"))
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()