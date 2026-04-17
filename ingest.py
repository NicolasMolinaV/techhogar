from pathlib import Path
from dotenv import load_dotenv
import os
import requests

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

DATA_DIR = Path("data")
PERSIST_DIR = "chroma_db"


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


def load_documents():
    docs = []
    for file_path in DATA_DIR.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = file_path.name
        docs.extend(loaded)
    return docs


def main():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = splitter.split_documents(docs)

    texts = [doc.page_content for doc in splits if doc.page_content.strip()]
    metadatas = [doc.metadata for doc in splits if doc.page_content.strip()]

    embeddings = GitHubEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        token=os.getenv("GITHUB_TOKEN"),
        url=os.getenv("GITHUB_EMBEDDINGS_URL"),
    )

    sample = embeddings.embed_documents(texts[:2])
    print("Prueba de embeddings OK. Cantidad:", len(sample))

    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR
    )

    print(f"Documentos cargados: {len(docs)}")
    print(f"Chunks generados: {len(texts)}")
    print("Base vectorial creada en chroma_db/")


if __name__ == "__main__":
    main()