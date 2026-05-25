from dotenv import load_dotenv
import os
import requests
import re

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()

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
                "X-GitHub-Api-Version": "2022-11-28",
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
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


# Memoria simple de conversación
memoria = []


def crear_retriever():
    embeddings = GitHubEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        token=os.getenv("GITHUB_TOKEN"),
        url=os.getenv("GITHUB_EMBEDDINGS_URL"),
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = crear_retriever()

llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL"),
    temperature=0,
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_CHAT_BASE_URL"),
)


def herramienta_rag(pregunta):
    docs = retriever.invoke(pregunta)

    if not docs:
        return "No se encontró información suficiente en los documentos."

    contexto = []
    for doc in docs:
        fuente = doc.metadata.get("source", "fuente desconocida")
        contexto.append(f"Fuente: {fuente}\nContenido: {doc.page_content}")

    return "\n\n".join(contexto)


def herramienta_calculadora(pregunta):
    try:
        numeros = re.findall(r"\d+", pregunta)
        if len(numeros) >= 2 and "%" in pregunta:
            precio = float(numeros[0])
            descuento = float(numeros[1])
            resultado = precio - (precio * descuento / 100)
            return f"El precio final con {descuento}% de descuento es {round(resultado)}."
        return "No se pudo identificar una operación matemática válida."
    except Exception:
        return "No se pudo realizar el cálculo."


def decidir_herramienta(pregunta):
    pregunta_lower = pregunta.lower()

    if any(palabra in pregunta_lower for palabra in ["descuento", "%", "calcula", "precio final"]):
        return "calculadora"

    if any(palabra in pregunta_lower for palabra in ["garantía", "garantia", "stock", "devolución", "devolucion", "despacho", "producto", "iphone", "samsung"]):
        return "rag"

    if any(palabra in pregunta_lower for palabra in ["y cuánto", "y cuanto", "y eso", "también", "tambien"]):
        return "memoria_rag"

    return "rag"


def generar_respuesta(pregunta, contexto, herramienta_usada):
    historial = "\n".join(memoria[-6:])

    prompt = f"""
Eres un agente inteligente de atención al cliente de TechHogar S.A.

Debes responder en español, de forma clara y breve.
No inventes información.
Usa solo el contexto entregado.
Si no existe información suficiente, indícalo.

Historial de conversación:
{historial}

Herramienta utilizada:
{herramienta_usada}

Pregunta del cliente:
{pregunta}

Contexto recuperado:
{contexto}

Respuesta:
"""

    response = llm.invoke(prompt)
    return response.content


def main():
    print("Agente funcional TechHogar listo.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        pregunta = input("Cliente: ").strip()

        if pregunta.lower() == "salir":
            break

        herramienta = decidir_herramienta(pregunta)

        if herramienta == "calculadora":
            contexto = herramienta_calculadora(pregunta)
        elif herramienta == "memoria_rag":
            pregunta_con_memoria = " ".join(memoria[-4:]) + " " + pregunta
            contexto = herramienta_rag(pregunta_con_memoria)
        else:
            contexto = herramienta_rag(pregunta)

        respuesta = generar_respuesta(pregunta, contexto, herramienta)

        memoria.append(f"Cliente: {pregunta}")
        memoria.append(f"Agente: {respuesta}")

        print("\nHerramienta usada:", herramienta)
        print("\nAgente:")
        print(respuesta)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()