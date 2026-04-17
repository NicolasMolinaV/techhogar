# TechHogar RAG

Prototipo de sistema de atención al cliente basado en LLM y arquitectura
RAG (Retrieval-Augmented Generation).

Este sistema permite responder consultas utilizando información
contenida en documentos internos de la empresa.

------------------------------------------------------------------------

## Requisitos

-   Python 3.10 o superior
-   Git
-   Token de GitHub Models con permisos de modelos

------------------------------------------------------------------------

## Configuración

1.  Clonar el repositorio:

git clone https://github.com/NicolasMolinaV/techhogar.git cd techhogar

2.  Crear entorno virtual:

python -m venv venv

3.  Activar entorno virtual:

venv`\Scripts`{=tex}`\activate`{=tex}

4.  Instalar dependencias:

pip install -r requirements.txt

5.  Crear archivo `.env` en la raíz del proyecto:

GITHUB_TOKEN=tu_token 

CHAT_MODEL=openai/gpt-4o

EMBEDDING_MODEL=openai/text-embedding-3-small

GITHUB_CHAT_BASE_URL=https://models.github.ai/inference

GITHUB_EMBEDDINGS_URL=https://models.github.ai/inference/embeddings

------------------------------------------------------------------------

## Ejecución del sistema

### Paso 1: Indexar los datos

python ingest.py

Este script: - carga los documentos desde la carpeta `data/` - genera
embeddings - crea la base vectorial en `chroma_db/`

------------------------------------------------------------------------

### Paso 2: Ejecutar el asistente

python app.py

El sistema quedará en modo interactivo para ingresar consultas.

------------------------------------------------------------------------

## Ejemplos de uso

-   ¿Cuánto dura la garantía de Samsung?
-   ¿Hay stock de iPhone 14?
-   ¿Cuánto demora un despacho a regiones?
-   ¿Puedo devolver un producto?

------------------------------------------------------------------------

## Validación del sistema

El evaluador podrá verificar: - Que el sistema responde correctamente -
Que las respuestas están basadas en los documentos - Que se muestran las
fuentes utilizadas

Ejemplo esperado:

Asistente: Los productos Samsung cuentan con 12 meses de garantía.

Fuentes usadas: - garantias.txt

------------------------------------------------------------------------

## Estructura del proyecto

techhogar/ ├── app.py ├── ingest.py ├── requirements.txt ├── README.md
├── data/ │ ├── catalogo.txt │ ├── garantias.txt │ ├── devoluciones.txt
│ └── despachos.txt

------------------------------------------------------------------------