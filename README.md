# TechHogar Agent

Sistema inteligente de atención al cliente basado en modelos de lenguaje (LLM), arquitectura RAG, agentes funcionales y observabilidad.

El sistema permite responder consultas utilizando información contenida en documentos internos de la empresa TechHogar, incorporando recuperación semántica, memoria conversacional, herramientas de razonamiento, escritura, planificación y métricas de monitoreo.

La versión de agente funcional fue implementada con LangChain Agents mediante `create_agent`, permitiendo que el modelo de lenguaje participe en la selección de herramientas disponibles. A diferencia de un enrutador por palabras clave, el LLM analiza la intención de la consulta y decide qué herramienta utilizar dentro del flujo de trabajo.

---

## Requisitos

* Python 3.10 o superior
* Git
* Token de GitHub Models con permisos de modelos
* Visual Studio Code
* Navegador web para visualizar el dashboard

---

## Configuración

1. Crear una carpeta y abrirla en Visual Studio Code.

2. Clonar el repositorio:

git clone https://github.com/NicolasMolinaV/techhogar.git

3. Entrar a la carpeta del proyecto:

cd techhogar

4. Crear entorno virtual:

python -m venv venv

5. Activar entorno virtual:

venv\Scripts\activate

6. Instalar dependencias:

pip install -r requirements.txt

7. Crear archivo `.env` en la raíz del proyecto:

GITHUB_TOKEN=tu_token

CHAT_MODEL=openai/gpt-4o

EMBEDDING_MODEL=openai/text-embedding-3-small

GITHUB_CHAT_BASE_URL=https://models.github.ai/inference

GITHUB_EMBEDDINGS_URL=https://models.github.ai/inference/embeddings

---

## Ejecución del sistema

### Paso 1: Indexar los datos

python ingest.py

Este script:

* carga los documentos desde la carpeta `data/`
* genera embeddings mediante GitHub Models
* crea la base vectorial en `chroma_db/`

---

### Paso 2: Ejecutar el asistente RAG básico

python app.py

Este modo ejecuta el sistema RAG base, permitiendo realizar consultas sobre los documentos internos de TechHogar.

---

### Paso 3: Ejecutar el agente funcional

python agent.py

Este modo ejecuta el agente inteligente con:

* LangChain Agents mediante `create_agent`
* memoria de corto plazo mediante `InMemorySaver`
* recuperación semántica con ChromaDB
* herramientas de consulta, cálculo, escritura y planificación
* toma de decisiones realizada por el modelo de lenguaje
* registro de logs y métricas de observabilidad

---

## Herramientas del agente

El agente utiliza distintas herramientas dependiendo del tipo de consulta realizada por el usuario. Estas herramientas son entregadas al agente mediante LangChain, permitiendo que el LLM analice la intención de la consulta y seleccione cuál utilizar.

| Herramienta               | Función                                                                   |
| ------------------------- | ------------------------------------------------------------------------- |
| `consultar_documentos`    | Busca información dentro de los documentos internos usando RAG y ChromaDB |
| `calcular_descuento`      | Realiza cálculos matemáticos, descuentos y precios finales                |
| `generar_resumen_soporte` | Genera resúmenes para derivación a soporte humano                         |
| `planificar_atencion`     | Organiza casos con múltiples pasos antes de responder                     |

---

## Memoria del agente

El agente incorpora memoria de corto plazo mediante `InMemorySaver`, permitiendo mantener continuidad dentro de una misma conversación.

La memoria se gestiona usando un `thread_id`, lo que permite que el agente conserve el historial de interacción y pueda responder consultas posteriores considerando el contexto anterior.

---

## Ejemplos del agente funcional

### Consulta documental

Cliente:

¿Cuánto dura la garantía de Samsung?

Respuesta esperada:

La garantía de los productos Samsung ofrecidos por TechHogar tiene una duración mínima de 12 meses.

---

### Consulta con memoria

Cliente:

¿Cuánto dura la garantía de Samsung?

Luego:

¿Y qué pasa si necesito despacho?

El agente mantiene el contexto de la conversación y responde utilizando la información de despacho disponible en los documentos.

---

### Consulta matemática

Cliente:

Si un producto vale 699990 y tiene 15% de descuento, cuánto queda?

Respuesta esperada:

El precio final del producto, aplicando un descuento del 15%, queda en $594.992 aproximadamente.

---

### Consulta de derivación

Cliente:

Tengo un reclamo y necesito ayuda humana

Respuesta esperada:

El agente genera un resumen de la solicitud y deriva el caso a soporte humano especializado.

---

### Consulta con planificación

Cliente:

Necesito resolver un caso donde quiero saber garantía, despacho y posible derivación

El agente organiza la solicitud en distintas etapas, consulta información documental, entrega la respuesta correspondiente y considera la posibilidad de derivación.

---

## Observabilidad - Evaluación 3

La Evaluación 3 incorpora métricas de observabilidad sobre el agente funcional de TechHogar, permitiendo registrar, analizar y visualizar su comportamiento durante la ejecución.

Se implementaron los siguientes componentes:

* `observability.py`: registra logs de ejecución del agente.
* `dashboard.py`: muestra un dashboard visual con métricas del agente.
* `run_tests.py`: calcula métricas de precisión, latencia, errores y resumen general desde los logs.
* `logs/agent_logs.csv`: almacena registros de ejecución.
* `logs/evaluation_results.csv`: almacena resultados de precisión.
* `logs/metrics_summary.csv`: almacena un resumen general de métricas.

---

## Métricas implementadas

| Métrica               | Descripción                                                       |
| --------------------- | ----------------------------------------------------------------- |
| Latencia              | Tiempo que demora el agente en responder                          |
| Frecuencia de errores | Cantidad de ejecuciones con estado ERROR                          |
| Tasa de error         | Porcentaje de errores sobre el total de consultas                 |
| Precisión             | Validación de respuestas mediante palabras clave esperadas        |
| Consistencia          | Comparación de respuestas ante consultas similares o equivalentes |
| Longitud de respuesta | Cantidad de caracteres generados por respuesta                    |
| Uso de herramientas   | Registro de herramientas utilizadas por el agente                 |
| Trazabilidad          | Registro de fecha, pregunta, respuesta, estado y errores          |

---

## Ejecución de observabilidad

Para ejecutar el agente con registro de logs:

python agent.py

Para calcular métricas desde los logs:

python run_tests.py

Para abrir el dashboard visual:

streamlit run dashboard.py

---

## Resultados observados

Durante las pruebas realizadas se obtuvieron los siguientes resultados:

* Precisión garantía: 100%
* Precisión descuento: 100%
* Precisión reclamo: 100%
* Precisión caso múltiple: 100%
* Total de consultas registradas: 4
* Latencia promedio: 7.315 segundos
* Latencia máxima: 17.349 segundos
* Errores registrados: 0
* Tasa de error: 0.0%

La consulta más lenta correspondió al caso múltiple de garantía, despacho y posible derivación, debido a que el agente utilizó varias herramientas en una misma ejecución.

---

## Hallazgos de observabilidad

A partir de los logs y métricas se identificó que las consultas simples presentan una latencia menor, mientras que los casos con múltiples objetivos aumentan el tiempo de respuesta.

También se observó que el agente no presentó errores durante las pruebas registradas. Sin embargo, durante pruebas automatizadas intensivas se identificó una limitación externa asociada al límite de solicitudes de GitHub Models. Por esta razón, se recomienda implementar pausas, reintentos y control de frecuencia en escenarios de mayor carga.

---

## Dashboard de monitoreo

El dashboard fue desarrollado con Streamlit y permite visualizar el comportamiento del agente a partir de los registros almacenados en `logs/agent_logs.csv`.

El dashboard muestra:

* total de consultas registradas
* latencia promedio
* tasa de error
* herramientas utilizadas
* longitud de respuestas
* registros de ejecución
* análisis automático de consultas lentas y posibles anomalías

Para ejecutarlo:

streamlit run dashboard.py

---

## Seguridad y uso responsable

El sistema considera medidas básicas de seguridad y uso responsable:

* El archivo `.env` no debe subirse al repositorio.
* El token de GitHub Models se mantiene fuera del código fuente.
* Los datos utilizados son simulados para fines académicos.
* No se almacenan datos personales reales de clientes.
* Las respuestas del agente deben basarse en documentos internos y no inventar información.
* Los errores y límites del proveedor externo deben ser monitoreados para evitar fallas en producción.

---

## Validación del sistema

El evaluador podrá verificar:

* que el sistema RAG base indexa documentos correctamente
* que se generan embeddings mediante GitHub Models
* que ChromaDB recupera contexto relevante
* que el agente funcional utiliza LangChain Agents mediante `create_agent`
* que el LLM participa en la selección de herramientas
* que existe memoria conversacional mediante `InMemorySaver`
* que se integran herramientas de consulta, cálculo, escritura y planificación
* que el agente registra logs de ejecución
* que se calculan métricas de precisión, latencia y errores
* que existe un dashboard visual para monitorear el comportamiento del agente
* que existen evidencias y capturas asociadas al funcionamiento del sistema

---

## Estructura del proyecto

techhogar/
├── agent.py
├── app.py
├── ingest.py
├── observability.py
├── dashboard.py
├── run_tests.py
├── requirements.txt
├── README.md
├── data/
│   ├── catalogo.txt
│   ├── garantias.txt
│   ├── devoluciones.txt
│   └── despachos.txt
├── logs/
│   ├── agent_logs.csv
│   ├── evaluation_results.csv
│   └── metrics_summary.csv
├── docs/
│   ├── diagrama_orquestacion_agente.png
│   ├── dashboard_metricas.png
│   ├── dashboard_latencia.png
│   ├── dashboard_herramientas.png
│   ├── dashboard_logs.png
│   └── pruebas_precision_metricas.png

---

## Diagrama de orquestación

El repositorio incluye un diagrama de orquestación en la carpeta `docs/`, donde se representa la relación entre el usuario, el agente LangChain, el modelo de lenguaje, las herramientas, la memoria, la base vectorial y los componentes de observabilidad.

Archivo:

docs/diagrama_orquestacion_agente.png

---

## Decisiones de diseño

| Componente      | Decisión                                     | Justificación                                                          |
| --------------- | -------------------------------------------- | ---------------------------------------------------------------------- |
| LangChain Agent | Uso de `create_agent`                        | Permite que el LLM seleccione herramientas dentro del flujo del agente |
| ChromaDB        | Base vectorial                               | Permite recuperación semántica sobre documentos internos               |
| GitHub Models   | LLM y embeddings                             | Permite usar modelos mediante token y endpoints compatibles            |
| InMemorySaver   | Memoria de corto plazo                       | Mantiene continuidad conversacional durante la sesión                  |
| RAG             | Consulta documental                          | Reduce respuestas inventadas al responder con contexto recuperado      |
| Herramientas    | Consulta, cálculo, escritura y planificación | Permiten cubrir distintos tipos de tareas organizacionales             |
| CSV Logs        | Registro estructurado                        | Permite analizar trazabilidad, errores y desempeño                     |
| Streamlit       | Dashboard visual                             | Permite visualizar métricas de forma clara e interactiva               |

---

## Recomendaciones de mejora

A partir de las métricas observadas se proponen las siguientes mejoras:

* Optimizar el uso de herramientas para evitar llamadas innecesarias.
* Reducir el contexto enviado al modelo en consultas simples.
* Implementar pausas y reintentos ante límites de GitHub Models.
* Agregar monitoreo continuo en escenarios de mayor cantidad de usuarios.
* Incorporar evaluación automática de consistencia con más casos de prueba.
* Mejorar el control de errores para detectar fallas del proveedor externo.
* Mantener logs históricos para comparar el desempeño del agente en el tiempo.

---

## Notas

* El archivo `.env` no debe subirse a GitHub.
* Los datos utilizados son simulados para fines académicos.
* El sistema corresponde a un prototipo funcional de agente organizacional basado en IA.
* Antes de ejecutar `agent.py`, se debe ejecutar `python ingest.py` para crear la base vectorial.
* Para visualizar métricas, primero se deben generar registros ejecutando consultas en `agent.py`.
