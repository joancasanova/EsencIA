# EsencIA: Guía de Uso

**EsencIA** es una aplicación en Python diseñada para realizar tareas avanzadas de procesamiento de texto utilizando modelos de lenguaje (LLMs). Incluye tanto una interfaz de línea de comandos (CLI) como una interfaz web (GUI), y está optimizada para ejecutarse en **hardware de consumo** (GPUs de 4-16GB VRAM).

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Filosofía de Ejecución Local](#filosofía-de-ejecución-local)
3. [Instalación](#instalación)
4. [Interfaz Web (GUI)](#interfaz-web-gui)
5. [Comando `pipeline`](#comando-pipeline)
    -   [Preparación](#preparación)
    -   [Ejecución](#ejecución)
    -   [Archivos de Configuración](#archivos-de-configuración)
    -   [Interpretación de Resultados](#interpretación-de-resultados)
6. [Comando `benchmark`](#comando-benchmark)
    -   [Preparación](#preparación-1)
    -   [Ejecución](#ejecución-1)
    -   [Archivos de Configuración](#archivos-de-configuración-1)
    -   [Interpretación de Resultados](#interpretación-de-resultados-1)
7. [Modelos Recomendados](#modelos-recomendados)
8. [Conclusión](#conclusión)
9. [Licencia](#licencia)

## Introducción

EsencIA ofrece una serie de comandos para interactuar con LLMs, entre ellos:

-   `generate`: Genera texto utilizando un LLM.
-   `parse`: Analiza texto y extrae información estructurada.
-   `verify`: Verifica la calidad del texto generado.
-   `pipeline`: Ejecuta una secuencia de pasos que combinan `generate`, `parse` y `verify` para realizar tareas complejas.
-   `benchmark`: Evalúa el rendimiento de un pipeline en un conjunto de datos de prueba.

Esta guía se centrará en los dos últimos comandos, `pipeline` y `benchmark`, proporcionando instrucciones detalladas para su uso.

## Filosofía de Ejecución Local

EsencIA está diseñado desde su concepción para ejecutarse **localmente** en hardware de consumo, sin depender de APIs externas de pago. Esta filosofía se basa en los siguientes principios:

### Principios de Diseño

1. **Accesibilidad**: Cualquier usuario con una GPU de consumo (4-16GB VRAM) puede ejecutar la aplicación.
2. **Transparencia**: El sistema informa proactivamente sobre las limitaciones de hardware antes de intentar cargar un modelo.
3. **Validación Preventiva**: Antes de cargar un modelo, se estiman los recursos necesarios y se comparan con los disponibles.
4. **Feedback Claro**: Cuando un modelo no puede ejecutarse, se proporcionan mensajes de error detallados y sugerencias de alternativas.

### Sistema de Validación de Recursos

EsencIA incluye un sistema de detección y validación de recursos que:

- **Detecta automáticamente** la GPU disponible, VRAM y RAM del sistema.
- **Estima los requisitos** de un modelo basándose en su nombre y características conocidas.
- **Valida la compatibilidad** antes de intentar cargar el modelo.
- **Sugiere alternativas** cuando el modelo seleccionado no es compatible.

### Requisitos de Hardware por Modelo

| Parámetros del Modelo | VRAM Estimada | Hardware Recomendado |
|-----------------------|---------------|----------------------|
| < 1B (tiny/small)     | ~2-3 GB       | GPU 4GB (GTX 1650, RTX 3050) |
| 1-3B (medium)         | ~4-8 GB       | GPU 6-8GB (RTX 2060, RTX 3060) |
| 3-7B (large)          | ~8-16 GB      | GPU 10-16GB (RTX 3080, RTX 4070) |
| > 7B (xlarge)         | > 16 GB       | GPU profesional o múltiples GPUs |

### Modo CPU

Si no hay GPU disponible o la VRAM es insuficiente, EsencIA puede ejecutarse en modo CPU. Sin embargo:
- La inferencia será **significativamente más lenta** (5-10x).
- Se requiere **más RAM** (~20% adicional sobre los requisitos de VRAM).
- Se recomienda usar modelos pequeños (< 3B parámetros) en este modo.

## Instalación

1. **Clonar el repositorio**:

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd EsencIA
    ```

2. **Crear y activar un entorno virtual** (recomendable):

    ```bash
    python -m venv .venv
    # Linux/Mac:
    source .venv/bin/activate
    # Windows:
    .venv\Scripts\activate
    ```

3. **Instalar dependencias** usando `pyproject.toml`:

    ```bash
    pip install --upgrade pip
    pip install -e ".[dev]"
    ```

    Esto instalará las librerías principales (transformers, typer, etc.) y las de desarrollo (pytest, black, pylint...).

## Interfaz Web (GUI)

EsencIA incluye una interfaz web construida con NiceGUI que permite:

- **Visualizar recursos del sistema**: GPU, VRAM, RAM disponibles.
- **Ver modelos recomendados**: Basados en el hardware detectado.
- **Configurar y ejecutar pipelines**: Sin necesidad de editar archivos JSON manualmente.
- **Ejecutar benchmarks**: Con visualización de métricas y matriz de confusión.

### Ejecución de la GUI

Para iniciar la interfaz web:

```bash
python gui/main.py
```

La aplicación se abrirá en `http://localhost:8080`.

### Características de la GUI

#### Página Principal
- Muestra información del sistema (GPU, RAM, CPU).
- Lista modelos recomendados para tu hardware.
- Indica el estado de uso de recursos en tiempo real.

#### Página Pipeline
- Carga configuraciones de pipeline desde archivo o de forma interactiva.
- Validación de compatibilidad del modelo seleccionado.
- Ejecución con feedback en tiempo real.
- Visualización de resultados estructurados.

#### Página Benchmark
- Carga de configuración y entradas de prueba.
- Validación de modelo antes de ejecutar.
- Métricas de rendimiento (Accuracy, Precision, Recall, F1).
- Matriz de confusión visual.
- Análisis de casos mal clasificados.

## Comando `pipeline`

El comando `pipeline` permite ejecutar una serie de pasos de procesamiento de texto definidos en un archivo de configuración. Estos pasos pueden incluir generación de texto, análisis y verificación, y pueden utilizar datos de referencia para personalizar el proceso.

### Preparación

Antes de ejecutar el comando `pipeline`, asegúrese de tener los siguientes archivos en la carpeta `config/pipeline`:

-   `pipeline_config.json`: Define la secuencia de pasos a ejecutar.
-   `pipeline_reference_data.json`: Contiene los datos de referencia que se utilizarán en los pasos del pipeline.

### Ejecución

Para ejecutar el pipeline, utilice el siguiente comando:

```bash
python app/main.py pipeline --config config/pipeline/pipeline_config.json --pipeline-model-name <model_name>
```

-   `--config`: Ruta al archivo de configuración del pipeline (por defecto: `config/pipeline/pipeline_config.json`).
-   `--pipeline-model-name`: (Opcional) especifica el modelo a usar en todos los pasos del pipeline (generación y verificación). Por defecto: `Qwen/Qwen2.5-1.5B-Instruct`.

**Ejemplo:**

```bash
python app/main.py pipeline --config config/pipeline/pipeline_config.json --pipeline-model-name "mistralai/Mistral-7B-Instruct-v0.2"
```

Este comando ejecutará el pipeline definido en `pipeline_config.json`, utilizando `mistralai/Mistral-7B-Instruct-v0.2` para todos los pasos. Los resultados se guardarán en la carpeta `out/pipeline`.

### Archivos de Configuración

#### `pipeline_config.json`

Este archivo define la secuencia de pasos del pipeline. Cada paso puede ser de tipo `generate`, `parse` o `verify`.

**Ejemplo:**

```json
{
    "steps": [
        {
            "type": "generate",
            "parameters": {
                "system_prompt": "Eres un asistente experto en {tema}.",
                "user_prompt": "Genera una descripción de {subtema}.",
                "num_sequences": 2,
                "max_tokens": 100,
                "temperature": 0.7
            },
            "uses_reference": true,
            "reference_step_numbers": []
        },
        {
            "type": "parse",
            "parameters": {
                "rules": [
                    {
                        "name": "Concepto",
                        "mode": "KEYWORD",
                        "pattern": "Concepto:",
                        "secondary_pattern": "Explicación:"
                    },
                    {
                        "name": "Explicacion",
                        "mode": "REGEX",
                        "pattern": "Explicación:\\s*(.+)",
                        "fallback_value": "No se encontró explicación"
                    }
                ],
                "output_filter": "successful"
            },
            "uses_reference": true,
            "reference_step_numbers": [0]
        },
        {
            "type": "verify",
            "parameters": {
                "methods": [
                    {
                        "mode": "CUMULATIVE",
                        "name": "Verificar_Concepto",
                        "system_prompt": "Eres un verificador de conceptos. ¿El siguiente texto contiene el concepto {Concepto}?",
                        "user_prompt": "{output_1}",
                        "valid_responses": ["{Concepto}"],
                        "num_sequences": 3,
                        "required_matches": 2
                    }
                ],
                "required_for_confirmed": 1,
                "required_for_review": 0
            },
            "uses_reference": true,
            "reference_step_numbers": [0, 1]
        }
    ],
    "global_references": {
        "tema": "ciencia ficción",
        "subtema": "viajes en el tiempo"
    }
}
```

**Explicación de los campos:**

-   `steps`: Una lista de pasos a ejecutar.
    -   `type`: El tipo de paso (`generate`, `parse` o `verify`).
    -   `parameters`: Los parámetros específicos para cada tipo de paso.
        -   `system_prompt`: (Solo en `generate` y `verify`) Define el contexto general para el LLM.
        -   `user_prompt`: (Solo en `generate` y `verify`) Define la tarea específica para el LLM.
        -   `num_sequences`: (Solo en `generate` y `verify`) Número de respuestas a generar.
        -   `max_tokens`: (Solo en `generate` y `verify`) Longitud máxima de la respuesta.
        -   `temperature`: (Solo en `generate` y `verify`) Controla la creatividad del LLM.
        -   `rules`: (Solo en `parse`) Define las reglas de extracción de información.
            -   `name`: Nombre de la regla.
            -   `mode`: Tipo de regla (`REGEX` o `KEYWORD`).
            -   `pattern`: Patrón de búsqueda.
            -   `secondary_pattern`: (Opcional) Patrón de límite para `KEYWORD`.
            -   `fallback_value`: (Opcional) Valor por defecto si no se encuentra el patrón.
        -   `output_filter`: (Solo en `parse`) Filtro para los resultados del parseo (`all`, `successful`, `first`, `first_n`).
        -   `output_limit`: (Solo en `parse`) Límite de resultados para `first_n`.
        -   `methods`: (Solo en `verify`) Define los métodos de verificación.
            -   `mode`: Modo de verificación (`CUMULATIVE` o `ELIMINATORY`).
            -   `name`: Nombre del método de verificación.
            -   `system_prompt`: Define el contexto para el LLM en la verificación.
            -   `user_prompt`: Define la tarea específica de verificación para el LLM.
            -   `valid_responses`: Lista de respuestas válidas.
            -   `num_sequences`: Número de respuestas a generar para la verificación.
            -   `required_matches`: Número mínimo de respuestas que deben coincidir con `valid_responses`.
        -   `required_for_confirmed`: (Solo en `verify`) Número mínimo de métodos de verificación que deben pasar para considerar el resultado como confirmado.
        -   `required_for_review`: (Solo en `verify`) Número mínimo de métodos de verificación que deben pasar para considerar el resultado como revisión.
    -   `uses_reference`: Indica si el paso utiliza datos de referencia.
    -   `reference_step_numbers`: Lista de números de paso (empezando desde 0) cuyos resultados se utilizarán como referencia.
-   `global_references`: Un diccionario de datos de referencia que estarán disponibles para todos los pasos.

#### `pipeline_reference_data.json`

Este archivo contiene los datos de referencia que se utilizarán en los pasos del pipeline. Cada entrada en este archivo es un diccionario que se utilizará para sustituir los placeholders en los prompts de los pasos.

**Ejemplo:**

```json
[
    {
        "tema": "historia",
        "subtema": "antiguo Egipto"
    },
    {
        "tema": "biología",
        "subtema": "fotosíntesis"
    }
]
```

En este ejemplo, cada diccionario en la lista se usará en una ejecución separada del pipeline. En el primer paso `generate`, los placeholders `{tema}` y `{subtema}` se sustituirán por `"historia"` y `"antiguo Egipto"` respectivamente en la primera ejecución, y por `"biología"` y `"fotosíntesis"` en la segunda.

### Interpretación de Resultados

El comando `pipeline` genera los siguientes archivos de salida en la carpeta `out/pipeline`:

-   `results/pipeline_results.json`: Contiene los resultados de cada paso del pipeline.
-   `verification/confirmed/confirmed.json`: Contiene los datos de referencia de los resultados que fueron confirmados por el paso de verificación.
-   `verification/to_verify/to_verify.json`: Contiene los datos de referencia de los resultados que requieren revisión manual.

#### `results/pipeline_results.json`

Este archivo contiene una lista de los resultados de cada paso. Cada entrada tiene la siguiente estructura:

```json
{
    "step_type": "generate",
    "step_data": [
        {
            "content": "...",
            "metadata": {
                "model_name": "...",
                "system_prompt": "...",
                "user_prompt": "...",
                "temperature": 1.0,
                "tokens_used": 185,
                "generation_time": 0.5123,
                "timestamp": "2024-07-24T14:35:12.345678"
            },
            "reference_data": {
                "tema": "ciencia ficción",
                "subtema": "viajes en el tiempo"
            }
        },
        // ... más resultados del paso ...
    ]
}
```

-   `step_type`: El tipo de paso (`generate`, `parse` o `verify`).
-   `step_data`: Una lista de resultados. La estructura de cada resultado depende del tipo de paso.
    -   Para `generate`:
        -   `content`: El texto generado.
        -   `metadata`: Metadatos de la generación, como el modelo utilizado, los prompts, la temperatura, el número de tokens utilizados y el tiempo de generación.
        -   `reference_data`: Los datos de referencia utilizados en este paso.
    -   Para `parse`:
        -   `entries`: Una lista de diccionarios, donde cada diccionario representa una entidad extraída del texto. Las claves son los nombres de las reglas y los valores son los textos extraídos.
    -   Para `verify`:
        -   `final_status`: El estado final de la verificación (`confirmed`, `review` o `discarded`).
        -   `success_rate`: La proporción de métodos de verificación que pasaron.
        -   `reference_data`: Los datos de referencia utilizados en este paso.
        -   `results`: Una lista de resultados de cada método de verificación.
            -   `method_name`: El nombre del método de verificación.
            -   `mode`: El modo del método de verificación.
            -   `passed`: Indica si el método de verificación pasó o no.
            -   `score`: La puntuación del método de verificación.
            -   `timestamp`: La marca de tiempo de la verificación.
            -   `details`: Detalles adicionales sobre la verificación.

#### `verification/confirmed/confirmed.json`

Este archivo contiene una lista de los datos de referencia de los resultados que fueron confirmados por el paso de verificación.

**Ejemplo:**

```json
[
    {
        "tema": "ciencia ficción",
        "subtema": "viajes en el tiempo",
        "Concepto": "Paradoja del abuelo",
        "Explicacion": "Si viajas en el tiempo y matas a tu abuelo..."
    },
    // ... más datos de referencia confirmados ...
]
```

#### `verification/to_verify/to_verify.json`

Este archivo contiene una lista de los datos de referencia de los resultados que requieren revisión manual.

**Ejemplo:**

```json
[
    {
        "tema": "biología",
        "subtema": "fotosíntesis",
        "Concepto": "Clorofila",
        "Explicacion": "Pigmento verde que captura la luz solar..."
    },
    // ... más datos de referencia que necesitan revisión ...
]
```

## Comando `benchmark`

El comando `benchmark` evalúa el rendimiento de un pipeline en un conjunto de datos de prueba. Permite medir la precisión, la exhaustividad y la puntuación F1 del pipeline, así como identificar los casos mal clasificados.

### Preparación

Antes de ejecutar el comando `benchmark`, asegúrese de tener los siguientes archivos en la carpeta `config/benchmark`:

-   `benchmark_config.json`: Define el pipeline a evaluar y los parámetros de evaluación.
-   `benchmark_entries.json`: Contiene el conjunto de datos de prueba.

### Ejecución

Para ejecutar el benchmark, utilice el siguiente comando:

```bash
python app/main.py benchmark --config config/benchmark/benchmark_config.json --entries config/benchmark/benchmark_entries.json
```

-   `--config`: Ruta al archivo de configuración del benchmark (por defecto: `config/benchmark/benchmark_config.json`).
-   `--entries`: Ruta al archivo con los datos de prueba (por defecto: `config/benchmark/benchmark_entries.json`).

**Ejemplo:**

```bash
python app/main.py benchmark --config config/benchmark/benchmark_config.json --entries config/benchmark/benchmark_entries.json
```

Este comando ejecutará el pipeline definido en `benchmark_config.json` en cada entrada del archivo `benchmark_entries.json` y calculará las métricas de rendimiento. Los resultados se guardarán en la carpeta `out/benchmark`.

### Archivos de Configuración

#### `benchmark_config.json`

Este archivo define el pipeline a evaluar y los parámetros de evaluación.

**Ejemplo:**

```json
{
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "pipeline_steps": [
        {
            "type": "generate",
            "parameters": {
                "system_prompt": "Eres un clasificador de texto.",
                "user_prompt": "Clasifica el siguiente texto como 'spam' o 'no spam': {text}",
                "num_sequences": 1,
                "max_tokens": 20,
                "temperature": 0.1
            },
            "uses_reference": true,
            "reference_step_numbers": []
        },
        {
            "type": "verify",
            "parameters": {
                "methods": [
                    {
                        "mode": "ELIMINATORY",
                        "name": "Verificar_Clasificacion",
                        "system_prompt": "Eres un verificador de clasificación. ¿La clasificación '{prediction}' es correcta para el texto '{text}'?",
                        "user_prompt": "El texto original es: {text}",
                        "valid_responses": ["Sí", "Si", "yes", "Yes"],
                        "num_sequences": 1,
                        "required_matches": 1
                    }
                ],
                "required_for_confirmed": 1,
                "required_for_review": 0
            },
            "uses_reference": true,
            "reference_step_numbers": [0]
        }
    ],
    "label_key": "label",
    "label_value": "spam"
}
```

**Explicación de los campos:**

-   `model_name`: El nombre del modelo de lenguaje a utilizar.
-   `pipeline_steps`: La definición del pipeline a evaluar (ver la sección [Comando `pipeline`](#comando-pipeline) para más detalles).
-   `label_key`: La clave en el archivo `benchmark_entries.json` que contiene la etiqueta real de cada entrada.
-   `label_value`: El valor de la etiqueta que se considera como positivo.

#### `benchmark_entries.json`

Este archivo contiene el conjunto de datos de prueba. Cada entrada es un diccionario que contiene el texto a clasificar y su etiqueta real.

**Ejemplo:**

```json
[
    {
        "text": "¡Gana un iPhone gratis! Haz clic aquí.",
        "label": "spam"
    },
    {
        "text": "Hola, ¿cómo estás? Te escribo para preguntarte...",
        "label": "no spam"
    },
    {
        "text": "Oferta especial: 50% de descuento en todos los productos.",
        "label": "spam"
    }
]
```

En este ejemplo, cada diccionario en la lista tiene una clave `text` que contiene el texto a clasificar y una clave `label` que contiene la etiqueta real (`spam` o `no spam`).

### Interpretación de Resultados

El comando `benchmark` genera los siguientes archivos de salida en la carpeta `out/benchmark`:

-   `results/benchmark_results.json`: Contiene las métricas de rendimiento del pipeline.
-   `misclassified/misclassified_*.json`: Contiene los casos mal clasificados por el pipeline.

#### `results/benchmark_results.json`

Este archivo contiene las siguientes métricas de rendimiento:

```json
{
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "f1_score": 0.85,
    "confusion_matrix": {
        "true_positive": 42,
        "false_positive": 9,
        "true_negative": 43,
        "false_negative": 6
    },
    "misclassified_count": 15
}
```

-   `accuracy`: Precisión general del pipeline.
-   `precision`: Proporción de verdaderos positivos entre los casos clasificados como positivos.
-   `recall`: Proporción de verdaderos positivos entre los casos que realmente son positivos.
-   `f1_score`: Media armónica de precisión y exhaustividad.
-   `confusion_matrix`: Matriz de confusión que muestra el número de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
-   `misclassified_count`: Número de casos mal clasificados.

#### `misclassified/misclassified_*.json`

Este archivo contiene una lista de los casos mal clasificados. Cada entrada tiene la siguiente estructura:

```json
{
    "input_data": {
        "text": "Este es un texto que fue mal clasificado."
    },
    "predicted_label": "spam",
    "actual_label": "no spam",
    "timestamp": "2024-07-24T14:35:12.345678"
}
```

-   `input_data`: Los datos de entrada del caso mal clasificado.
-   `predicted_label`: La etiqueta predicha por el pipeline.
-   `actual_label`: La etiqueta real del caso.
-   `timestamp`: La marca de tiempo de la predicción.

## Modelos Recomendados

EsencIA recomienda los siguientes modelos según el hardware disponible:

### Para GPUs de Gama Baja (4-6GB VRAM)

| Modelo | Parámetros | VRAM | Descripción |
|--------|------------|------|-------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~2GB | Muy rápido, ideal para pruebas |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | ~3GB | Buen balance velocidad/calidad |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~4GB | Recomendado para uso general |

### Para GPUs de Gama Media (8-12GB VRAM)

| Modelo | Parámetros | VRAM | Descripción |
|--------|------------|------|-------------|
| `microsoft/phi-2` | 2.7B | ~6GB | Alta calidad para su tamaño |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~7GB | Mejor calidad, más recursos |

### Para GPUs de Gama Alta (16GB+ VRAM)

| Modelo | Parámetros | VRAM | Descripción |
|--------|------------|------|-------------|
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | ~14GB | Alta calidad, recomendado |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14GB | Excelente para tareas complejas |

### Consideraciones sobre Cuantización

Para ejecutar modelos más grandes en hardware limitado, considere usar versiones cuantizadas:

- **AWQ/GPTQ**: Reducen VRAM ~50% con pérdida mínima de calidad.
- **GGUF (llama.cpp)**: Permite ejecución eficiente en CPU.

Ejemplo: `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` requiere ~7GB VRAM en lugar de ~14GB.

## Conclusión

Esta guía ha proporcionado una descripción detallada de cómo utilizar los comandos `pipeline` y `benchmark` de EsencIA, así como la interfaz web. Con esta información, podrá ejecutar pipelines de procesamiento de texto personalizados y evaluar su rendimiento en conjuntos de datos de prueba.

La filosofía de ejecución local de EsencIA garantiza que pueda aprovechar modelos de lenguaje avanzados sin depender de servicios externos costosos, utilizando únicamente el hardware disponible en su equipo.
