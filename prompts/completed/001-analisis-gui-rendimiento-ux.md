<objective>
Realizar un análisis exhaustivo de la GUI de EsencIA (gui/main.py) para identificar problemas de rendimiento y UX.

El usuario reporta:
1. La carga inicial de las pantallas tarda varios segundos
2. El buscador de modelos de HuggingFace en Pipeline es lento
3. Se necesita mejorar la fluidez general para el usuario

Este análisis informará las optimizaciones posteriores.
</objective>

<context>
Proyecto: EsencIA - Software de procesamiento de textos con LLMs
Framework: NiceGUI (basado en Vue.js + FastAPI)
Archivo principal: gui/main.py (~2500 líneas)

Estructura de la GUI:
- 3 pestañas: Inicio, Pipeline, Benchmark
- Buscador de modelos HuggingFace con autocompletado
- Sistema de detección de recursos del sistema (RAM, GPU, VRAM)
- Lazy loading parcial para torch y use cases pesados

Lee el CLAUDE.md para contexto adicional del proyecto.
</context>

<data_sources>
Archivos a analizar:
- @gui/main.py (archivo principal - analizar exhaustivamente)
- @gui/test_resources_display.py (si existe, puede tener pistas)
- @app/config/ (configuraciones que pueden afectar rendimiento)
</data_sources>

<analysis_requirements>
Realizar análisis profundo en estas áreas:

1. **CARGA INICIAL (CRÍTICO)**
   - Identificar qué se ejecuta síncronamente al cargar la página
   - Analizar el ciclo de vida de @ui.page('/') y main()
   - Evaluar si las 3 pestañas se renderizan todas al inicio
   - Detectar imports pesados que bloquean el arranque
   - Buscar llamadas a get_system_resources() o similares en init
   - Medir/estimar qué operaciones son costosas

2. **BUSCADOR DE MODELOS HUGGINGFACE**
   - Analizar search_huggingface_models() y su performance
   - Evaluar el sistema de cache (_hf_search_cache)
   - Revisar preload_recommended_models() y su efectividad
   - Analizar el debouncing actual (si existe)
   - Evaluar el timeout de httpx (2 segundos)
   - Revisar el evento keyup y su manejo

3. **RENDERIZADO Y REACTIVIDAD**
   - Identificar re-renders innecesarios
   - Buscar operaciones costosas en callbacks
   - Evaluar el uso de ui.refreshable y ui.update()
   - Detectar CSS/estilos que puedan causar reflow

4. **GESTIÓN DE ESTADO**
   - Analizar AppState y su uso
   - Buscar mutaciones de estado que causen re-renders
   - Evaluar el uso de session_state si existe

5. **PATRONES PROBLEMÁTICOS**
   - Buscar bucles síncronos largos
   - Detectar await sin necesidad
   - Identificar funciones que deberían ser async pero no lo son
   - Buscar logging excesivo o print() en hot paths
</analysis_requirements>

<output_format>
Generar un informe estructurado con:

## Resumen Ejecutivo
[3-5 bullet points con hallazgos más críticos]

## Problemas Críticos (Alta Prioridad)
Para cada problema:
- **Ubicación**: archivo:línea
- **Descripción**: Qué está mal
- **Impacto**: Cómo afecta al usuario
- **Solución propuesta**: Cómo solucionarlo

## Problemas Moderados (Media Prioridad)
[Mismo formato]

## Mejoras Recomendadas (Baja Prioridad)
[Mismo formato]

## Métricas de Referencia
[Si es posible, estimar tiempos o identificar benchmarks]

## Plan de Acción Priorizado
[Lista ordenada de acciones a tomar]

Guardar el informe en: `./docs/analisis-gui-rendimiento.md`
</output_format>

<verification>
Antes de completar el análisis, verificar:
- [ ] Se analizó todo gui/main.py, no solo fragmentos
- [ ] Se identificaron al menos 3 problemas de carga inicial
- [ ] Se analizó el buscador de modelos en detalle
- [ ] Se propusieron soluciones concretas y accionables
- [ ] El informe incluye ubicaciones específicas (archivo:línea)
</verification>

<success_criteria>
- Informe completo en ./docs/analisis-gui-rendimiento.md
- Problemas ordenados por prioridad/impacto
- Cada problema tiene solución propuesta concreta
- El análisis es lo suficientemente detallado para implementar mejoras sin investigación adicional
</success_criteria>
