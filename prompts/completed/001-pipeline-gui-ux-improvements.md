<objective>
Mejorar la usabilidad y claridad de la pagina PIPELINE en la GUI de EsencIA.

Se requieren 3 mejoras especificas de UX en el archivo gui/main.py:
1. Selector de modelo con autocomplete de HuggingFace mas visible
2. Renombrar "Inicio rapido" a "Selecciona plantilla" con mejor diseno
3. Hacer los botones del Constructor de Pipeline intuitivos con etiquetas claras
</objective>

<context>
Proyecto: EsencIA - Software de tratamiento de textos con LLMs
Framework: NiceGUI (Python)
Archivo a modificar: gui/main.py

El selector de modelo ya tiene implementada la busqueda en HuggingFace (funcion `search_huggingface_models`), pero la UI del dropdown de resultados usa `position: absolute` y puede no ser visible.

Los botones del Constructor (lineas 1842-1846) actualmente son:
- Un contador de pasos
- Un `ui.upload` minimo para importar JSON
- Un boton con icono `file_download` para exportar

El usuario no entiende para que sirven estos botones.
</context>

<requirements>
1. **Selector de Modelo HuggingFace**
   - Mejorar la visibilidad del dropdown de resultados
   - Asegurar que el dropdown aparezca correctamente debajo del input
   - Considerar usar un componente `ui.select` con busqueda en lugar del enfoque manual actual
   - Mantener la busqueda asincrona en la API de HuggingFace

2. **Seccion de Plantillas**
   - Cambiar el texto "Inicio rapido:" por "Selecciona plantilla"
   - Mejorar el diseno visual para que sea mas claro que son opciones seleccionables
   - Anadir una descripcion corta o tooltip en cada plantilla
   - Mantener el mensaje "o construye tu pipeline desde cero" como alternativa

3. **Botones del Constructor de Pipeline**
   - Anadir etiquetas de texto junto a los iconos:
     - Importar: "Importar" o "Cargar config"
     - Exportar: "Exportar" o "Guardar config"
   - Mejorar los tooltips para explicar claramente que hacen
   - Considerar agruparlos visualmente (ej: dentro de un menu desplegable o con separador)
</requirements>

<implementation>
Buscar y modificar estas secciones en gui/main.py:

1. **Lineas ~1782-1822**: Selector de modelo
   - Evaluar usar `ui.select` con `with_input=True` para el autocomplete
   - O mejorar el posicionamiento del `search_results_container`

2. **Lineas ~1824-1834**: Seccion de plantillas
   - Cambiar texto "Inicio rapido:" -> "Selecciona plantilla"
   - Mejorar estilo de los chips de plantilla

3. **Lineas ~1842-1846**: Botones del Constructor
   - Cambiar iconos solos por botones con texto visible
   - Mejorar tooltips

Patrones a seguir del codigo existente:
- Usar clases Tailwind consistentes con el resto de la UI
- Mantener el tema oscuro (bg-slate-*, text-slate-*)
- Usar colores semanticos (purple para Constructor, amber para plantillas)
</implementation>

<verification>
Antes de finalizar, verificar:
1. El dropdown de modelos aparece correctamente al escribir
2. Las plantillas se ven claramente como opciones seleccionables
3. Los botones del Constructor tienen etiquetas legibles
4. Ejecutar tests: `python -m pytest tests/ -v --tb=short`
5. Probar visualmente la GUI: `python gui/main.py`
</verification>

<success_criteria>
- El usuario puede buscar modelos y ver claramente los resultados
- La seccion de plantillas dice "Selecciona plantilla" y es intuitiva
- Los botones de importar/exportar tienen texto visible y son comprensibles
- Todos los tests pasan (620/620)
- La GUI se ve consistente con el resto de la aplicacion
</success_criteria>
