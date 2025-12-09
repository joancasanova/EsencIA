Debes realizar un trabajo de fin de grado (TFG) titulado: "Tratamiento de textos para aplicaciones de lectura fácil con LLMs"

El trabajo se basa en refinar un software ya existente (EsencIA) con el objetivo de cumplir con los objetivos del TFG

Resumen general del trabajo: Este trabajo consiste en el desarrollo de un software basado en LLMs llamado "EsencIA" capaz de tratar textos —adaptar, resumir, cambios de tono, extracción de información, etc— de forma autónoma, con una elevada precisión y fiabilidad. Este sistema será configurable para que el usuario pueda determinar la tarea concreta que desee realizar. 


Lista de objetivos concretos del trabajo: 

1. Diseño, implementación, y documentación de una metodología basada en LLMs capaz de tratar textos.
2. Evaluación de la efectividad del sistema para la adaptación de redundancias (detección y corrección de éstas). 
3. Demostración de que la metodología es capaz de aplicarse a otras tareas como el aumento de datos, resumen de textos, o cambios de tono.
4. Creación de una interfaz web (GUI).

Documentación: 
1. docs/Memoria - TFG.DOCX: Se puede encontrar lo desarrollado hasta ahora en este documento. (work in progress)
2. README.md: Explica cómo funciona el software. Muy importante

A tener en cuenta:
1. Es un programa SIN clientes concurrentes.

IMPORTANTE: Debes cerciorarte de que las decisiones que tomes sean las correctas, ya que no hay un supervisor que te ayude en el proceso. Chequea doblemente antes de tomar una decisión.

---

## Metodologia de Desarrollo: TDD (Test-Driven Development)

Este proyecto sigue estrictamente la metodologia TDD. Tras cada cambio en el codigo, se deben ejecutar los tests para verificar que todo funciona correctamente.

### Ciclo TDD Obligatorio

1. **RED**: Escribir test que falla (documenta el comportamiento esperado)
2. **GREEN**: Escribir codigo minimo para pasar el test
3. **REFACTOR**: Mejorar codigo manteniendo tests verdes

### Comandos de Tests

```bash
# Ejecutar todos los tests (OBLIGATORIO antes de cada commit)
python -m pytest tests/ -v --tb=short

# Ejecutar con cobertura
python -m pytest tests/ --cov=app --cov-report=term-missing

# Ejecutar tests rapidos
python -m pytest tests/ -m "not slow" --no-cov

# Verificar cobertura minima (85%)
python -m pytest tests/ --cov=app --cov-fail-under=85
```

### Estructura de Tests

```
tests/
├── conftest.py              # Fixtures compartidos
├── unit/                    # Tests unitarios
│   ├── application/use_cases/
│   ├── domain/entities/
│   ├── domain/services/
│   ├── infrastructure/
│   └── test_main.py
```

### Reglas de Testing

1. **Siempre ejecutar tests** despues de cualquier cambio de codigo
2. **Cobertura minima**: 85% total, 90% para componentes criticos
