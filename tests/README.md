# Tests for EsencIA

Este directorio contiene la suite de tests unitarios y de integración para el proyecto EsencIA.

## Estructura

```
tests/
├── unit/                           # Tests unitarios
│   ├── domain/                    # Tests del dominio
│   │   ├── entities/              # Tests de entidades
│   │   │   ├── test_parsing.py
│   │   │   ├── test_verification.py
│   │   │   └── test_generation.py
│   │   └── services/              # Tests de servicios
│   │       └── test_parse_service.py
│   ├── application/               # Tests de casos de uso
│   │   └── use_cases/
│   │       └── test_parse_use_case.py
│   └── infrastructure/            # Tests de infraestructura
│       └── test_file_repository.py
├── fixtures/                       # Datos de test compartidos
├── conftest.py                     # Fixtures de pytest
└── README.md                       # Este archivo
```

## Requisitos

Instala las dependencias de desarrollo:

```bash
pip install -r requirements-dev.txt
```

## Ejecutar Tests

### Todos los tests
```bash
pytest
```

### Tests con coverage
```bash
pytest --cov=app --cov-report=html
```

### Solo tests unitarios
```bash
pytest -m unit
```

### Solo tests de integración
```bash
pytest -m integration
```

### Tests específicos
```bash
# Por archivo
pytest tests/unit/domain/entities/test_parsing.py

# Por clase
pytest tests/unit/domain/entities/test_parsing.py::TestParseRule

# Por función
pytest tests/unit/domain/entities/test_parsing.py::TestParseRule::test_valid_parse_rule_creation
```

### Tests con output verbose
```bash
pytest -v
```

### Tests con output detallado
```bash
pytest -vv
```

### Excluir tests lentos
```bash
pytest -m "not slow"
```

## Escribir Tests

### Estructura de un test

```python
import pytest
from app.module import ClassToTest


class TestClassName:
    """Tests for ClassName"""

    def test_method_does_something(self):
        """Test that method does something correctly"""
        # Arrange
        instance = ClassToTest()
        expected = "expected_result"

        # Act
        result = instance.method()

        # Assert
        assert result == expected
```

### Usar fixtures

```python
def test_with_fixture(sample_parse_rules):
    """Test using a fixture from conftest.py"""
    # Arrange
    rules = sample_parse_rules

    # Act & Assert
    assert len(rules) > 0
```

### Tests parametrizados

```python
@pytest.mark.parametrize("input_val,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
    ("test3", "result3"),
])
def test_multiple_cases(input_val, expected):
    """Test multiple cases with parametrize"""
    result = function_to_test(input_val)
    assert result == expected
```

### Verificar excepciones

```python
def test_raises_error():
    """Test that function raises appropriate error"""
    with pytest.raises(ValueError, match="error message"):
        function_that_should_raise()
```

## Coverage

### Ver reporte de coverage
```bash
# Generar reporte HTML
pytest --cov=app --cov-report=html

# Abrir reporte
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Coverage mínimo
Se recomienda mantener un coverage mínimo del 80% para código crítico.

## Markers

Los tests están organizados con markers:

- `@pytest.mark.unit`: Tests unitarios
- `@pytest.mark.integration`: Tests de integración
- `@pytest.mark.slow`: Tests que toman tiempo
- `@pytest.mark.smoke`: Tests de smoke testing

## Mejores Prácticas

1. **Un concepto por test**: Cada test debe verificar un solo comportamiento
2. **Nombres descriptivos**: Los nombres de test deben describir qué se está probando
3. **AAA Pattern**: Arrange, Act, Assert
4. **Tests independientes**: Los tests no deben depender unos de otros
5. **Fixtures para setup**: Usar fixtures para configuración compartida
6. **Docstrings**: Cada test debe tener un docstring explicativo

## CI/CD

Los tests se ejecutan automáticamente en el pipeline de CI/CD en cada commit y pull request.

## Recursos

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
