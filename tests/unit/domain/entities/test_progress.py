# tests/unit/domain/entities/test_progress.py
"""Tests para el modulo de progreso."""

import pytest
from domain.model.entities.progress import (
    ProgressPhase,
    ProgressUpdate,
    NullProgressReporter,
    SimpleProgressReporter
)


class TestProgressPhase:
    """Tests para ProgressPhase enum."""

    def test_model_phases_exist(self):
        """Verifica que existen las fases de modelo."""
        assert ProgressPhase.MODEL_DOWNLOAD.value == "model_download"
        assert ProgressPhase.MODEL_LOADING.value == "model_loading"
        assert ProgressPhase.MODEL_READY.value == "model_ready"

    def test_pipeline_phases_exist(self):
        """Verifica que existen las fases de pipeline."""
        assert ProgressPhase.PIPELINE_START.value == "pipeline_start"
        assert ProgressPhase.PIPELINE_STEP.value == "pipeline_step"
        assert ProgressPhase.PIPELINE_COMPLETE.value == "pipeline_complete"

    def test_entry_phases_exist(self):
        """Verifica que existen las fases de entrada."""
        assert ProgressPhase.ENTRY_START.value == "entry_start"
        assert ProgressPhase.ENTRY_COMPLETE.value == "entry_complete"


class TestProgressUpdate:
    """Tests para ProgressUpdate dataclass."""

    def test_create_basic_update(self):
        """Crea un update basico con campos requeridos."""
        update = ProgressUpdate(
            phase=ProgressPhase.MODEL_LOADING,
            current=1,
            total=3,
            message="Cargando modelo..."
        )

        assert update.phase == ProgressPhase.MODEL_LOADING
        assert update.current == 1
        assert update.total == 3
        assert update.message == "Cargando modelo..."
        assert update.details is None

    def test_create_update_with_details(self):
        """Crea un update con detalles opcionales."""
        update = ProgressUpdate(
            phase=ProgressPhase.PIPELINE_STEP,
            current=2,
            total=5,
            message="Ejecutando paso 2",
            details="generate"
        )

        assert update.details == "generate"

    def test_percentage_auto_calculated(self):
        """Verifica que el porcentaje se calcula automaticamente."""
        update = ProgressUpdate(
            phase=ProgressPhase.PIPELINE_STEP,
            current=1,
            total=4,
            message="Test"
        )

        assert update.percentage == 25.0

    def test_percentage_100_when_complete(self):
        """Verifica que el porcentaje es 100 cuando esta completo."""
        update = ProgressUpdate(
            phase=ProgressPhase.PIPELINE_COMPLETE,
            current=5,
            total=5,
            message="Completado"
        )

        assert update.percentage == 100.0

    def test_percentage_explicit_override(self):
        """Verifica que se puede establecer un porcentaje explicito."""
        update = ProgressUpdate(
            phase=ProgressPhase.MODEL_DOWNLOAD,
            current=0,
            total=0,  # Total 0 normalmente causaria division por cero
            message="Test",
            percentage=50.0
        )

        assert update.percentage == 50.0

    def test_percentage_none_with_zero_total(self):
        """Verifica que porcentaje queda None si total es 0 y no se especifica."""
        # Con percentage=None y total=0, post_init no puede calcular
        # Pero lo inicializamos explicitamente a None para evitar calculo
        update = ProgressUpdate(
            phase=ProgressPhase.MODEL_DOWNLOAD,
            current=0,
            total=0,
            message="Test"
        )
        # Con total=0, el calculo da division por cero, asi que el resultado es None
        # Pero el codigo actual usa percentage=None como default y si total>0 lo calcula
        # Si total=0, no puede calcular, entonces queda None
        assert update.percentage is None

    def test_update_is_frozen(self):
        """Verifica que el update es inmutable."""
        update = ProgressUpdate(
            phase=ProgressPhase.MODEL_LOADING,
            current=1,
            total=3,
            message="Test"
        )

        with pytest.raises(AttributeError):
            update.current = 2


class TestNullProgressReporter:
    """Tests para NullProgressReporter."""

    def test_on_progress_does_nothing(self):
        """Verifica que on_progress no hace nada."""
        reporter = NullProgressReporter()
        update = ProgressUpdate(
            phase=ProgressPhase.MODEL_LOADING,
            current=1,
            total=3,
            message="Test"
        )

        # No deberia lanzar excepcion
        reporter.on_progress(update)

    def test_on_model_loading_does_nothing(self):
        """Verifica que on_model_loading no hace nada."""
        reporter = NullProgressReporter()
        reporter.on_model_loading("test-model", "downloading", 50.0)

    def test_on_pipeline_step_does_nothing(self):
        """Verifica que on_pipeline_step no hace nada."""
        reporter = NullProgressReporter()
        reporter.on_pipeline_step(1, 5, "generate")

    def test_on_entry_processing_does_nothing(self):
        """Verifica que on_entry_processing no hace nada."""
        reporter = NullProgressReporter()
        reporter.on_entry_processing(0, 10)


class TestSimpleProgressReporter:
    """Tests para SimpleProgressReporter."""

    def test_on_progress_calls_callback(self):
        """Verifica que on_progress llama al callback."""
        received_updates = []

        def callback(update):
            received_updates.append(update)

        reporter = SimpleProgressReporter(callback=callback)
        update = ProgressUpdate(
            phase=ProgressPhase.MODEL_LOADING,
            current=1,
            total=3,
            message="Test"
        )

        reporter.on_progress(update)

        assert len(received_updates) == 1
        assert received_updates[0] == update

    def test_on_progress_no_callback(self):
        """Verifica que on_progress no falla sin callback."""
        reporter = SimpleProgressReporter(callback=None)
        update = ProgressUpdate(
            phase=ProgressPhase.MODEL_LOADING,
            current=1,
            total=3,
            message="Test"
        )

        # No deberia lanzar excepcion
        reporter.on_progress(update)

    def test_on_model_loading_creates_update(self):
        """Verifica que on_model_loading crea un ProgressUpdate correcto."""
        received_updates = []

        def callback(update):
            received_updates.append(update)

        reporter = SimpleProgressReporter(callback=callback)
        reporter.on_model_loading("Qwen/Qwen2.5-0.5B", "downloading", 75.0)

        assert len(received_updates) == 1
        update = received_updates[0]
        assert update.phase == ProgressPhase.MODEL_LOADING
        assert update.percentage == 75.0
        assert "Cargando modelo" in update.message
        assert update.details == "Qwen/Qwen2.5-0.5B"

    def test_on_pipeline_step_creates_update(self):
        """Verifica que on_pipeline_step crea un ProgressUpdate correcto."""
        received_updates = []

        def callback(update):
            received_updates.append(update)

        reporter = SimpleProgressReporter(callback=callback)
        reporter.on_pipeline_step(2, 5, "generate")

        assert len(received_updates) == 1
        update = received_updates[0]
        assert update.phase == ProgressPhase.PIPELINE_STEP
        assert update.current == 3  # step_number + 1
        assert update.total == 5
        assert "3/5" in update.message
        assert update.details == "generate"

    def test_on_entry_processing_creates_update(self):
        """Verifica que on_entry_processing crea un ProgressUpdate correcto."""
        received_updates = []

        def callback(update):
            received_updates.append(update)

        reporter = SimpleProgressReporter(callback=callback)
        reporter.on_entry_processing(3, 10)

        assert len(received_updates) == 1
        update = received_updates[0]
        assert update.phase == ProgressPhase.ENTRY_START
        assert update.current == 4  # entry_index + 1
        assert update.total == 10
        assert "4/10" in update.message
