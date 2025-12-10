# domain/model/entities/progress.py
"""
Tipos y entidades para el sistema de callbacks de progreso.

Este modulo define las interfaces para reportar progreso durante:
- Carga de modelos (descarga, inicializacion)
- Ejecucion del pipeline (paso a paso)
- Procesamiento de entradas multiples
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol


class ProgressPhase(Enum):
    """Fases del proceso de ejecucion."""
    # Fases de carga de modelo
    MODEL_DOWNLOAD = "model_download"
    MODEL_LOADING = "model_loading"
    MODEL_READY = "model_ready"

    # Fases del pipeline
    PIPELINE_START = "pipeline_start"
    PIPELINE_STEP = "pipeline_step"
    PIPELINE_COMPLETE = "pipeline_complete"

    # Fases de procesamiento multiple
    ENTRY_START = "entry_start"
    ENTRY_COMPLETE = "entry_complete"


@dataclass(frozen=True)
class ProgressUpdate:
    """
    Representa una actualizacion de progreso.

    Attributes:
        phase: Fase actual del proceso
        current: Valor actual (ej: paso actual, bytes descargados)
        total: Valor total (ej: total de pasos, bytes totales)
        message: Mensaje descriptivo para mostrar al usuario
        percentage: Porcentaje calculado (0-100), None si no aplica
        details: Informacion adicional opcional (ej: nombre del paso)
    """
    phase: ProgressPhase
    current: int
    total: int
    message: str
    percentage: Optional[float] = None
    details: Optional[str] = None

    def __post_init__(self):
        # Calcular porcentaje si no se proporciono
        if self.percentage is None and self.total > 0:
            object.__setattr__(self, 'percentage', (self.current / self.total) * 100)


# Tipo para callbacks de progreso
ProgressCallback = Callable[[ProgressUpdate], None]


class ProgressReporter(Protocol):
    """
    Protocolo para reportadores de progreso.

    Permite diferentes implementaciones (GUI, CLI, logging, etc.)
    """

    def on_progress(self, update: ProgressUpdate) -> None:
        """Llamado cuando hay una actualizacion de progreso."""
        ...

    def on_model_loading(self, model_name: str, phase: str, progress: float) -> None:
        """Llamado durante la carga del modelo."""
        ...

    def on_pipeline_step(self, step_number: int, total_steps: int, step_type: str) -> None:
        """Llamado al iniciar un paso del pipeline."""
        ...

    def on_entry_processing(self, entry_index: int, total_entries: int) -> None:
        """Llamado al procesar una entrada en modo multiple."""
        ...


class NullProgressReporter:
    """
    Implementacion nula del reporter de progreso.

    Usado cuando no se necesita reportar progreso (ej: CLI sin output).
    """

    def on_progress(self, update: ProgressUpdate) -> None:
        pass

    def on_model_loading(self, model_name: str, phase: str, progress: float) -> None:
        pass

    def on_pipeline_step(self, step_number: int, total_steps: int, step_type: str) -> None:
        pass

    def on_entry_processing(self, entry_index: int, total_entries: int) -> None:
        pass


@dataclass
class SimpleProgressReporter:
    """
    Implementacion simple que usa un callback.

    Convierte todas las llamadas en ProgressUpdate y las envia al callback.
    """
    callback: Optional[ProgressCallback] = None

    def on_progress(self, update: ProgressUpdate) -> None:
        if self.callback:
            self.callback(update)

    def on_model_loading(self, model_name: str, phase: str, progress: float) -> None:
        if self.callback:
            update = ProgressUpdate(
                phase=ProgressPhase.MODEL_LOADING,
                current=int(progress),
                total=100,
                message=f"Cargando modelo: {phase}",
                percentage=progress,
                details=model_name
            )
            self.callback(update)

    def on_pipeline_step(self, step_number: int, total_steps: int, step_type: str) -> None:
        if self.callback:
            update = ProgressUpdate(
                phase=ProgressPhase.PIPELINE_STEP,
                current=step_number + 1,
                total=total_steps,
                message=f"Ejecutando paso {step_number + 1}/{total_steps}",
                details=step_type
            )
            self.callback(update)

    def on_entry_processing(self, entry_index: int, total_entries: int) -> None:
        if self.callback:
            update = ProgressUpdate(
                phase=ProgressPhase.ENTRY_START,
                current=entry_index + 1,
                total=total_entries,
                message=f"Procesando entrada {entry_index + 1}/{total_entries}"
            )
            self.callback(update)
