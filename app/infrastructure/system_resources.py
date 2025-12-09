# infrastructure/system_resources.py
"""
Sistema de deteccion y validacion de recursos del sistema.

Este modulo implementa la filosofia central de EsencIA: ejecucion local con GPUs "normalitas".
Permite detectar los recursos disponibles y validar si un modelo puede ejecutarse
antes de intentar cargarlo, evitando errores de memoria costosos.

Filosofia:
    EsencIA esta disenado para ejecutarse en hardware de consumo (GPUs de 4-16GB VRAM).
    Los modelos recomendados son aquellos que pueden cargarse completamente en la
    VRAM disponible o en RAM si no hay GPU. La aplicacion informa proactivamente
    sobre la viabilidad de ejecutar un modelo antes de intentar cargarlo.
"""

import logging
import platform
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Tipos de dispositivo de computo disponibles."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon


class ModelSize(Enum):
    """Categorias de tamano de modelos por parametros."""
    TINY = "tiny"           # < 500M params (~1GB)
    SMALL = "small"         # 500M - 1.5B params (~3GB)
    MEDIUM = "medium"       # 1.5B - 3B params (~6GB)
    LARGE = "large"         # 3B - 7B params (~14GB)
    XLARGE = "xlarge"       # 7B - 13B params (~26GB)
    XXLARGE = "xxlarge"     # > 13B params (>26GB)


@dataclass
class GPUInfo:
    """Informacion detallada de una GPU."""
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    compute_capability: Optional[Tuple[int, int]] = None

    @property
    def utilization_percent(self) -> float:
        """Porcentaje de memoria utilizada."""
        if self.total_memory_gb == 0:
            return 0.0
        return (self.used_memory_gb / self.total_memory_gb) * 100


@dataclass
class SystemResources:
    """
    Recursos del sistema disponibles para ejecucion de modelos.

    Attributes:
        device_type: Tipo de dispositivo principal (CUDA, CPU, MPS)
        gpus: Lista de GPUs disponibles
        total_ram_gb: RAM total del sistema
        available_ram_gb: RAM disponible actualmente
        cpu_cores: Numero de nucleos de CPU
        platform_info: Informacion del sistema operativo
    """
    device_type: DeviceType
    gpus: List[GPUInfo] = field(default_factory=list)
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    cpu_cores: int = 1
    platform_info: str = ""

    @property
    def has_gpu(self) -> bool:
        """Indica si hay GPU disponible (CUDA o MPS)."""
        if self.device_type == DeviceType.MPS:
            return True  # Apple Silicon siempre tiene GPU integrada
        return len(self.gpus) > 0 and self.device_type == DeviceType.CUDA

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        """Retorna la GPU principal (con mas VRAM libre)."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.free_memory_gb)

    @property
    def total_vram_gb(self) -> float:
        """VRAM total disponible sumando todas las GPUs."""
        return sum(gpu.total_memory_gb for gpu in self.gpus)

    @property
    def available_vram_gb(self) -> float:
        """VRAM libre de la GPU principal."""
        if self.primary_gpu:
            return self.primary_gpu.free_memory_gb
        return 0.0


@dataclass
class ModelRequirements:
    """
    Requisitos estimados de un modelo.

    Attributes:
        model_name: Nombre del modelo
        estimated_vram_gb: VRAM estimada necesaria
        estimated_ram_gb: RAM estimada necesaria (modo CPU)
        parameters_billions: Parametros del modelo en miles de millones
        size_category: Categoria de tamano
        recommended_for_local: Si se recomienda para ejecucion local
        notes: Notas adicionales sobre el modelo
    """
    model_name: str
    estimated_vram_gb: float
    estimated_ram_gb: float
    parameters_billions: float
    size_category: ModelSize
    recommended_for_local: bool = True
    notes: str = ""


@dataclass
class CompatibilityResult:
    """
    Resultado de la validacion de compatibilidad.

    Attributes:
        is_compatible: Si el modelo puede ejecutarse
        can_use_gpu: Si puede usar GPU
        recommended_device: Dispositivo recomendado
        warnings: Lista de advertencias
        error_message: Mensaje de error si no es compatible
        estimated_load_time: Tiempo estimado de carga
    """
    is_compatible: bool
    can_use_gpu: bool
    recommended_device: DeviceType
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    estimated_load_time: str = ""


class SystemResourceDetector:
    """
    Detecta los recursos disponibles del sistema.

    Esta clase es el punto de entrada para obtener informacion sobre
    el hardware disponible para ejecutar modelos de lenguaje.

    Example:
        >>> detector = SystemResourceDetector()
        >>> resources = detector.detect()
        >>> print(f"GPU disponible: {resources.has_gpu}")
        >>> print(f"VRAM: {resources.available_vram_gb:.1f} GB")
    """

    def __init__(self):
        self._torch_available = False
        self._torch_cuda_available = False
        self._pynvml_available = False
        self._pynvml_module = None  # Referencia al módulo nvml disponible
        self._psutil_available = False
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Verifica disponibilidad de dependencias opcionales."""
        try:
            import torch
            self._torch_available = True
            self._torch_cuda_available = torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch no disponible - deteccion de GPU limitada")

        # Verificar pynvml/nvidia-ml-py para detección directa de NVIDIA
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            self._pynvml_available = True
            self._pynvml_module = pynvml
        except (ImportError, Exception):
            try:
                # Alternativa: py3nvml
                from py3nvml import py3nvml
                py3nvml.nvmlInit()
                py3nvml.nvmlShutdown()
                self._pynvml_available = True
                self._pynvml_module = py3nvml
            except (ImportError, Exception):
                logger.debug("pynvml/py3nvml no disponible")

        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            logger.debug("psutil no disponible - deteccion de RAM limitada")

    def detect(self) -> SystemResources:
        """
        Detecta todos los recursos del sistema.

        Returns:
            SystemResources: Informacion completa del sistema
        """
        device_type = self._detect_device_type()
        gpus = self._detect_gpus() if device_type == DeviceType.CUDA else []
        ram_total, ram_available = self._detect_ram()
        cpu_cores = self._detect_cpu_cores()
        platform_info = self._detect_platform()

        resources = SystemResources(
            device_type=device_type,
            gpus=gpus,
            total_ram_gb=ram_total,
            available_ram_gb=ram_available,
            cpu_cores=cpu_cores,
            platform_info=platform_info
        )

        logger.info(
            f"Sistema detectado: {device_type.value.upper()}, "
            f"{'VRAM: ' + f'{resources.available_vram_gb:.1f}GB' if resources.has_gpu else 'Sin GPU'}, "
            f"RAM: {ram_available:.1f}/{ram_total:.1f}GB"
        )

        return resources

    def _detect_device_type(self) -> DeviceType:
        """Detecta el tipo de dispositivo de computo disponible."""
        # Primero verificar con PyTorch CUDA
        if self._torch_cuda_available:
            return DeviceType.CUDA

        # Si PyTorch no tiene CUDA, intentar detectar GPU NVIDIA via pynvml
        if self._pynvml_available and self._pynvml_module:
            try:
                nvml = self._pynvml_module
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                nvml.nvmlShutdown()
                if device_count > 0:
                    return DeviceType.CUDA
            except Exception:
                pass

        # Verificar MPS (Apple Silicon)
        if self._torch_available:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return DeviceType.MPS

        return DeviceType.CPU

    def _detect_gpus(self) -> List[GPUInfo]:
        """Detecta GPUs CUDA/NVIDIA disponibles."""
        gpus = []

        # Método 1: Usar pynvml directamente (más preciso y no requiere PyTorch CUDA)
        if self._pynvml_available and self._pynvml_module:
            try:
                nvml = self._pynvml_module
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')

                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                    # Obtener compute capability
                    try:
                        major = nvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                        minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                        compute_cap = (major, minor)
                    except Exception:
                        compute_cap = None

                    gpus.append(GPUInfo(
                        index=i,
                        name=name,
                        total_memory_gb=mem_info.total / (1024**3),
                        free_memory_gb=mem_info.free / (1024**3),
                        used_memory_gb=mem_info.used / (1024**3),
                        compute_capability=compute_cap
                    ))

                nvml.nvmlShutdown()
                return gpus

            except Exception as e:
                logger.warning(f"Error detectando GPUs con pynvml: {e}")

        # Método 2: Fallback a PyTorch CUDA (si está disponible)
        if self._torch_cuda_available:
            import torch

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)

                total_mem = props.total_memory / (1024**3)
                # memory_reserved funciona sin necesidad de set_device
                reserved = torch.cuda.memory_reserved(i)
                free_mem = (props.total_memory - reserved) / (1024**3)
                used_mem = reserved / (1024**3)

                gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_gb=total_mem,
                    free_memory_gb=free_mem,
                    used_memory_gb=used_mem,
                    compute_capability=(props.major, props.minor)
                ))

        return gpus

    def _detect_ram(self) -> Tuple[float, float]:
        """Detecta RAM total y disponible."""
        if self._psutil_available:
            import psutil
            mem = psutil.virtual_memory()
            return (mem.total / (1024**3), mem.available / (1024**3))

        # Fallbacks sin psutil
        system = platform.system()

        # Windows: usar ctypes
        if system == "Windows":
            try:
                import ctypes
                from ctypes import wintypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64),
                    ]

                meminfo = MEMORYSTATUSEX()
                meminfo.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(meminfo))

                total = meminfo.ullTotalPhys / (1024**3)
                available = meminfo.ullAvailPhys / (1024**3)
                return (total, available)
            except Exception as e:
                logger.warning(f"Error detectando RAM en Windows: {e}")

        # Linux: leer /proc/meminfo
        elif system == "Linux":
            try:
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    total = int([l for l in lines if 'MemTotal' in l][0].split()[1]) / (1024**2)
                    available = int([l for l in lines if 'MemAvailable' in l][0].split()[1]) / (1024**2)
                    return (total, available)
            except Exception as e:
                logger.warning(f"Error detectando RAM en Linux: {e}")

        # macOS: usar sysctl
        elif system == "Darwin":
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
                total = int(result.stdout.strip()) / (1024**3)
                # En macOS es difícil obtener la RAM disponible sin psutil
                # Estimamos 50% disponible
                return (total, total * 0.5)
            except Exception as e:
                logger.warning(f"Error detectando RAM en macOS: {e}")

        # Fallback final: valores conservadores
        logger.warning("No se pudo detectar RAM, usando valores por defecto")
        return (16.0, 8.0)

    def _detect_cpu_cores(self) -> int:
        """Detecta numero de nucleos de CPU."""
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1

    def _detect_platform(self) -> str:
        """Detecta informacion del sistema operativo."""
        return f"{platform.system()} {platform.release()} ({platform.machine()})"


class ModelRequirementsEstimator:
    """
    Estima los requisitos de hardware para modelos de lenguaje.

    Utiliza heuristicas basadas en el nombre del modelo y patrones conocidos
    para estimar la VRAM/RAM necesaria sin necesidad de descargar el modelo.

    La regla general es:
    - Modelo en FP32: ~4 bytes por parametro
    - Modelo en FP16: ~2 bytes por parametro
    - Modelo cuantizado (INT8): ~1 byte por parametro
    - Modelo cuantizado (INT4): ~0.5 bytes por parametro

    Ademas, se necesita memoria adicional para:
    - KV Cache durante inferencia (~10-20% extra)
    - Buffers y overhead del framework (~1-2GB)
    """

    # Patrones conocidos de modelos y sus parametros aproximados
    KNOWN_MODELS = {
        # Qwen
        "qwen2.5-0.5b": 0.5,
        "qwen2.5-1.5b": 1.5,
        "qwen2.5-3b": 3.0,
        "qwen2.5-7b": 7.0,
        "qwen2.5-14b": 14.0,
        "qwen2.5-32b": 32.0,
        "qwen2.5-72b": 72.0,
        # Llama
        "llama-2-7b": 7.0,
        "llama-2-13b": 13.0,
        "llama-2-70b": 70.0,
        "llama-3-8b": 8.0,
        "llama-3-70b": 70.0,
        # Mistral
        "mistral-7b": 7.0,
        "mixtral-8x7b": 47.0,  # MoE, ~47B total pero ~13B activos
        # Phi
        "phi-2": 2.7,
        "phi-3-mini": 3.8,
        # GPT-2
        "gpt2": 0.12,
        "gpt2-medium": 0.35,
        "gpt2-large": 0.77,
        "gpt2-xl": 1.5,
        # Bloom
        "bloom-560m": 0.56,
        "bloom-1b": 1.0,
        "bloom-3b": 3.0,
        "bloom-7b": 7.0,
        # TinyLlama
        "tinyllama": 1.1,
    }

    # Multiplicador de VRAM segun precision
    PRECISION_MULTIPLIERS = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
        "awq": 0.5,
        "gptq": 0.5,
        "gguf": 0.6,  # Depende de la cuantizacion especifica
    }

    # Overhead adicional (KV cache, buffers, etc.)
    BASE_OVERHEAD_GB = 1.5
    KV_CACHE_MULTIPLIER = 1.15  # 15% extra para KV cache

    def estimate(self, model_name: str) -> ModelRequirements:
        """
        Estima los requisitos de un modelo.

        Args:
            model_name: Nombre del modelo (ej: "Qwen/Qwen2.5-1.5B-Instruct")

        Returns:
            ModelRequirements: Requisitos estimados
        """
        model_lower = model_name.lower()

        # Detectar parametros
        params_b = self._estimate_parameters(model_lower)

        # Detectar precision
        precision = self._detect_precision(model_lower)
        multiplier = self.PRECISION_MULTIPLIERS.get(precision, 2.0)  # Default FP16

        # Calcular memoria
        base_memory = params_b * multiplier
        vram_needed = (base_memory * self.KV_CACHE_MULTIPLIER) + self.BASE_OVERHEAD_GB
        ram_needed = vram_needed * 1.2  # CPU necesita ~20% mas por overhead

        # Determinar categoria
        size_category = self._categorize_size(params_b)

        # Determinar si es recomendado para local
        recommended = params_b <= 7.0 and vram_needed <= 16.0

        # Generar notas
        notes = self._generate_notes(params_b, precision, vram_needed)

        return ModelRequirements(
            model_name=model_name,
            estimated_vram_gb=round(vram_needed, 1),
            estimated_ram_gb=round(ram_needed, 1),
            parameters_billions=params_b,
            size_category=size_category,
            recommended_for_local=recommended,
            notes=notes
        )

    def _estimate_parameters(self, model_name: str) -> float:
        """Estima el numero de parametros en miles de millones."""
        # Buscar en modelos conocidos
        for pattern, params in self.KNOWN_MODELS.items():
            if pattern in model_name:
                return params

        # Intentar extraer de patrones comunes en el nombre
        import re

        # Patron: XXB o XX.XB (ej: 7b, 1.5b, 70b)
        match = re.search(r'(\d+\.?\d*)[bB]', model_name)
        if match:
            return float(match.group(1))

        # Patron: XXm o XXM (millones)
        match = re.search(r'(\d+)[mM]', model_name)
        if match:
            return float(match.group(1)) / 1000

        # Patron: small/medium/large/xl en el nombre
        if 'xxl' in model_name or 'xxxl' in model_name:
            return 11.0
        elif 'xl' in model_name:
            return 3.0
        elif 'large' in model_name:
            return 1.5
        elif 'medium' in model_name:
            return 0.5
        elif 'small' in model_name or 'mini' in model_name:
            return 0.3
        elif 'tiny' in model_name or 'nano' in model_name:
            return 0.1

        # Default conservador
        logger.warning(f"No se pudo estimar parametros para '{model_name}', usando 3B por defecto")
        return 3.0

    def _detect_precision(self, model_name: str) -> str:
        """Detecta la precision del modelo por su nombre."""
        if 'awq' in model_name:
            return 'awq'
        elif 'gptq' in model_name:
            return 'gptq'
        elif 'gguf' in model_name:
            return 'gguf'
        elif 'int8' in model_name or '8bit' in model_name:
            return 'int8'
        elif 'int4' in model_name or '4bit' in model_name:
            return 'int4'
        elif 'fp32' in model_name:
            return 'fp32'
        elif 'bf16' in model_name:
            return 'bf16'
        # Default: asumir FP16 (lo mas comun en HuggingFace)
        return 'fp16'

    def _categorize_size(self, params_b: float) -> ModelSize:
        """Categoriza el tamano del modelo."""
        if params_b < 0.5:
            return ModelSize.TINY
        elif params_b < 1.5:
            return ModelSize.SMALL
        elif params_b < 3:
            return ModelSize.MEDIUM
        elif params_b < 7:
            return ModelSize.LARGE
        elif params_b < 13:
            return ModelSize.XLARGE
        else:
            return ModelSize.XXLARGE

    def _generate_notes(self, params_b: float, precision: str, vram_gb: float) -> str:
        """Genera notas informativas sobre el modelo."""
        notes = []

        if params_b <= 1.5:
            notes.append("Ideal para GPUs de gama baja (4GB VRAM)")
        elif params_b <= 3:
            notes.append("Requiere GPU de gama media (6-8GB VRAM)")
        elif params_b <= 7:
            notes.append("Requiere GPU de gama alta (10-16GB VRAM)")
        else:
            notes.append("Requiere GPU profesional o multiples GPUs")

        if precision in ['awq', 'gptq', 'int4', 'int8']:
            notes.append(f"Modelo cuantizado ({precision.upper()}) - menor precision pero mas eficiente")

        if vram_gb > 24:
            notes.append("ADVERTENCIA: Probablemente no ejecutable en hardware de consumo")

        return ". ".join(notes)


class ModelCompatibilityChecker:
    """
    Valida la compatibilidad entre un modelo y los recursos disponibles.

    Esta clase es el nucleo de la filosofia de EsencIA: verificar ANTES
    de intentar cargar un modelo si es viable ejecutarlo con los recursos
    disponibles, proporcionando feedback claro al usuario.

    Example:
        >>> checker = ModelCompatibilityChecker()
        >>> result = checker.check("Qwen/Qwen2.5-1.5B-Instruct")
        >>> if result.is_compatible:
        ...     print(f"Modelo compatible, usar {result.recommended_device}")
        ... else:
        ...     print(f"Error: {result.error_message}")
    """

    # Margenes de seguridad
    VRAM_SAFETY_MARGIN = 0.9  # Usar max 90% de VRAM disponible
    RAM_SAFETY_MARGIN = 0.7   # Usar max 70% de RAM disponible

    def __init__(self):
        self.resource_detector = SystemResourceDetector()
        self.requirements_estimator = ModelRequirementsEstimator()
        self._cached_resources: Optional[SystemResources] = None

    def get_system_resources(self, refresh: bool = False) -> SystemResources:
        """
        Obtiene los recursos del sistema (con cache).

        Args:
            refresh: Forzar re-deteccion de recursos

        Returns:
            SystemResources: Recursos del sistema
        """
        if self._cached_resources is None or refresh:
            self._cached_resources = self.resource_detector.detect()
        return self._cached_resources

    def check(self, model_name: str, force_cpu: bool = False) -> CompatibilityResult:
        """
        Verifica si un modelo puede ejecutarse con los recursos disponibles.

        Args:
            model_name: Nombre del modelo a verificar
            force_cpu: Forzar uso de CPU aunque haya GPU

        Returns:
            CompatibilityResult: Resultado detallado de la compatibilidad
        """
        resources = self.get_system_resources()
        requirements = self.requirements_estimator.estimate(model_name)

        warnings = []
        error_message = None

        # Log de diagnostico
        logger.info(
            f"Verificando compatibilidad: {model_name} "
            f"(~{requirements.parameters_billions}B params, "
            f"~{requirements.estimated_vram_gb}GB VRAM)"
        )

        # Determinar si puede usar GPU
        can_use_gpu = False
        if resources.has_gpu and not force_cpu:
            available_vram = resources.available_vram_gb * self.VRAM_SAFETY_MARGIN
            if requirements.estimated_vram_gb <= available_vram:
                can_use_gpu = True
            else:
                warnings.append(
                    f"VRAM insuficiente: necesita ~{requirements.estimated_vram_gb}GB, "
                    f"disponible ~{resources.available_vram_gb:.1f}GB"
                )

        # Verificar CPU como fallback
        available_ram = resources.available_ram_gb * self.RAM_SAFETY_MARGIN
        can_use_cpu = requirements.estimated_ram_gb <= available_ram

        # Determinar compatibilidad y dispositivo recomendado
        is_compatible = can_use_gpu or can_use_cpu

        if can_use_gpu:
            recommended_device = DeviceType.CUDA
            estimated_time = self._estimate_load_time(requirements, DeviceType.CUDA)
        elif can_use_cpu:
            recommended_device = DeviceType.CPU
            estimated_time = self._estimate_load_time(requirements, DeviceType.CPU)
            warnings.append(
                "Se usara CPU - la inferencia sera significativamente mas lenta"
            )
        else:
            recommended_device = DeviceType.CPU
            estimated_time = "N/A"
            is_compatible = False
            error_message = (
                f"Recursos insuficientes para el modelo '{model_name}'.\n"
                f"Requisitos estimados: {requirements.estimated_vram_gb}GB VRAM / "
                f"{requirements.estimated_ram_gb}GB RAM\n"
                f"Disponible: {resources.available_vram_gb:.1f}GB VRAM / "
                f"{resources.available_ram_gb:.1f}GB RAM\n"
                f"Sugerencia: Use un modelo mas pequeno como 'Qwen/Qwen2.5-0.5B-Instruct' "
                f"o 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'"
            )

        # Agregar advertencias adicionales
        if not requirements.recommended_for_local:
            warnings.append(
                f"Este modelo ({requirements.parameters_billions}B params) no es ideal "
                f"para ejecucion local. Considere modelos mas pequenos."
            )

        if requirements.notes:
            warnings.append(requirements.notes)

        result = CompatibilityResult(
            is_compatible=is_compatible,
            can_use_gpu=can_use_gpu,
            recommended_device=recommended_device,
            warnings=warnings,
            error_message=error_message,
            estimated_load_time=estimated_time
        )

        if is_compatible:
            logger.info(
                f"Modelo compatible: {recommended_device.value.upper()}, "
                f"tiempo estimado: {estimated_time}"
            )
        else:
            logger.warning(f"Modelo no compatible: {error_message}")

        return result

    def _estimate_load_time(self, requirements: ModelRequirements, device: DeviceType) -> str:
        """Estima el tiempo de carga del modelo."""
        # Heuristica simple basada en tamano y dispositivo
        base_seconds = requirements.parameters_billions * 2  # ~2s por B de params

        if device == DeviceType.CPU:
            base_seconds *= 3  # CPU es ~3x mas lento en carga

        if base_seconds < 10:
            return "< 10 segundos"
        elif base_seconds < 60:
            return f"~{int(base_seconds)} segundos"
        elif base_seconds < 300:
            return f"~{int(base_seconds / 60)} minutos"
        else:
            return f"~{int(base_seconds / 60)} minutos (puede variar)"

    def get_recommended_models(self) -> List[dict]:
        """
        Retorna una lista de modelos recomendados para los recursos actuales.

        Returns:
            Lista de diccionarios con informacion de modelos recomendados
        """
        resources = self.get_system_resources()

        # Modelos candidatos ordenados por tamano
        candidates = [
            ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B params - Muy rapido, ideal para pruebas"),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B params - Buen balance velocidad/calidad"),
            ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B params - Recomendado para uso general"),
            ("microsoft/phi-2", "2.7B params - Alta calidad para su tamano"),
            ("Qwen/Qwen2.5-3B-Instruct", "3B params - Mejor calidad, requiere mas recursos"),
            ("mistralai/Mistral-7B-Instruct-v0.2", "7B params - Alta calidad, requiere GPU potente"),
        ]

        recommended = []
        for model_name, description in candidates:
            result = self.check(model_name)
            if result.is_compatible:
                requirements = self.requirements_estimator.estimate(model_name)
                recommended.append({
                    "model_name": model_name,
                    "description": description,
                    "vram_needed": requirements.estimated_vram_gb,
                    "device": result.recommended_device.value,
                    "load_time": result.estimated_load_time
                })

        return recommended


# Funciones de conveniencia para uso rapido
def check_model_compatibility(model_name: str) -> CompatibilityResult:
    """
    Funcion de conveniencia para verificar compatibilidad de un modelo.

    Args:
        model_name: Nombre del modelo

    Returns:
        CompatibilityResult: Resultado de la verificacion
    """
    checker = ModelCompatibilityChecker()
    return checker.check(model_name)


def get_system_info() -> SystemResources:
    """
    Funcion de conveniencia para obtener informacion del sistema.

    Returns:
        SystemResources: Recursos del sistema
    """
    detector = SystemResourceDetector()
    return detector.detect()


def get_model_requirements(model_name: str) -> ModelRequirements:
    """
    Funcion de conveniencia para estimar requisitos de un modelo.

    Args:
        model_name: Nombre del modelo

    Returns:
        ModelRequirements: Requisitos estimados
    """
    estimator = ModelRequirementsEstimator()
    return estimator.estimate(model_name)
