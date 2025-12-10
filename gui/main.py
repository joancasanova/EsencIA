"""
EsencIA - GUI Profesional
Interfaz para procesamiento de textos con LLMs.
Arquitectura: NiceGUI + Backend asíncrono
"""

import sys
import json
import platform
import asyncio
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import OrderedDict

import psutil
import httpx

# Lazy loading para torch (es muy pesado y ralentiza el arranque)
_torch = None
_torch_checked = False

def _get_torch():
    """Carga torch de forma diferida solo cuando se necesita."""
    global _torch, _torch_checked
    if not _torch_checked:
        _torch_checked = True
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
    return _torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

GUI_DIR = Path(__file__).parent
APP_DIR = GUI_DIR.parent / "app"
sys.path.insert(0, str(APP_DIR))

from nicegui import ui, run, app

# Imports ligeros (configuración y entidades - no importan torch)
from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_PIPELINE_REFERENCE_DATA,
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_BENCHMARK_ENTRIES,
)

# Lazy loading para use cases pesados (importan torch/transformers)
_pipeline_use_case = None
_benchmark_use_case = None

def _get_pipeline_use_case():
    """Carga PipelineUseCase de forma diferida."""
    global _pipeline_use_case
    if _pipeline_use_case is None:
        from application.use_cases.pipeline_use_case import PipelineUseCase
        _pipeline_use_case = PipelineUseCase
    return _pipeline_use_case

def _get_benchmark_use_case():
    """Carga BenchmarkUseCase de forma diferida."""
    global _benchmark_use_case
    if _benchmark_use_case is None:
        from application.use_cases.benchmark_use_case import BenchmarkUseCase
        _benchmark_use_case = BenchmarkUseCase
    return _benchmark_use_case

# Imports de entidades (ligeros - directamente desde domain, no importan torch)
from domain.model.entities.generation import GenerateTextRequest
from domain.model.entities.parsing import ParseMode, ParseRule, ParseRequest
from domain.model.entities.verification import VerificationMethod, VerificationMode, VerifyRequest
from domain.model.entities.pipeline import PipelineStep, PipelineRequest
from domain.model.entities.benchmark import BenchmarkConfig, BenchmarkEntry
from domain.model.entities.progress import ProgressUpdate, ProgressPhase


# =============================================================================
# GLOBAL STATE
# =============================================================================
class AppState:
    """Estado global de la aplicación."""
    def __init__(self):
        self.model = ''  # Vacío para mostrar placeholder
        self.model_info: Optional[Dict] = None  # Info del modelo (compat_status, vram_str, etc.)
        # Pipeline
        self.pipeline_config: Optional[Dict] = None
        self.pipeline_data: List[Dict] = []
        self.pipeline_results: Optional[Any] = None
        self.pipeline_running = False
        # Benchmark
        self.benchmark_config: Optional[Dict] = None
        self.benchmark_entries: List[Dict] = []
        self.benchmark_results: Optional[Any] = None
        self.benchmark_running = False

state = AppState()


# =============================================================================
# UTILITIES
# =============================================================================
def load_json_file(path: str) -> Optional[Any]:
    """Carga un archivo JSON de forma segura."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None


def format_json_preview(data: Any, max_lines: int = 6) -> str:
    """Formatea datos JSON para preview legible."""
    try:
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        lines = formatted.split('\n')
        if len(lines) > max_lines:
            return '\n'.join(lines[:max_lines]) + '\n  ...'
        return formatted
    except Exception:
        return str(data)[:200]


def format_bytes(bytes_val: int) -> str:
    """Formatea bytes a GB con 1 decimal."""
    return f"{bytes_val / (1024**3):.1f} GB"


# =============================================================================
# SYSTEM RESOURCES - Cache y optimizaciones
# =============================================================================
_system_resources_cache: Optional[Dict[str, Any]] = None
_system_resources_timestamp: Optional[float] = None
_RESOURCES_CACHE_TTL = 60  # 60 segundos de TTL

# pynvml singleton - inicializar una sola vez
_pynvml_initialized = False
_pynvml_handle = None


def _init_pynvml_once():
    """Inicializa pynvml una sola vez (singleton)."""
    global _pynvml_initialized, _pynvml_handle
    if PYNVML_AVAILABLE and not _pynvml_initialized:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                _pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _pynvml_initialized = True
        except Exception:
            _pynvml_initialized = True  # Marcar como intentado para no reintentar
            _pynvml_handle = None


def get_system_resources() -> Dict[str, Any]:
    """Obtiene información de recursos del sistema con cache TTL de 60s."""
    global _system_resources_cache, _system_resources_timestamp

    # Retornar cache si es reciente
    now = datetime.now().timestamp()
    if _system_resources_cache and _system_resources_timestamp:
        if now - _system_resources_timestamp < _RESOURCES_CACHE_TTL:
            return _system_resources_cache

    info = {
        'ram_total': 0,
        'ram_available': 0,
        'ram_used_percent': 0,
        'cpu_name': platform.processor() or 'Desconocido',
        'cpu_cores': psutil.cpu_count(logical=False) or 0,
        'cpu_threads': psutil.cpu_count(logical=True) or 0,
        'os': f"{platform.system()} {platform.release()}",
        'python_version': platform.python_version(),
        'gpu_detected': False,
        'gpu_name': None,
        'vram_total': 0,
        'vram_available': 0,
        'vram_used_percent': 0,
        'cuda_available': False,
        'cuda_version': None,
        'torch_version': None,
    }

    # RAM
    mem = psutil.virtual_memory()
    info['ram_total'] = mem.total
    info['ram_available'] = mem.available
    info['ram_used_percent'] = mem.percent

    # GPU con pynvml (singleton - no reinicializa)
    _init_pynvml_once()
    if _pynvml_handle is not None:
        try:
            info['gpu_detected'] = True
            gpu_name = pynvml.nvmlDeviceGetName(_pynvml_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            info['gpu_name'] = gpu_name

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(_pynvml_handle)
            info['vram_total'] = mem_info.total
            info['vram_available'] = mem_info.free
            info['vram_used_percent'] = (mem_info.used / mem_info.total) * 100
            info['cuda_available'] = True
        except Exception:
            pass

    # PyTorch info - siempre obtenemos la versión de torch
    torch = _get_torch()
    if torch is not None:
        info['torch_version'] = torch.__version__
        # Solo usamos torch para GPU si pynvml no la detectó
        if not info['gpu_detected']:
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_version'] = torch.version.cuda
                try:
                    info['gpu_detected'] = True
                    info['gpu_name'] = torch.cuda.get_device_name(0)
                    info['vram_total'] = torch.cuda.get_device_properties(0).total_memory
                    info['vram_available'] = info['vram_total'] - torch.cuda.memory_allocated(0)
                    info['vram_used_percent'] = (torch.cuda.memory_allocated(0) / info['vram_total']) * 100
                except Exception:
                    pass

    # Guardar en cache SOLO si torch se detectó correctamente
    # Si torch_version es None, puede que aún esté cargando - no cachear
    if info['torch_version'] is not None:
        _system_resources_cache = info
        _system_resources_timestamp = now

    return info


# Cache LRU para búsqueda de modelos HuggingFace (máximo 50 entradas)
_hf_search_cache: OrderedDict = OrderedDict()
_HF_CACHE_MAX_ENTRIES = 50

# Cliente HTTP reutilizable para HuggingFace API (evita crear conexiones nuevas cada vez)
_hf_http_client: Optional[httpx.AsyncClient] = None


def _get_hf_client() -> httpx.AsyncClient:
    """Obtiene o crea el cliente HTTP reutilizable para HuggingFace."""
    global _hf_http_client
    if _hf_http_client is None:
        _hf_http_client = httpx.AsyncClient(
            timeout=5.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    return _hf_http_client


async def warmup_hf_connection():
    """Pre-establece conexión a HuggingFace para reducir latencia en primera búsqueda."""
    try:
        client = _get_hf_client()
        # HEAD request ligero solo para establecer conexión TLS
        await client.head("https://huggingface.co/api/models", timeout=3.0)
    except Exception:
        pass  # Ignorar errores de warmup


def _add_to_hf_cache(key: str, value: List[Dict]):
    """Añade al cache LRU, eliminando entradas antiguas si excede el límite."""
    # Si ya existe, moverlo al final (más reciente)
    if key in _hf_search_cache:
        _hf_search_cache.move_to_end(key)
    _hf_search_cache[key] = value
    # Eliminar entradas más antiguas si excede el límite
    while len(_hf_search_cache) > _HF_CACHE_MAX_ENTRIES:
        _hf_search_cache.popitem(last=False)


# Cache de recursos del sistema (para no recalcular cada vez)
_system_vram_gb: Optional[float] = None
_system_resources_loading: bool = False

async def init_system_resources_async():
    """Inicializa recursos del sistema en background (no bloquea UI)."""
    global _system_vram_gb, _system_resources_loading
    if _system_vram_gb is not None or _system_resources_loading:
        return
    _system_resources_loading = True
    try:
        # Ejecutar en thread para no bloquear
        import asyncio
        loop = asyncio.get_event_loop()
        resources = await loop.run_in_executor(None, get_system_resources)
        if resources['gpu_detected'] and resources['vram_total'] > 0:
            _system_vram_gb = resources['vram_available'] / (1024**3)
        else:
            _system_vram_gb = resources['ram_available'] / (1024**3) * 0.5
    except Exception:
        _system_vram_gb = 8.0  # Default razonable si falla
    finally:
        _system_resources_loading = False

def get_available_vram_gb() -> float:
    """Obtiene la VRAM disponible en GB (usa valor cacheado o default)."""
    global _system_vram_gb
    if _system_vram_gb is None:
        return 8.0  # Default mientras carga (8GB es común)
    return _system_vram_gb

def estimate_vram_gb(params_billions: float) -> float:
    """Estima VRAM necesaria en GB basado en parámetros (asume fp16)."""
    # Regla: ~2 bytes por parámetro en fp16 + overhead (~20%)
    return params_billions * 2 * 1.2

def extract_params_from_name(model_id: str) -> Optional[float]:
    """Extrae número de parámetros del nombre del modelo."""
    import re
    model_lower = model_id.lower()

    # Patrones comunes: 7b, 7B, 1.5b, 70b, 0.5b, etc.
    patterns = [
        r'(\d+\.?\d*)b(?:illion)?(?:-|_|$)',  # 7b, 7B, 1.5b
        r'(\d+\.?\d*)B(?:-|_|$)',
        r'-(\d+\.?\d*)b',
        r'_(\d+\.?\d*)b',
    ]

    for pattern in patterns:
        match = re.search(pattern, model_id, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # Casos especiales conocidos
    special_cases = {
        'tiny': 0.1, 'small': 0.3, 'mini': 0.5, 'base': 0.1,
        'medium': 0.4, 'large': 0.8, 'xl': 1.5, 'xxl': 3.0,
    }
    for key, val in special_cases.items():
        if key in model_lower:
            return val

    return None

def format_params(params_b: Optional[float]) -> str:
    """Formatea parámetros para mostrar."""
    if params_b is None:
        return "?"
    if params_b >= 1:
        return f"{params_b:.1f}B"
    return f"{params_b*1000:.0f}M"

def format_vram(vram_gb: float) -> str:
    """Formatea VRAM para mostrar."""
    if vram_gb < 1:
        return f"{vram_gb*1024:.0f}MB"
    return f"{vram_gb:.1f}GB"

async def search_huggingface_models(query: str, limit: int = 15) -> List[Dict]:
    """Busca modelos en HuggingFace Hub API con información extendida.

    Prioriza modelos:
    1. Ejecutables localmente (suficiente VRAM)
    2. Con más descargas
    3. Actualizados recientemente (últimos 12 meses)
    """
    if not query or len(query) < 2:
        return []

    # Usar cache si existe
    cache_key = f"{query}:{limit}"
    if cache_key in _hf_search_cache:
        return _hf_search_cache[cache_key]

    available_vram = get_available_vram_gb()

    # Fecha límite: modelos actualizados en los últimos 12 meses
    cutoff_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    try:
        # Usar cliente reutilizable para mejor rendimiento
        client = _get_hf_client()
        response = await client.get(
            "https://huggingface.co/api/models",
            params={
                "search": query,
                "limit": 25,  # Pedir más para filtrar
                "filter": "text-generation",
                "sort": "downloads",
                "direction": -1,
            }
        )
        if response.status_code == 200:
            models = response.json()
            results = []
            for m in models:
                model_id = m.get('id', '')

                # Filtrar modelos que NO son de generación de texto
                # (embeddings, sentence-transformers, encoders, etc.)
                # y modelos cuantizados no compatibles (GGUF, etc.)
                model_id_lower = model_id.lower()
                skip_patterns = [
                    # Embeddings y encoders
                    'embed', 'embedding', 'sentence-transformer', 'bge-', 'e5-',
                    'gte-', 'instructor', 'encoder', 'retriev', 'rerank',
                    'clip', 'bert-base', 'bert-large', 'roberta', 'xlm-roberta',
                    'minilm', 'mpnet', 'contriever', 'colbert', 'splade',
                    # Formatos cuantizados NO compatibles con transformers
                    'gguf', 'ggml', '-gguf', '-ggml',
                ]
                if any(pattern in model_id_lower for pattern in skip_patterns):
                    continue

                # Verificar que el pipeline_tag sea text-generation (si está disponible)
                pipeline_tag = m.get('pipeline_tag', '')
                if pipeline_tag and pipeline_tag not in ['text-generation', 'text2text-generation']:
                    continue

                # Extraer parámetros desde safetensors o nombre
                params_b = None
                safetensors = m.get('safetensors', {})
                if safetensors and 'total' in safetensors:
                    params_b = safetensors['total'] / 1_000_000_000
                if params_b is None:
                    params_b = extract_params_from_name(model_id)

                # Estimar VRAM necesaria
                vram_needed = estimate_vram_gb(params_b) if params_b else None

                # Compatibilidad: 'compatible', 'limite', 'incompatible', None
                compat_status = None
                if vram_needed is not None and available_vram:
                    ratio = vram_needed / available_vram
                    if ratio <= 0.7:
                        compat_status = 'compatible'
                    elif ratio <= 1.0:
                        compat_status = 'limite'
                    else:
                        compat_status = 'incompatible'

                # Fecha de última modificación
                last_modified = m.get('lastModified', '')
                date_str = last_modified[:10] if last_modified else ''

                # Downloads (valor numérico para ordenar)
                downloads_num = m.get('downloads', 0)
                if downloads_num >= 1_000_000:
                    dl_str = f"{downloads_num/1_000_000:.1f}M"
                elif downloads_num >= 1_000:
                    dl_str = f"{downloads_num/1_000:.0f}K"
                else:
                    dl_str = str(downloads_num)

                results.append({
                    'value': model_id,
                    'model_id': model_id,
                    'params_b': params_b,
                    'params_str': format_params(params_b),
                    'vram_gb': vram_needed,
                    'vram_str': format_vram(vram_needed) if vram_needed else "?",
                    'compat_status': compat_status,
                    'date': date_str,
                    'downloads': dl_str,
                    'downloads_num': downloads_num,
                    'is_recent': date_str >= cutoff_date if date_str else False,
                })

            # Ordenar: primero ejecutables localmente, luego por descargas
            # Prioridad: compatible + reciente > compatible > reciente > otros
            def sort_key(r):
                compat = r.get('compat_status')
                is_recent = r.get('is_recent', False)
                downloads = r.get('downloads_num', 0)

                score = 0
                if compat == 'compatible':
                    score += 20000
                elif compat == 'limite':
                    score += 10000
                if is_recent:
                    score += 1000
                score += min(downloads / 1000, 999)
                return -score  # Negativo para orden descendente

            results.sort(key=sort_key)
            results = results[:limit]  # Limitar a los mejores

            _add_to_hf_cache(cache_key, results)
            return results
    except httpx.TimeoutException:
        print("HuggingFace search: timeout de conexión (red lenta)")
    except httpx.ConnectError:
        print("HuggingFace search: error de conexión (sin internet)")
    except Exception as e:
        print(f"HuggingFace search error: {type(e).__name__}: {e}")
    return []


def build_pipeline_steps(config: Dict) -> List[PipelineStep]:
    """Construye objetos PipelineStep desde configuración JSON."""
    steps = []
    raw_steps = config.get('steps', config.get('pipeline_steps', []))

    for step_config in raw_steps:
        step_type = step_config.get('type', '')
        params = step_config.get('parameters', {})

        try:
            if step_type == 'generate':
                parameters = GenerateTextRequest(
                    system_prompt=params['system_prompt'],
                    user_prompt=params['user_prompt'],
                    num_sequences=params.get('num_sequences', 1),
                    max_tokens=params.get('max_tokens', 100),
                    temperature=params.get('temperature', 0.7)
                )
            elif step_type == 'parse':
                rules = [
                    ParseRule(
                        name=r['name'],
                        pattern=r['pattern'],
                        mode=ParseMode[r.get('mode', 'KEYWORD').upper()],
                        secondary_pattern=r.get('secondary_pattern'),
                        fallback_value=r.get('fallback_value')
                    ) for r in params.get('rules', [])
                ]
                parameters = ParseRequest(
                    rules=rules,
                    output_filter=params.get('output_filter', 'all'),
                    output_limit=params.get('output_limit')
                )
            elif step_type == 'verify':
                methods = [
                    VerificationMethod(
                        mode=VerificationMode[m.get('mode', 'cumulative').upper()],
                        name=m['name'],
                        system_prompt=m['system_prompt'],
                        user_prompt=m['user_prompt'],
                        num_sequences=m.get('num_sequences', 3),
                        valid_responses=m['valid_responses'],
                        required_matches=m.get('required_matches', 1),
                        max_tokens=m.get('max_tokens', 10),
                        temperature=m.get('temperature', 1.0)
                    ) for m in params.get('methods', [])
                ]
                parameters = VerifyRequest(
                    methods=methods,
                    required_for_confirmed=params.get('required_for_confirmed', 1),
                    required_for_review=params.get('required_for_review', 0)
                )
            else:
                continue

            steps.append(PipelineStep(
                type=step_type,
                parameters=parameters,
                uses_reference=step_config.get('uses_reference', False),
                reference_step_numbers=step_config.get('reference_step_numbers', []),
                llm_config=step_config.get('llm_config')
            ))
        except Exception as e:
            print(f"Error building step {step_type}: {e}")
            continue

    return steps


# =============================================================================
# CSS STYLES
# =============================================================================
CUSTOM_CSS = '''
<style>
:root {
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-input: #334155;
    --border: #334155;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
}

body {
    background: var(--bg-dark) !important;
    color: var(--text) !important;
}

/* Fondo uniforme con degradado sutil */
.nicegui-content {
    background: linear-gradient(180deg, #0f172a 0%, #1a1f35 50%, #0f172a 100%) !important;
    min-height: 100vh;
}

/* Header uniforme */
.q-header {
    background: rgba(15, 23, 42, 0.95) !important;
    backdrop-filter: blur(8px);
}

/* Remove default padding and background from tab panels */
.q-tab-panel {
    padding: 0 !important;
    background: transparent !important;
}
.q-tab-panels {
    background: transparent !important;
}
.q-page-container {
    background: transparent !important;
}
.q-layout {
    background: transparent !important;
}

/* Card styles */
.card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(71, 85, 105, 0.5);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(4px);
}

/* Step cards */
.step-card {
    background: rgba(30, 41, 59, 0.5);
    border: 2px solid rgba(71, 85, 105, 0.4);
    border-radius: 12px;
    padding: 16px;
    transition: all 0.2s ease;
}
.step-card.active {
    border-color: var(--accent);
    background: rgba(99, 102, 241, 0.08);
}
.step-card.done {
    border-color: var(--success);
    background: rgba(34, 197, 94, 0.08);
}

/* Step number badge */
.step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--border);
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 13px;
    flex-shrink: 0;
}
.step-card.active .step-num {
    background: var(--accent);
    color: white;
}
.step-card.done .step-num {
    background: var(--success);
    color: white;
}

/* Metric display */
.metric-box {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    min-width: 100px;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* JSON preview box */
.json-preview {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 11px;
    color: var(--text-muted);
    max-height: 120px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

/* Buttons - Ejecutar usa emerald oscuro (paso 3) */
.btn-run {
    background: linear-gradient(135deg, #0f4342, #0d3a39) !important;
    color: #6ee7b7 !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    border: 1px solid rgba(16, 185, 129, 0.5) !important;
}
.btn-run:disabled {
    opacity: 0.5 !important;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
}
.pill-info { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; }
.pill-success { background: rgba(34, 197, 94, 0.2); color: #86efac; }
.pill-error { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }

/* Loading container - usa emerald (paso 3) */
.loading-box {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 10px;
    padding: 16px;
}

/* Confusion matrix cells */
.cm-cell {
    padding: 12px;
    border-radius: 6px;
    text-align: center;
}
.cm-good { background: rgba(34, 197, 94, 0.15); }
.cm-bad { background: rgba(239, 68, 68, 0.15); }

/* Feature cards on home */
.feature-box {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 16px;
    padding: 28px 32px;
    transition: all 0.25s ease;
    backdrop-filter: blur(4px);
    position: relative;
    overflow: hidden;
}
.feature-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), #8b5cf6);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.feature-box:hover {
    transform: translateY(-4px);
    border-color: var(--accent);
    background: rgba(30, 41, 59, 0.7);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
}
.feature-box:hover::before {
    opacity: 1;
}

/* Tabs in header */
.q-tab {
    text-transform: none !important;
    font-weight: 500 !important;
    opacity: 0.7;
}
.q-tab--active {
    opacity: 1 !important;
    color: #818cf8 !important;
}
.q-tabs__content { border: none !important; }
.q-tab-panels { padding-top: 0 !important; }

/* Scrollbar sutil */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Placeholder más gris */
.q-field input::placeholder,
.q-field textarea::placeholder,
.q-field--dense textarea::placeholder,
.q-textarea textarea::placeholder,
.q-field__native::placeholder {
    color: #64748b !important;
    opacity: 0.7 !important;
}

/* Consistencia total entre placeholder y texto en inputs/textareas */
.q-field input,
.q-field textarea,
.q-field__native,
input,
textarea {
    font-family: inherit !important;
    font-size: inherit !important;
    line-height: inherit !important;
}
.q-field input::placeholder,
.q-field textarea::placeholder,
input::placeholder,
textarea::placeholder {
    font-family: inherit !important;
    font-size: inherit !important;
    line-height: inherit !important;
}

/* Fix padding vertical en textareas dense para centrar texto */
.q-field--dense.q-textarea .q-field__native {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

/* Bordes más sutiles en inputs/textareas outlined */
.q-field--outlined .q-field__control:before {
    border-color: rgba(100, 116, 139, 0.3) !important;
}
.q-field--outlined.q-field--focused .q-field__control:before {
    border-color: rgba(168, 85, 247, 0.5) !important;
}
.q-field--outlined:hover .q-field__control:before {
    border-color: rgba(100, 116, 139, 0.5) !important;
}

/* Toggle de datos de entrada - centrado entre título y subtítulo */
.input-vars-toggle .q-toggle.mt-px {
    margin-top: 6px !important;
}

/* Fix alineación vertical en botones con icono + texto */
.q-btn.flex.items-center .nicegui-label,
.q-btn.flex.items-center .q-icon {
    display: inline-flex !important;
    align-items: center !important;
    vertical-align: middle !important;
    height: 16px !important;
    line-height: 16px !important;
}

/* Textarea JSON sin resize */
.json-editor-textarea textarea {
    resize: none !important;
}
.json-editor-textarea .q-field__control {
    height: 100% !important;
}
.json-editor-textarea .q-field__native {
    height: 100% !important;
}

/* Tooltips - estilo unificado */
.q-tooltip {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    border: 1px solid rgba(71, 85, 105, 0.5) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    white-space: pre-line !important;
}

/* Slider de temperatura personalizado */
.temp-slider .q-slider__track-container {
    color: #90a1b9 !important;
}
.temp-slider .q-slider__track {
    background: rgba(144, 161, 185, 0.3) !important;
}
.temp-slider .q-slider__selection {
    background: #90a1b9 !important;
}
.temp-slider .q-slider__thumb {
    color: #90a1b9 !important;
}
.temp-slider .q-slider__focus-ring {
    background: rgba(144, 161, 185, 0.3) !important;
}

/* Checkboxes con color de fondo oscuro y borde */
.q-checkbox__inner {
    color: #151c30 !important;
}
.q-checkbox__inner--truthy {
    color: #151c30 !important;
}
.q-checkbox__bg {
    border: 1.5px solid #64748b !important;  /* slate-500 border */
    border-radius: 3px !important;
}
.q-checkbox__inner--truthy .q-checkbox__bg {
    border-color: #94a3b8 !important;  /* slate-400 cuando está activo */
}
</style>
'''


# =============================================================================
# HOME PAGE
# =============================================================================
def home_page(tabs=None):
    """Página de inicio."""
    with ui.column().classes('w-full max-w-4xl mx-auto gap-8 p-6'):
        # Hero
        with ui.column().classes('w-full items-center text-center gap-3 py-4'):
            # Logo + Título en la misma línea
            with ui.row().classes('items-center gap-4'):
                with ui.element('div').classes('w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30'):
                    ui.icon('psychology', size='lg').classes('text-white')
                ui.label('EsencIA').classes('text-5xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text')
            ui.label('Procesamiento local y confiable con modelos de lenguaje').classes('text-lg text-slate-400')

        # Cómo empezar - Línea de tiempo horizontal
        with ui.column().classes('w-full items-center gap-4 mb-0'):
            ui.label('Cómo funciona').classes('text-base text-slate-400 uppercase tracking-widest mb-2')

            with ui.row().classes('w-full justify-center items-start gap-0'):
                # Paso 1
                with ui.column().classes('items-center gap-2 flex-1 max-w-[200px]'):
                    with ui.element('div').classes('w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500/30 to-indigo-600/20 border-2 border-indigo-500/50 flex items-center justify-center'):
                        ui.label('1').classes('text-indigo-400 font-bold text-sm')
                    ui.label('Elige Modo').classes('text-xl font-semibold text-white')
                    ui.label('Pipeline o Benchmark').classes('text-sm text-slate-400 text-center')

                # Conector
                ui.element('div').classes('h-0.5 w-12 bg-gradient-to-r from-indigo-500/50 to-purple-500/50 mt-5')

                # Paso 2
                with ui.column().classes('items-center gap-2 flex-1 max-w-[200px]'):
                    with ui.element('div').classes('w-8 h-8 rounded-full bg-gradient-to-br from-purple-500/30 to-purple-600/20 border-2 border-purple-500/50 flex items-center justify-center'):
                        ui.label('2').classes('text-purple-400 font-bold text-sm')
                    ui.label('Configura').classes('text-xl font-semibold text-white')
                    with ui.column().classes('items-center gap-0'):
                        ui.label('Elige modelos de lenguaje').classes('text-sm text-slate-400 text-center')
                        ui.label('Crea un flujo de procesamiento').classes('text-sm text-slate-400 text-center')
                        ui.label('Guarda configuraciones').classes('text-sm text-slate-400 text-center')

                # Conector
                ui.element('div').classes('h-0.5 w-12 bg-gradient-to-r from-purple-500/50 to-emerald-500/50 mt-5')

                # Paso 3
                with ui.column().classes('items-center gap-2 flex-1 max-w-[200px]'):
                    with ui.element('div').classes('w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500/30 to-emerald-600/20 border-2 border-emerald-500/50 flex items-center justify-center'):
                        ui.label('3').classes('text-emerald-400 font-bold text-sm')
                    ui.label('Ejecuta').classes('text-xl font-semibold text-white')
                    ui.label('Visualiza y exporta resultados').classes('text-sm text-slate-400 text-center')

        # Título MODOS + Cards juntos para reducir espacio
        with ui.column().classes('w-full items-center gap-4 mt-1'):
            ui.label('Modos').classes('text-base text-slate-400 uppercase tracking-widest')

            # Features - Cards lado a lado (borde indigo para conectar con paso 1 "Elige Modo")
            with ui.row().classes('w-full gap-4 justify-center'):
                # Pipeline - clickable
                with ui.column().classes('feature-box w-[360px] gap-4 cursor-pointer items-center relative border-2 border-indigo-500/50').on('click', lambda: tabs.set_value('Pipeline') if tabs else None):
                    # Icono centrado
                    with ui.element('div').classes('w-14 h-14 rounded-xl bg-indigo-500/20 flex items-center justify-center'):
                        ui.icon('account_tree', size='lg').classes('text-indigo-400')

                    # Descripción centrada
                    with ui.row().classes('justify-center gap-1'):
                        ui.label('Diseña y Ejecuta').classes('text-lg text-slate-400')
                        ui.label('Pipelines').classes('text-lg text-indigo-400 font-bold')

                    # Features list - iconos representativos en azul
                    with ui.column().classes('gap-2 mt-2 w-full'):
                        pipeline_features = [
                            ('playlist_play', 'Ejecución con múltiples inputs'),
                            ('auto_fix_high', 'Genera, detecta patrones, y verifica'),
                            ('file_download', 'Exporta tus configuraciones')
                        ]
                        for icon_name, txt in pipeline_features:
                            with ui.row().classes('items-center gap-3'):
                                ui.icon(icon_name, size='xs').classes('text-indigo-400')
                                ui.label(txt).classes('text-base text-slate-300')

                    # Flecha abajo derecha
                    ui.icon('arrow_forward', size='sm').classes('absolute bottom-4 right-4 text-indigo-400')

                # Benchmark - clickable
                with ui.column().classes('feature-box w-[360px] gap-4 cursor-pointer items-center relative border-2 border-indigo-500/50').on('click', lambda: tabs.set_value('Benchmark') if tabs else None):
                    # Icono centrado
                    with ui.element('div').classes('w-14 h-14 rounded-xl bg-indigo-500/20 flex items-center justify-center'):
                        ui.icon('analytics', size='lg').classes('text-indigo-400')

                    # Descripción centrada
                    with ui.row().classes('justify-center gap-1'):
                        ui.label('Realiza').classes('text-lg text-slate-400')
                        ui.label('Benchmarks').classes('text-lg text-indigo-400 font-bold')

                    # Features list - iconos representativos en azul
                    with ui.column().classes('gap-2 mt-2 w-full'):
                        benchmark_features = [
                            ('percent', 'Accuracy, precision, recall'),
                            ('grid_on', 'Matriz de confusión'),
                            ('bug_report', 'Análisis de errores')
                        ]
                        for icon_name, txt in benchmark_features:
                            with ui.row().classes('items-center gap-3'):
                                ui.icon(icon_name, size='xs').classes('text-indigo-400')
                                ui.label(txt).classes('text-base text-slate-300')

                    # Flecha abajo derecha
                    ui.icon('arrow_forward', size='sm').classes('absolute bottom-4 right-4 text-indigo-400')

        # Recursos del sistema - diseño compacto con barras (carga asíncrona)
        with ui.column().classes('w-full items-center gap-4 mt-1'):
            ui.label('RECURSOS DEL SISTEMA').classes('text-sm text-slate-400 uppercase tracking-widest')

            # Contenedor para actualización asíncrona
            resources_container = ui.column().classes('w-full items-center')

            def render_resources_ui(resources: Dict[str, Any]):
                """Renderiza la UI de recursos del sistema."""
                resources_container.clear()
                with resources_container:
                    with ui.row().classes('items-center gap-8 px-6 py-4 rounded-xl bg-slate-800/30 border border-slate-700/50'):
                        # GPU status (tema azul indigo para coherencia visual)
                        with ui.column().classes('gap-1 min-w-[200px]'):
                            with ui.row().classes('items-center gap-2'):
                                if resources['gpu_detected']:
                                    ui.icon('memory', size='xs').classes('text-indigo-400')
                                    ui.label(resources['gpu_name'] or 'GPU').classes('text-sm text-slate-300 truncate')
                                else:
                                    ui.icon('memory', size='xs').classes('text-slate-400')
                                    ui.label('Sin GPU').classes('text-sm text-slate-400')

                            if resources['gpu_detected'] and resources['vram_total'] > 0:
                                with ui.row().classes('w-full items-center gap-2'):
                                    with ui.element('div').classes('flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden'):
                                        ui.element('div').classes('h-full bg-indigo-500 rounded-full').style(f"width: {resources['vram_used_percent']:.0f}%")
                                    vram_used = resources['vram_total'] - resources['vram_available']
                                    ui.label(f"{format_bytes(vram_used)}/{format_bytes(resources['vram_total'])}").classes('text-xs text-slate-400 whitespace-nowrap')

                        # Separador
                        ui.element('div').classes('w-px h-8 bg-slate-600')

                        # RAM status
                        with ui.column().classes('gap-1 min-w-[180px]'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('storage', size='xs').classes('text-indigo-400')
                                ui.label('RAM').classes('text-sm text-slate-300')

                            with ui.row().classes('w-full items-center gap-2'):
                                with ui.element('div').classes('flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden'):
                                    ui.element('div').classes('h-full bg-indigo-500 rounded-full').style(f"width: {resources['ram_used_percent']:.0f}%")
                                ram_used = resources['ram_total'] - resources['ram_available']
                                ui.label(f"{format_bytes(ram_used)}/{format_bytes(resources['ram_total'])}").classes('text-xs text-slate-400 whitespace-nowrap')

                    # Avisos de configuración (solo problemas reales y confiables)
                    warnings = []

                    # GPU detectada pero CUDA no disponible (problema de drivers)
                    if resources['gpu_detected'] and not resources['cuda_available']:
                        warnings.append({
                            'icon': 'warning',
                            'color': 'amber',
                            'title': 'GPU detectada sin CUDA',
                            'msg': 'Instala CUDA Toolkit para usar la GPU'
                        })

                    # Nota: No mostramos aviso de PyTorch porque con lazy loading
                    # puede dar falsos positivos mientras torch se carga en background

                    if warnings:
                        with ui.column().classes('w-full items-center gap-2 mt-2'):
                            for w in warnings:
                                with ui.row().classes(f'items-center gap-3 p-3 rounded-lg bg-{w["color"]}-500/10 border border-{w["color"]}-500/30'):
                                    ui.icon(w['icon'], size='sm').classes(f'text-{w["color"]}-400')
                                    with ui.column().classes('gap-0'):
                                        ui.label(w['title']).classes(f'text-sm font-medium text-{w["color"]}-300')
                                        ui.label(w['msg']).classes('text-xs text-slate-400')

            # Mostrar placeholder mientras carga
            with resources_container:
                with ui.row().classes('items-center gap-3 px-6 py-4 rounded-xl bg-slate-800/30 border border-slate-700/50'):
                    ui.spinner('dots', size='sm').classes('text-indigo-400')
                    ui.label('Detectando recursos del sistema...').classes('text-sm text-slate-400')

            # Cargar recursos de forma asíncrona
            async def load_resources_async():
                """Carga recursos del sistema sin bloquear la UI."""
                try:
                    resources = await run.io_bound(get_system_resources)
                    render_resources_ui(resources)
                except Exception:
                    resources_container.clear()
                    with resources_container:
                        with ui.row().classes('items-center gap-2 text-slate-400'):
                            ui.icon('info', size='xs')
                            ui.label('No se pudieron detectar los recursos').classes('text-sm')

            asyncio.create_task(load_resources_async())



# =============================================================================
# PIPELINE PAGE - TEMPLATES
# =============================================================================

# Plantillas predefinidas para tareas comunes
PIPELINE_TEMPLATES = {
    'lectura_facil': {
        'name': 'Adaptar a Lectura Fácil',
        'description': 'Simplifica textos para hacerlos más accesibles',
        'icon': 'accessibility_new',
        'color': 'emerald',
        'steps': [
            {
                'type': 'generate',
                'parameters': {
                    'system_prompt': 'Eres un experto en lectura fácil. Adapta textos usando oraciones cortas, vocabulario simple y estructura clara.',
                    'user_prompt': 'Adapta el siguiente texto a lectura fácil:\n\n{texto}',
                    'num_sequences': 1,
                    'max_tokens': 500,
                    'temperature': 0.3
                },
                'uses_reference': True
            }
        ],
        'variables': ['texto']
    },
    'resumir': {
        'name': 'Resumir Texto',
        'description': 'Genera resúmenes concisos de textos largos',
        'icon': 'summarize',
        'color': 'blue',
        'steps': [
            {
                'type': 'generate',
                'parameters': {
                    'system_prompt': 'Eres un asistente que genera resúmenes concisos y precisos.',
                    'user_prompt': 'Resume el siguiente texto en máximo 3 oraciones:\n\n{texto}',
                    'num_sequences': 1,
                    'max_tokens': 200,
                    'temperature': 0.5
                },
                'uses_reference': True
            }
        ],
        'variables': ['texto']
    },
    'extraer_info': {
        'name': 'Extraer Información',
        'description': 'Extrae datos estructurados de textos',
        'icon': 'find_in_page',
        'color': 'amber',
        'steps': [
            {
                'type': 'generate',
                'parameters': {
                    'system_prompt': 'Extrae información del texto y responde en formato:\nDato: valor',
                    'user_prompt': 'Del siguiente texto, extrae {campo}:\n\n{texto}',
                    'num_sequences': 1,
                    'max_tokens': 150,
                    'temperature': 0.2,
                    'parse_rules': [
                        {'name': 'dato', 'mode': 'KEYWORD', 'pattern': 'Dato:', 'secondary_pattern': '\n', 'fallback_value': 'no_encontrado'}
                    ]
                },
                'uses_reference': True
            }
        ],
        'variables': ['texto', 'campo']
    },
    'verificar': {
        'name': 'Generar y Verificar',
        'description': 'Genera contenido y lo valida automáticamente',
        'icon': 'verified',
        'color': 'violet',
        'steps': [
            {
                'type': 'generate',
                'parameters': {
                    'system_prompt': 'Genera contenido siguiendo el formato: Contenido: "tu respuesta"',
                    'user_prompt': '{instruccion}',
                    'num_sequences': 2,
                    'max_tokens': 200,
                    'temperature': 0.7,
                    'parse_rules': [
                        {'name': 'contenido', 'mode': 'KEYWORD', 'pattern': 'Contenido:', 'secondary_pattern': '"', 'fallback_value': ''}
                    ]
                },
                'uses_reference': True
            },
            {
                'type': 'verify',
                'parameters': {
                    'methods': [
                        {
                            'mode': 'cumulative',
                            'name': 'validar_contenido',
                            'system_prompt': 'Responde solo Yes o No.',
                            'user_prompt': 'Es el siguiente contenido apropiado y coherente? {contenido}',
                            'num_sequences': 3,
                            'valid_responses': ['Yes', 'yes', 'Si', 'si'],
                            'required_matches': 2,
                            'max_tokens': 5,
                            'temperature': 0.8
                        }
                    ],
                    'required_for_confirmed': 1,
                    'required_for_review': 0
                },
                'uses_reference': True,
                'reference_step_numbers': [0]
            }
        ],
        'variables': ['instruccion']
    },
    'personalizado': {
        'name': 'Pipeline Personalizado',
        'description': 'Crea tu propio flujo desde cero',
        'icon': 'build',
        'color': 'slate',
        'steps': [],
        'variables': []
    }
}


# =============================================================================
# PIPELINE PAGE
# =============================================================================
def pipeline_page():
    """Página del pipeline con constructor visual mejorado."""
    import re
    import copy

    # Estado local del constructor
    local_state = {
        'selected_template': None,
        'selected_step': 0,  # Índice del paso seleccionado para edición
        'steps': [],
        'input_vars_enabled': False,  # Toggle para habilitar variables de entrada
        'input_fields': [],  # Campos definidos (crean variables)
        'input_values': {},  # Valores por campo: {'texto': ['val1', 'val2'], 'autor': ['a1']}
        'placeholder_fields': set(),  # Variables que muestran placeholder (sin nombre asignado aún)
        'expanded_var': None,  # Variable expandida (None = todas colapsadas)
        'detected_vars': set(),  # Variables detectadas en prompts (compatibilidad)
        'json_view': False,  # Toggle para vista JSON de variables
        'json_edit_mode': False,  # Modo edición del JSON (False = solo lectura)
        'json_edit_buffer': '',  # Buffer para edición del JSON
        'pipeline_json_view': False,  # Toggle para vista JSON del pipeline
        'pipeline_json_edit_mode': False,  # Modo edición del JSON del pipeline
        'pipeline_json_edit_buffer': ''  # Buffer para edición del JSON del pipeline
    }

    # Referencias UI
    steps_list_container = None
    data_container = None
    results_container = None
    run_btn = None
    loading_box = None
    steps_counter_label = None
    entries_counter_label = None
    model_selector_ref = {'container': None, 'switch_to_edit': None}

    # Referencias a campos de formulario para validación
    # Estructura: field_refs[step_idx] = {'system_prompt': textarea_ref, 'user_prompt': textarea_ref, ...}
    field_refs = {}

    # CSS para animación de error (shake) e inputs sutiles
    ui.add_css('''
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            20%, 60% { transform: translateX(-4px); }
            40%, 80% { transform: translateX(4px); }
        }
        .shake-error {
            animation: shake 0.4s ease-in-out;
        }
        .input-error .q-field__control {
            border-color: rgb(239 68 68) !important;
        }
        /* Inputs sutiles - igual que el buscador */
        .input-subtle.q-field--outlined .q-field__control:before,
        .input-subtle.q-field--outlined .q-field__control:after,
        .input-subtle.q-textarea--outlined .q-field__control:before,
        .input-subtle.q-textarea--outlined .q-field__control:after {
            border: 1px solid rgba(71, 85, 105, 0.5) !important;
            border-width: 1px !important;
        }
        .input-subtle.q-field--outlined:hover .q-field__control:before,
        .input-subtle.q-field--outlined:hover .q-field__control:after,
        .input-subtle.q-textarea--outlined:hover .q-field__control:before,
        .input-subtle.q-textarea--outlined:hover .q-field__control:after {
            border-color: rgba(168, 85, 247, 0.5) !important;
            border-width: 1px !important;
        }
        .input-subtle.q-field--outlined.q-field--focused .q-field__control:before,
        .input-subtle.q-field--outlined.q-field--focused .q-field__control:after,
        .input-subtle.q-textarea--outlined.q-field--focused .q-field__control:before,
        .input-subtle.q-textarea--outlined.q-field--focused .q-field__control:after {
            border-color: rgb(168, 85, 247) !important;
            border-width: 1px !important;
        }
        .input-subtle .q-field__native,
        .input-subtle textarea {
            color: #e2e8f0 !important;
        }
        .input-subtle .q-field__native::placeholder,
        .input-subtle textarea::placeholder {
            color: rgba(148, 163, 184, 0.5) !important;
        }
        /* Input compacto para nombre de variable */
        .var-name-input {
            min-height: 0 !important;
            padding: 0 !important;
        }
        .var-name-input .q-field__control {
            height: auto !important;
            min-height: 0 !important;
            padding: 0 !important;
        }
        .var-name-input .q-field__native {
            padding: 0 !important;
            min-height: 0 !important;
            height: auto !important;
            line-height: 1.2 !important;
        }
        .var-name-input .q-field__marginal {
            height: auto !important;
        }
        /* Animación de loading para barra de progreso */
        @keyframes loading-progress {
            0% { width: 20%; opacity: 0.5; }
            50% { width: 80%; opacity: 1; }
            100% { width: 20%; opacity: 0.5; }
        }
        /* Ocultar spinners de inputs numéricos */
        .q-field input[type="number"]::-webkit-outer-spin-button,
        .q-field input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        .q-field input[type="number"] {
            -moz-appearance: textfield;
        }
    ''')

    def detect_variables(steps: List[Dict]) -> set:
        """Detecta variables {var} en los prompts de los pasos."""
        vars_found = set()
        pattern = r'\{(\w+)\}'
        for step in steps:
            params = step.get('parameters', {})
            for key in ['system_prompt', 'user_prompt']:
                if key in params:
                    matches = re.findall(pattern, params[key])
                    vars_found.update(matches)
            # También buscar en métodos de verificación
            methods = params.get('methods', [])
            for method in methods:
                for key in ['system_prompt', 'user_prompt']:
                    if key in method:
                        matches = re.findall(pattern, method[key])
                        vars_found.update(matches)
        return vars_found

    def select_template(template_key: str):
        """Selecciona una plantilla y carga sus pasos."""
        template = PIPELINE_TEMPLATES.get(template_key)
        if not template:
            return

        local_state['selected_template'] = template_key
        local_state['steps'] = [dict(s) for s in template['steps']]  # Copia profunda
        local_state['detected_vars'] = detect_variables(local_state['steps'])

        # Actualizar UI
        refresh_builder()
        refresh_data_section()

    def add_step(step_type: str):
        """Añade un nuevo paso al pipeline."""
        if step_type == 'generate':
            new_step = {
                'type': 'generate',
                'parameters': {
                    'system_prompt': '',
                    'user_prompt': '',
                    'num_sequences': 1,
                    'max_tokens': 200,
                    'temperature': 0.7
                },
                'uses_reference': False  # El paso 0 no tiene referencias; se activa manualmente
            }
        elif step_type == 'parse':
            new_step = {
                'type': 'parse',
                'parameters': {
                    'rules': [{
                        'name': '',
                        'mode': 'KEYWORD',
                        'pattern': '',
                        'secondary_pattern': '',
                        'fallback_value': ''
                    }],
                    'output_filter': 'all'
                },
                'uses_reference': True,
                'reference_step_numbers': [len(local_state['steps']) - 1] if local_state['steps'] else []
            }
        elif step_type == 'verify':
            new_step = {
                'type': 'verify',
                'parameters': {
                    'methods': [{
                        'mode': 'eliminatory',
                        'name': 'metodo_1',
                        'system_prompt': '',
                        'user_prompt': '',
                        'num_sequences': 3,
                        'valid_responses': [],
                        'required_matches': 2,
                        'max_tokens': 5,
                        'temperature': 0.8
                    }],
                    'required_for_confirmed': 1,
                    'required_for_review': 0
                },
                'uses_reference': True,
                'reference_step_numbers': [len(local_state['steps']) - 1] if local_state['steps'] else []
            }
        else:
            return

        local_state['steps'].append(new_step)
        local_state['expanded_step'] = len(local_state['steps']) - 1  # Expandir el nuevo paso
        local_state['detected_vars'] = detect_variables(local_state['steps'])
        refresh_builder()
        refresh_data_section()

    def remove_step(index: int):
        """Elimina un paso del pipeline."""
        if 0 <= index < len(local_state['steps']):
            local_state['steps'].pop(index)
            # Actualizar referencias
            for step in local_state['steps']:
                refs = step.get('reference_step_numbers', [])
                step['reference_step_numbers'] = [r for r in refs if r < len(local_state['steps'])]
            local_state['detected_vars'] = detect_variables(local_state['steps'])
            refresh_builder()
            refresh_data_section()

    def update_step_param(step_idx: int, param_path: str, value):
        """Actualiza un parámetro de un paso."""
        if 0 <= step_idx < len(local_state['steps']):
            step = local_state['steps'][step_idx]
            params = step.get('parameters', {})
            params[param_path] = value
            step['parameters'] = params
            local_state['detected_vars'] = detect_variables(local_state['steps'])
            refresh_data_section()

    def update_step_field(step_idx: int, field: str, value):
        """Actualiza un campo a nivel de step (no en parameters)."""
        if 0 <= step_idx < len(local_state['steps']):
            local_state['steps'][step_idx][field] = value
            if field in ('uses_reference', 'reference_step_numbers'):
                local_state['detected_vars'] = detect_variables(local_state['steps'])
                refresh_data_section()

    def toggle_reference_step(step_idx: int, ref_step: int, checked: bool):
        """Añade o quita un paso de las referencias."""
        if 0 <= step_idx < len(local_state['steps']):
            step = local_state['steps'][step_idx]
            refs = step.get('reference_step_numbers', [])
            if checked and ref_step not in refs:
                refs.append(ref_step)
                refs.sort()
            elif not checked and ref_step in refs:
                refs.remove(ref_step)
            step['reference_step_numbers'] = refs
            step['uses_reference'] = len(refs) > 0
            local_state['detected_vars'] = detect_variables(local_state['steps'])
            refresh_builder()  # Re-renderizar para actualizar checkboxes

    def add_parse_rule(step_idx: int):
        """Añade una regla de parsing."""
        if 0 <= step_idx < len(local_state['steps']):
            step = local_state['steps'][step_idx]
            if 'parameters' not in step:
                step['parameters'] = {}
            if 'rules' not in step['parameters']:
                step['parameters']['rules'] = []
            rules = step['parameters']['rules']
            rules.append({
                'name': '',
                'mode': 'KEYWORD',
                'pattern': '',
                'secondary_pattern': '',
                'fallback_value': ''
            })
            refresh_builder()

    def add_verify_method(step_idx: int):
        """Añade un método de verificación con nombre incremental único."""
        if 0 <= step_idx < len(local_state['steps']):
            methods = local_state['steps'][step_idx]['parameters'].get('methods', [])

            # Generar nombre único incremental
            existing_names = {m.get('name', '') for m in methods}
            counter = 1
            while f'metodo_{counter}' in existing_names:
                counter += 1

            methods.append({
                'mode': 'eliminatory',
                'name': f'metodo_{counter}',
                'system_prompt': '',
                'user_prompt': '',
                'num_sequences': 3,
                'valid_responses': [],
                'required_matches': 2,
                'max_tokens': 5,
                'temperature': 0.8,
                'ignore_case': True
            })
            local_state['steps'][step_idx]['parameters']['methods'] = methods
            refresh_builder()

    def add_input_field(field_name: str = None, show_placeholder: bool = False):
        """Añade un nuevo campo de entrada.

        Args:
            field_name: Nombre del campo. Si None, genera uno automático.
            show_placeholder: Si True, el campo se renderiza con placeholder vacío.
        """
        if field_name is None:
            # Generar nombre único: var_1, var_2, var_3, etc.
            existing_fields = local_state['input_fields']
            counter = 1
            while f'var_{counter}' in existing_fields:
                counter += 1
            field_name = f'var_{counter}'

        # Evitar duplicados
        if field_name in local_state['input_fields']:
            return False

        local_state['input_fields'].append(field_name)

        # Marcar como placeholder si corresponde
        if show_placeholder:
            local_state['placeholder_fields'].add(field_name)

        # Sincronizar con el número de filas existente
        current_row_count = max((len(vals) for vals in local_state['input_values'].values()), default=0)
        if current_row_count == 0:
            # Primera variable: iniciar con una fila vacía
            local_state['input_values'][field_name] = ['']
        else:
            # Hay filas existentes: igualar el número de filas
            local_state['input_values'][field_name] = [''] * current_row_count

        refresh_data_section()
        refresh_builder()  # Actualizar chips de variables en pasos
        return True

    def remove_input_field(field_name: str):
        """Elimina un campo de entrada."""
        if field_name in local_state['input_fields']:
            local_state['input_fields'].remove(field_name)
            local_state['input_values'].pop(field_name, None)
            local_state['placeholder_fields'].discard(field_name)
            refresh_data_section()
            refresh_builder()  # Actualizar chips de variables en pasos

    def update_input_field_name(old_name: str, new_name: str, refresh: bool = True):
        """Actualiza el nombre de un campo.

        Args:
            old_name: Nombre actual del campo
            new_name: Nuevo nombre para el campo
            refresh: Si True, refresca la UI (usar False durante typing)
        """
        new_name_stripped = new_name.strip()
        # Durante typing, permitir nombre vacío temporalmente
        if not refresh and not new_name_stripped:
            return False
        if refresh and not new_name_stripped:
            return False
        if old_name not in local_state['input_fields']:
            return False
        if new_name_stripped == old_name:
            return False  # Sin cambios
        if new_name_stripped in local_state['input_fields'] and new_name_stripped != old_name:
            if refresh:
                ui.notify(f'Ya existe una variable "{new_name_stripped}"', type='warning')
            return False

        idx = local_state['input_fields'].index(old_name)
        local_state['input_fields'][idx] = new_name_stripped

        # Actualizar en input_values
        if old_name in local_state['input_values']:
            local_state['input_values'][new_name_stripped] = local_state['input_values'].pop(old_name)

        # Quitar de placeholder_fields (ya tiene nombre asignado)
        local_state['placeholder_fields'].discard(old_name)

        # Actualizar expanded_var si era esta variable la expandida
        if local_state.get('expanded_var') == old_name:
            local_state['expanded_var'] = new_name_stripped

        if refresh:
            refresh_data_section()
            refresh_builder()  # Actualizar chips de variables en pasos
        return True

    async def validate_and_update_field_name(old_name: str, new_value: str, input_element):
        """Valida el nombre de variable y muestra feedback visual si hay error."""
        import asyncio
        new_name = new_value.strip()

        # Si está vacío, restaurar nombre actual
        if not new_name:
            input_element.value = old_name
            return

        # Si no hay cambio, no hacer nada
        if new_name == old_name:
            return

        # Verificar si ya existe otra variable con ese nombre
        if new_name in local_state['input_fields']:
            # Mostrar error visual: borde rojo + shake
            input_element.classes(add='input-error shake-error')
            ui.notify(f'Ya existe una variable "{new_name}"', type='negative')

            # Restaurar nombre actual
            input_element.value = old_name

            # Quitar clases de error después de la animación
            await asyncio.sleep(0.5)
            input_element.classes(remove='input-error shake-error')
            return

        # Nombre válido, actualizar
        update_input_field_name(old_name, new_name)

    def add_value(field_name: str):
        """Añade un valor vacío a una variable específica."""
        if field_name in local_state['input_values']:
            local_state['input_values'][field_name].append('')
            refresh_data_section()

    def remove_value(field_name: str, index: int):
        """Elimina un valor de una variable específica."""
        if field_name in local_state['input_values']:
            values = local_state['input_values'][field_name]
            if 0 <= index < len(values):
                values.pop(index)
                refresh_data_section()

    def toggle_var_expand(field_name: str):
        """Expande o colapsa una variable."""
        if local_state.get('expanded_var') == field_name:
            local_state['expanded_var'] = None
        else:
            local_state['expanded_var'] = field_name
        refresh_data_section()

    # ════════════════════════════════════════════════════════════════
    # Funciones para tabla de variables (row-based)
    # ════════════════════════════════════════════════════════════════

    def add_row():
        """Añade una nueva fila (ejecución) a la tabla."""
        fields = local_state['input_fields']
        if not fields:
            return
        for field in fields:
            if field not in local_state['input_values']:
                local_state['input_values'][field] = []
            local_state['input_values'][field].append('')
        refresh_data_section()

    def remove_row(row_idx: int):
        """Elimina una fila específica de la tabla."""
        for field in local_state['input_fields']:
            values = local_state['input_values'].get(field, [])
            if 0 <= row_idx < len(values):
                values.pop(row_idx)
        refresh_data_section()

    def update_cell(field_name: str, row_idx: int, value: str):
        """Actualiza el valor de una celda específica."""
        if field_name in local_state['input_values']:
            values = local_state['input_values'][field_name]
            # Extender la lista si es necesario
            while len(values) <= row_idx:
                values.append('')
            values[row_idx] = value

    def get_row_count() -> int:
        """Obtiene el número de filas (máximo de valores en cualquier campo)."""
        return max((len(vals) for vals in local_state['input_values'].values()), default=0)

    def select_step(index: int):
        """Selecciona un paso para edición."""
        if 0 <= index < len(local_state['steps']):
            local_state['selected_step'] = index
            refresh_builder()

    def move_step_up(index: int):
        """Mueve un paso hacia arriba."""
        if index > 0 and index < len(local_state['steps']):
            local_state['steps'][index], local_state['steps'][index - 1] = \
                local_state['steps'][index - 1], local_state['steps'][index]
            # Actualizar referencias
            for step in local_state['steps']:
                refs = step.get('reference_step_numbers', [])
                new_refs = []
                for r in refs:
                    if r == index:
                        new_refs.append(index - 1)
                    elif r == index - 1:
                        new_refs.append(index)
                    else:
                        new_refs.append(r)
                step['reference_step_numbers'] = new_refs
            local_state['selected_step'] = index - 1
            refresh_builder()

    def move_step_down(index: int):
        """Mueve un paso hacia abajo."""
        if index >= 0 and index < len(local_state['steps']) - 1:
            local_state['steps'][index], local_state['steps'][index + 1] = \
                local_state['steps'][index + 1], local_state['steps'][index]
            # Actualizar referencias
            for step in local_state['steps']:
                refs = step.get('reference_step_numbers', [])
                new_refs = []
                for r in refs:
                    if r == index:
                        new_refs.append(index + 1)
                    elif r == index + 1:
                        new_refs.append(index)
                    else:
                        new_refs.append(r)
                step['reference_step_numbers'] = new_refs
            local_state['selected_step'] = index + 1
            refresh_builder()

    def duplicate_step(index: int):
        """Duplica un paso."""
        if 0 <= index < len(local_state['steps']):
            original = local_state['steps'][index]
            duplicated = copy.deepcopy(original)
            local_state['steps'].insert(index + 1, duplicated)
            local_state['selected_step'] = index + 1
            local_state['detected_vars'] = detect_variables(local_state['steps'])
            refresh_builder()
            refresh_data_section()

    def export_config():
        """Exporta la configuración actual del pipeline."""
        if not local_state['steps']:
            ui.notify('No hay pasos para exportar', type='warning')
            return
        config = {'steps': local_state['steps']}
        ui.download(
            json.dumps(config, indent=2, ensure_ascii=False).encode('utf-8'),
            f'pipeline_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        ui.notify('Configuración exportada', type='positive')

    def on_config_upload(e):
        """Maneja la carga de configuración desde archivo."""
        try:
            content = e.content.read().decode('utf-8')
            data = json.loads(content)
            local_state['steps'] = data.get('steps', [])
            local_state['selected_step'] = 0 if local_state['steps'] else 0
            local_state['detected_vars'] = detect_variables(local_state['steps'])
            refresh_builder()
            refresh_data_section()
            ui.notify(f'{len(local_state["steps"])} paso(s) importados', type='positive')
        except Exception as ex:
            ui.notify(f'Error al importar: {ex}', type='negative')

    def toggle_json_view():
        """Alterna entre vista formulario y JSON."""
        local_state['json_view'] = not local_state['json_view']
        refresh_data_section()

    def toggle_pipeline_json_view():
        """Alterna entre vista constructor y JSON del pipeline."""
        local_state['pipeline_json_view'] = not local_state['pipeline_json_view']
        local_state['pipeline_json_edit_mode'] = False
        local_state['pipeline_json_edit_buffer'] = ''
        refresh_builder()

    def export_pipeline_json():
        """Exporta la configuración del pipeline como JSON."""
        steps = local_state['steps']
        if not steps:
            ui.notify('No hay pasos configurados para exportar', type='warning')
            return
        # Construir configuración completa del pipeline
        pipeline_config = {
            'model': state.model,
            'steps': steps
        }
        ui.download(
            json.dumps(pipeline_config, indent=2, ensure_ascii=False).encode('utf-8'),
            f'pipeline_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        ui.notify('Pipeline exportado', type='positive')

    def update_counters():
        """Actualiza los contadores dinámicos."""
        if steps_counter_label:
            steps_counter_label.set_text(f'{len(local_state["steps"])} pasos configurados')
        # entries_counter_label ya no se usa - el estado está integrado en render_data_section

    def update_value(field_name: str, index: int, value: str):
        """Actualiza un valor de una variable específica."""
        if field_name in local_state['input_values']:
            values = local_state['input_values'][field_name]
            if 0 <= index < len(values):
                values[index] = value

    def load_example_data():
        """Carga datos de ejemplo."""
        data = load_json_file(DEFAULT_PIPELINE_REFERENCE_DATA)
        if data:
            entries = data if isinstance(data, list) else [data]
            local_state['input_vars_enabled'] = True

            # Convertir de formato row-based a column-based
            if entries:
                new_fields = list(entries[0].keys())
                local_state['input_fields'] = new_fields
                # Inicializar input_values con listas vacías
                local_state['input_values'] = {field: [] for field in new_fields}
                # Poblar valores de cada entrada
                for entry in entries:
                    for field in new_fields:
                        local_state['input_values'][field].append(entry.get(field, ''))

            refresh_data_section()
            ui.notify(f'{len(entries)} entradas cargadas', type='positive')

    def render_pipeline_json_view():
        """Renderiza la vista JSON del pipeline con edición."""
        steps = local_state['steps']
        pipeline_config = {'model': state.model, 'steps': steps}
        json_text = json.dumps(pipeline_config, indent=2, ensure_ascii=False)

        def enter_pipeline_edit_mode():
            """Entra en modo edición del pipeline."""
            local_state['pipeline_json_edit_mode'] = True
            local_state['pipeline_json_edit_buffer'] = json_text
            refresh_builder()

        def cancel_pipeline_edit_mode():
            """Cancela la edición del pipeline."""
            local_state['pipeline_json_edit_mode'] = False
            local_state['pipeline_json_edit_buffer'] = ''
            refresh_builder()

        def apply_pipeline_json_changes():
            """Aplica los cambios del JSON al pipeline."""
            try:
                parsed = json.loads(local_state['pipeline_json_edit_buffer'])
                if not isinstance(parsed, dict):
                    ui.notify('El JSON debe ser un objeto con "model" y "steps"', type='warning')
                    return

                # Validar estructura
                if 'steps' not in parsed or not isinstance(parsed['steps'], list):
                    ui.notify('El JSON debe contener una lista "steps"', type='warning')
                    return

                # Actualizar estado
                if 'model' in parsed:
                    state.model = parsed['model']
                local_state['steps'] = parsed['steps']
                local_state['selected_step'] = 0 if parsed['steps'] else -1
                local_state['pipeline_json_edit_mode'] = False
                local_state['pipeline_json_edit_buffer'] = ''

                ui.notify('Pipeline actualizado correctamente', type='positive')
                refresh_builder()

            except json.JSONDecodeError as err:
                ui.notify(f'JSON inválido: {err.msg}', type='negative')

        def copy_pipeline_json():
            """Copia el JSON del pipeline al portapapeles."""
            ui.run_javascript(f'navigator.clipboard.writeText({json.dumps(json_text)})')
            ui.notify('JSON copiado al portapapeles', type='positive')

        def update_pipeline_buffer(e):
            """Actualiza el buffer de edición del pipeline."""
            local_state['pipeline_json_edit_buffer'] = e.value

        with ui.column().classes('w-full flex-1 gap-2'):
            if not local_state['pipeline_json_edit_mode']:
                # MODO VISTA (solo lectura)
                with ui.row().classes('w-full items-center justify-start gap-2'):
                    with ui.button(on_click=copy_pipeline_json).props('flat dense no-caps').classes(
                        'px-2 py-1 text-slate-200 hover:text-white hover:bg-slate-700/50 rounded'
                    ):
                        ui.icon('content_copy', size='xs').classes('mr-1 text-slate-200')
                        ui.label('Copiar').classes('text-xs text-slate-200')
                    with ui.button(on_click=enter_pipeline_edit_mode).props('flat dense no-caps').classes(
                        'px-2 py-1 text-slate-200 hover:text-white hover:bg-slate-700/50 rounded'
                    ):
                        ui.icon('edit', size='xs').classes('mr-1 text-slate-200')
                        ui.label('Editar').classes('text-xs text-slate-200')

                ui.textarea(
                    value=json_text,
                ).props('outlined dark readonly').classes(
                    'w-full flex-1 font-mono text-sm input-subtle json-editor-textarea'
                ).style('min-height: 280px')

            else:
                # MODO EDICIÓN
                with ui.row().classes('w-full items-center gap-2'):
                    ui.icon('edit_note', size='xs').classes('text-slate-400')
                    ui.label('Modo edición: Los cambios reemplazarán la configuración actual del pipeline').classes('text-xs text-slate-400')
                    ui.element('div').classes('flex-1')
                    with ui.button(on_click=cancel_pipeline_edit_mode).props('flat dense no-caps').classes(
                        'px-2 py-1 text-slate-200 hover:text-white hover:bg-slate-700/50 rounded'
                    ):
                        ui.icon('close', size='xs').classes('mr-1 text-slate-200')
                        ui.label('Cancelar').classes('text-xs text-slate-200')
                    with ui.button(on_click=apply_pipeline_json_changes).props('flat dense no-caps').classes(
                        'px-2 py-1 bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 rounded'
                    ):
                        ui.icon('check', size='xs').classes('mr-1 text-purple-300')
                        ui.label('Aplicar cambios').classes('text-xs text-purple-300')

                ui.textarea(
                    value=local_state['pipeline_json_edit_buffer'],
                    on_change=update_pipeline_buffer
                ).props('outlined dark').classes(
                    'w-full flex-1 font-mono text-sm input-subtle json-editor-textarea'
                ).style('min-height: 280px')

    def refresh_builder():
        """Refresca el constructor de pasos con accordion o vista JSON."""
        nonlocal steps_list_container
        if steps_list_container:
            steps_list_container.clear()
            with steps_list_container:
                if local_state['pipeline_json_view'] and len(local_state['steps']) > 0:
                    render_pipeline_json_view()
                else:
                    render_steps_accordion()
        update_counters()
        # Actualizar checklist de ejecución si existe
        if 'update_execution_checklist' in local_state and local_state['update_execution_checklist']:
            local_state['update_execution_checklist']()

    def refresh_data_section():
        """Refresca la sección de datos."""
        data_container.clear()
        with data_container:
            render_data_section()
        update_counters()
        # Actualizar checklist de ejecución si existe
        if 'update_execution_checklist' in local_state and local_state['update_execution_checklist']:
            local_state['update_execution_checklist']()

    def render_steps_accordion():
        """Renderiza los pasos como barras desplegables (accordion)."""
        step_colors = {
            'generate': ('purple', 'auto_awesome', 'Generar'),
            'parse': ('purple', 'find_in_page', 'Parsear'),
            'verify': ('purple', 'verified', 'Verificar')
        }

        # Botones para añadir pasos con etiqueta clara
        with ui.row().classes('w-full items-center gap-3'):
            with ui.button(on_click=lambda: add_step('generate')).props('flat dense no-caps').classes(
                'h-8 pl-2 pr-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                'hover:bg-purple-500/20 hover:border-purple-500/50 transition-all flex items-center'
            ):
                ui.icon('add', size='xs').classes('text-purple-400 mr-1.5')
                ui.label('GENERAR').classes('text-xs text-slate-300 leading-none uppercase')

            # Parse step está integrado en Generate, pero el botón se mantiene oculto para
            # poder cargar pipelines antiguos con pasos parse separados
            # with ui.button(on_click=lambda: add_step('parse'))...

            with ui.button(on_click=lambda: add_step('verify')).props('flat dense no-caps').classes(
                'h-8 pl-2 pr-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                'hover:bg-purple-500/20 hover:border-purple-500/50 transition-all flex items-center'
            ):
                ui.icon('add', size='xs').classes('text-purple-400 mr-1.5')
                ui.label('VERIFICAR').classes('text-xs text-slate-300 leading-none uppercase')

        if not local_state['steps']:
            with ui.column().classes('w-full items-center justify-center py-8 gap-2'):
                ui.icon('touch_app', size='lg').classes('text-slate-600')
                ui.label('Sin pasos configurados').classes('text-slate-400 text-sm')
                ui.label('Añade pasos con los botones de generar y verificar').classes('text-xs text-slate-600')
        else:
            for idx, step in enumerate(local_state['steps']):
                stype = step.get('type', 'generate')
                color, icon, type_name = step_colors.get(stype, ('slate', 'help', 'Paso'))
                params = step.get('parameters', {})
                is_expanded = local_state.get('expanded_step') == idx

                # Resumen de datos relevantes según tipo
                if stype == 'generate':
                    summary = f"Temp: {params.get('temperature', 0.7)} · Max: {params.get('max_tokens', 200)} tokens"
                elif stype == 'parse':
                    rules_count = len(params.get('rules', []))
                    summary = f"{rules_count} regla{'s' if rules_count != 1 else ''} de parseo"
                elif stype == 'verify':
                    methods_count = len(params.get('methods', []))
                    summary = f"{methods_count} método{'s' if methods_count != 1 else ''} de verificación"
                else:
                    summary = ""

                # Barra del paso (fondo neutro + línea de color lateral)
                with ui.column().classes('w-full gap-0'):
                    # Header de la barra (siempre visible)
                    with ui.element('div').classes(
                        f'w-full bg-slate-800/60 border border-slate-600/40 rounded-lg '
                        f'{"rounded-b-none border-b-0" if is_expanded else ""} '
                        'cursor-pointer hover:bg-purple-500/10 hover:border-purple-500/50 transition-all'
                    ).on('click', lambda i=idx: toggle_step_expand(i)):
                        with ui.row().classes('w-full items-center justify-between px-3 py-2'):
                            # Lado izquierdo: número, tipo, resumen
                            with ui.row().classes('items-center gap-3 flex-1'):
                                # Número del paso
                                with ui.element('div').classes('w-7 h-7 rounded-full bg-slate-700/80 flex items-center justify-center'):
                                    ui.label(str(idx + 1)).classes('text-sm font-bold text-slate-300')
                                # Icono y tipo
                                _, step_icon, _ = step_colors.get(stype, ('slate', 'help', 'Paso'))
                                with ui.row().classes('items-center gap-1.5 w-32'):
                                    ui.icon(step_icon, size='xs').classes('text-slate-400')
                                    ui.label(type_name).classes('text-sm font-semibold text-slate-200 uppercase')
                                # Separador
                                ui.element('div').classes('w-px h-4 bg-slate-600/50')
                                # Resumen
                                ui.label(summary).classes('text-xs text-slate-400 ml-2')

                            # Lado derecho: botones de acción + expand
                            with ui.row().classes('items-center gap-1'):
                                ui.button(icon='arrow_upward').props('flat round size=xs color=none').classes('!text-purple-400/60 hover:!text-purple-300').tooltip('Subir').on('click.stop', lambda i=idx: move_step_up(i))
                                ui.button(icon='arrow_downward').props('flat round size=xs color=none').classes('!text-purple-400/60 hover:!text-purple-300').tooltip('Bajar').on('click.stop', lambda i=idx: move_step_down(i))
                                ui.button(icon='content_copy').props('flat round size=xs color=none').classes('!text-purple-400/60 hover:!text-purple-300').tooltip('Duplicar').on('click.stop', lambda i=idx: duplicate_step(i))
                                ui.button(icon='delete').props('flat round size=xs color=none').classes('!text-purple-400/60 hover:!text-purple-300').tooltip('Eliminar').on('click.stop', lambda i=idx: remove_step(i))
                                # Icono expandir/colapsar
                                ui.element('div').classes('w-px h-4 bg-slate-600/50 mx-1')
                                ui.icon('expand_more' if not is_expanded else 'expand_less', size='sm').classes('text-purple-400')

                    # Contenido expandible
                    if is_expanded:
                        with ui.element('div').classes(
                            'w-full bg-slate-900/50 border border-slate-600/40 border-t-0 '
                            'rounded-b-lg p-4'
                        ):
                            if stype == 'generate':
                                render_generate_config(idx, step)
                            elif stype == 'parse':
                                render_parse_config(idx, step)
                            elif stype == 'verify':
                                render_verify_config(idx, step)

    def toggle_step_expand(idx: int):
        """Expande o colapsa un paso."""
        if local_state.get('expanded_step') == idx:
            local_state['expanded_step'] = None
        else:
            local_state['expanded_step'] = idx
        refresh_builder()

    def render_step_config():
        """Renderiza el panel de configuración del paso seleccionado (columna derecha)."""
        if not local_state['steps']:
            with ui.column().classes('w-full items-center justify-center py-12 gap-3'):
                ui.icon('touch_app', size='xl').classes('text-slate-600')
                ui.label('Selecciona una plantilla o añade un paso').classes('text-slate-400')
            return

        idx = local_state['selected_step']
        if idx >= len(local_state['steps']):
            idx = len(local_state['steps']) - 1
            local_state['selected_step'] = idx

        step = local_state['steps'][idx]
        stype = step.get('type', 'generate')

        step_colors = {
            'generate': ('purple', 'auto_awesome', 'Generar'),
            'parse': ('purple', 'find_in_page', 'Parsear'),
            'verify': ('purple', 'verified', 'Verificar')
        }
        color, icon, title = step_colors.get(stype, ('slate', 'help', 'Paso'))

        # Header del panel
        with ui.row().classes('w-full items-center gap-2 mb-4'):
            ui.icon(icon, size='sm').classes(f'text-{color}-400')
            ui.label(f'Paso {idx + 1}: {title}').classes(f'text-lg font-semibold text-{color}-300')

        # Configuración según tipo
        if stype == 'generate':
            render_generate_config(idx, step)
        elif stype == 'parse':
            render_parse_config(idx, step)
        elif stype == 'verify':
            render_verify_config(idx, step)

    def render_generate_config(idx: int, step: Dict):
        """Renderiza configuración de paso generate."""
        params = step.get('parameters', {})

        def get_step_output_variables(step_data: Dict, step_idx: int) -> List[str]:
            """Retorna lista de nombres de variables que produce un paso."""
            step_type = step_data.get('type', 'generate')
            if step_type == 'generate':
                vars_list = [f'output_{step_idx + 1}']
                # Incluir variables de parse_rules integradas
                parse_rules = step_data.get('parameters', {}).get('parse_rules', [])
                for rule in parse_rules:
                    if rule.get('name'):
                        vars_list.append(rule['name'])
                return vars_list
            elif step_type == 'parse':
                rules = step_data.get('parameters', {}).get('rules', [])
                return [rule['name'] for rule in rules if rule.get('name')]
            elif step_type == 'verify':
                return ['status', 'details']
            return []

        # Obtener variables disponibles (entrada + pasos referenciados)
        def get_available_variables() -> List[Dict[str, str]]:
            """Retorna lista de variables disponibles (entrada + pasos referenciados)."""
            variables = []

            # Variables de entrada (si están habilitadas)
            if local_state['input_vars_enabled'] and local_state['input_fields']:
                for field in local_state['input_fields']:
                    variables.append({'name': field, 'step': -1, 'type': 'input'})

            # Variables de pasos referenciados
            current_refs = step.get('reference_step_numbers', [])
            for ref_idx in current_refs:
                if 0 <= ref_idx < len(local_state['steps']):
                    ref_step = local_state['steps'][ref_idx]
                    ref_type = ref_step.get('type', 'generate')
                    if ref_type == 'generate':
                        variables.append({'name': f'output_{ref_idx + 1}', 'step': ref_idx, 'type': 'generate'})
                    elif ref_type == 'parse':
                        rules = ref_step.get('parameters', {}).get('rules', [])
                        for rule in rules:
                            if rule.get('name'):
                                variables.append({'name': rule['name'], 'step': ref_idx, 'type': 'parse'})
                    elif ref_type == 'verify':
                        variables.append({'name': 'status', 'step': ref_idx, 'type': 'verify'})
                        variables.append({'name': 'details', 'step': ref_idx, 'type': 'verify'})
            return variables

        def render_variable_chips(textarea_element, field: str):
            """Renderiza los chips de variables clickables que insertan en la posición del cursor."""
            variables = get_available_variables()
            if not variables:
                return

            with ui.row().classes('w-full flex-wrap items-center gap-1.5 px-2 py-1.5 bg-slate-800/40 rounded-b border border-t-0 border-slate-600/50'):
                # Label explicativo
                ui.icon('add_link', size='xs').classes('text-slate-400')
                ui.label('Insertar:').classes('text-xs text-slate-400 mr-1')

                for var in variables:
                    var_text = f'{{{var["name"]}}}'
                    type_names = {'generate': 'Generado', 'parse': 'Detectado', 'verify': 'Verificado', 'input': 'Entrada'}
                    color = 'purple'

                    # Nombre descriptivo según tipo
                    if var['type'] == 'input':
                        display_name = f'{{{var["name"]}}}'
                        tooltip_text = f'Variable de entrada. Click para insertar {{{var["name"]}}}'
                    else:
                        step_num = var['step'] + 1
                        display_name = f'Paso {step_num}: {{{var["name"]}}}'
                        tooltip_text = f'{type_names.get(var["type"], "Output")} del paso {step_num}. Click para insertar la variable'

                    async def insert_var_at_cursor(v=var_text, fld=field, i=idx, ta=textarea_element):
                        # Usar el ID de NiceGUI del textarea para insertar en la posición del cursor
                        nicegui_id = ta.id
                        js_code = f'''
                        (function() {{
                            // En NiceGUI, el ID está directamente en el textarea
                            const textarea = document.getElementById('c{nicegui_id}');
                            if (!textarea) {{
                                console.error('Textarea c{nicegui_id} not found');
                                return null;
                            }}
                            const start = textarea.selectionStart || 0;
                            const end = textarea.selectionEnd || 0;
                            const text = textarea.value || '';
                            const varText = '{v}';
                            const newText = text.substring(0, start) + varText + text.substring(end);
                            textarea.value = newText;
                            const newCursorPos = start + varText.length;
                            textarea.selectionStart = textarea.selectionEnd = newCursorPos;
                            textarea.focus();
                            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            return newText;
                        }})()
                        '''
                        result = await ui.run_javascript(js_code)
                        if result is not None:
                            update_step_param(i, fld, result)

                    with ui.button(on_click=insert_var_at_cursor).props('flat dense no-caps color=none').classes(
                        'px-2 py-0.5 min-h-0 hover:bg-slate-700/50 transition-all'
                    ):
                        ui.label(display_name).classes('font-mono text-xs text-slate-200 normal-case')
                        ui.tooltip(tooltip_text).classes('bg-slate-800 text-slate-200')

        with ui.column().classes('w-full gap-3'):
            # === SECCIÓN: PROMPTS ===
            with ui.column().classes('w-full gap-2'):
                # Inicializar referencias para este paso
                if idx not in field_refs:
                    field_refs[idx] = {}

                # System Prompt
                with ui.column().classes('w-full gap-0'):
                    ui.label('System Prompt').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Define el rol y comportamiento del modelo')

                    # Contenedor para textarea + chips (textarea primero para obtener su referencia)
                    has_vars = bool(get_available_variables())
                    with ui.column().classes('w-full gap-0 mt-1'):
                        system_textarea = ui.textarea(
                            value=params.get('system_prompt', ''),
                            placeholder='Eres un asistente capaz de...',
                            on_change=lambda e, i=idx: update_step_param(i, 'system_prompt', e.value)
                        ).props('outlined dark dense rows=1 autogrow').classes('w-full input-subtle' + (' rounded-b-none' if has_vars else ''))
                        field_refs[idx]['system_prompt'] = system_textarea
                        render_variable_chips(system_textarea, 'system_prompt')

                # User Prompt
                with ui.column().classes('w-full gap-0 mt-2'):
                    ui.label('User Prompt').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Instrucciones específicas para cada entrada')

                    # Contenedor para textarea + chips
                    has_vars = bool(get_available_variables())
                    with ui.column().classes('w-full gap-0 mt-1'):
                        user_textarea = ui.textarea(
                            value=params.get('user_prompt', ''),
                            placeholder='Analiza el siguiente texto: {texto}',
                            on_change=lambda e, i=idx: update_step_param(i, 'user_prompt', e.value)
                        ).props('outlined dark dense rows=1 autogrow').classes('w-full input-subtle' + (' rounded-b-none' if has_vars else ''))
                        field_refs[idx]['user_prompt'] = user_textarea
                        render_variable_chips(user_textarea, 'user_prompt')

            # === SECCIÓN: REFERENCIAS + PARÁMETROS (en fila) ===
            ui.element('div').classes('w-full h-px bg-slate-700/50 my-1')

            with ui.row().classes('w-full gap-6 items-start'):
                # --- COLUMNA IZQUIERDA: REFERENCIAS ---
                with ui.column().classes('flex-1 gap-2'):
                    # === REFERENCIAS A PASOS ANTERIORES ===
                    with ui.row().classes('items-center gap-1.5'):
                        ui.icon('data_object').classes('text-slate-400 text-sm -mt-0.5')
                        ui.label('Referenciar outputs de pasos anteriores').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Selecciona pasos anteriores para usar sus outputs como variables en tus prompts.\nPodrás insertar su valor en el prompt pulsando el botón "Insertar" que aparecerá bajo el campo de texto del prompt.')

                    if idx > 0:
                        current_refs = step.get('reference_step_numbers', [])
                        with ui.column().classes('w-full gap-0'):
                            for prev_idx in range(idx):
                                prev_step = local_state['steps'][prev_idx]
                                prev_type = prev_step.get('type', 'generate')
                                type_icons = {'generate': 'auto_awesome', 'parse': 'find_in_page', 'verify': 'verified'}
                                type_names = {'generate': 'Generar', 'parse': '', 'verify': 'Verificar'}
                                is_checked = prev_idx in current_refs

                                # Obtener variables que produce este paso
                                step_vars = get_step_output_variables(prev_step, prev_idx)
                                vars_display = ', '.join(f'{{{v}}}' for v in step_vars) if step_vars else '(sin variables establecidas)'

                                with ui.row().classes('w-full items-center gap-0.5 py-1 ml-2'):
                                    cb = ui.checkbox(
                                        value=is_checked,
                                        on_change=lambda e, i=idx, r=prev_idx: toggle_reference_step(i, r, e.value)
                                    ).props('dense size=sm')
                                    ui.label(f'Paso {prev_idx + 1}:').classes('text-sm text-slate-400 cursor-pointer select-none ml-0.5').on('click', lambda _, c=cb: c.set_value(not c.value))
                                    ui.label(type_names.get(prev_type, prev_type)).classes('text-sm text-slate-400 cursor-pointer select-none').on('click', lambda _, c=cb: c.set_value(not c.value))
                                    ui.label('→').classes('text-slate-500 text-xs mx-0.5')
                                    vars_classes = 'text-xs text-slate-500 font-mono' if step_vars else 'text-xs text-slate-500 italic'
                                    ui.label(vars_display).classes(vars_classes)
                    else:
                        ui.label('(primer paso, sin referencias disponibles)').classes('text-xs text-slate-400 italic mt-1 ml-2')

                # --- COLUMNA DERECHA: PARÁMETROS ---
                with ui.column().classes('flex-1 gap-2'):
                    with ui.row().classes('items-center gap-1.5'):
                        ui.icon('tune').classes('text-slate-400 text-sm')
                        ui.label('Parámetros').classes('text-xs font-medium text-slate-300 uppercase')

                    # Temperatura, Max Tokens y Secuencias en una fila
                    with ui.row().classes('w-full gap-4 items-start'):
                        # Temperatura
                        with ui.column().classes('gap-1 flex-1'):
                            with ui.row().classes('w-full justify-between items-center'):
                                ui.label('Temperatura').classes('text-xs text-slate-400 cursor-help').tooltip('Controla la creatividad del modelo.\n0 = Respuestas más predecibles\n2 = Respuestas más variadas')
                                temp_label = ui.label(f'{params.get("temperature", 0.7):.1f}').classes('text-xs text-slate-300 font-mono')

                            def on_temp_change(e, i=idx):
                                temp_label.text = f'{e.value:.1f}'
                                update_step_param(i, 'temperature', float(e.value))

                            ui.slider(
                                value=params.get('temperature', 0.7),
                                min=0, max=2, step=0.1,
                                on_change=on_temp_change
                            ).classes('w-full temp-slider')
                            with ui.row().classes('w-full justify-between -mt-1'):
                                ui.label('Preciso').classes('text-[10px] text-slate-500')
                                ui.label('Creativo').classes('text-[10px] text-slate-500')

                        # Max Tokens
                        with ui.column().classes('gap-1 w-21 ml-4'):
                            ui.label('Max Tokens').classes('text-xs text-slate-400 cursor-help').tooltip('Longitud máxima de la respuesta generada (1-4096)')
                            ui.number(
                                value=params.get('max_tokens', 200),
                                min=1, max=4096,
                                on_change=lambda e, i=idx: update_step_param(i, 'max_tokens', int(e.value) if e.value else 200)
                            ).props('outlined dark dense color=purple input-class="text-center !py-1"').classes('w-full')

                        # Secuencias
                        with ui.column().classes('gap-1 w-20'):
                            ui.label('Secuencias').classes('text-xs text-slate-400 cursor-help').tooltip('Número de variaciones a generar por cada entrada (1-10)')
                            ui.number(
                                value=params.get('num_sequences', 1),
                                min=1, max=10,
                                on_change=lambda e, i=idx: update_step_param(i, 'num_sequences', int(e.value) if e.value else 1)
                            ).props('outlined dark dense color=purple input-class="text-center !py-1"').classes('w-full')

            # === SECCIÓN: CREAR VARIABLES ===
            ui.element('div').classes('w-full h-px bg-slate-700/50 my-1')

            with ui.column().classes('w-full gap-2'):
                has_parsing = bool(params.get('parse_rules'))

                with ui.row().classes('w-full items-center gap-1.5'):
                    ui.switch(
                        value=has_parsing,
                        on_change=lambda e, i=idx: toggle_generate_parsing(i, e.value)
                    ).props('dense color=purple')
                    ui.label('Crear variables a partir del texto generado').classes('text-xs font-medium text-slate-300 uppercase mt-px cursor-help').tooltip('Define patrones para crear variables a partir del texto generado.\nLas variables creadas estarán disponibles para pasos siguientes.')

                if has_parsing:
                    ui.element('div').classes('h-1')  # Espacio entre toggle y reglas
                    parse_rules = params.get('parse_rules', [])

                    for r_idx, rule in enumerate(parse_rules):
                        # Separador entre reglas
                        if r_idx > 0:
                            ui.element('div').classes('w-full h-px bg-slate-700/50 my-2')

                        # Layout: número (con X al hover) | campos
                        with ui.row().classes('w-full gap-3'):
                            # Número a la izquierda (se convierte en X al hover), centrado verticalmente
                            with ui.element('div').classes('w-5 flex items-center justify-center relative group/rule self-center'):
                                ui.label(f'#{r_idx + 1}').classes('text-xs text-slate-500 group-hover/rule:opacity-0 transition-opacity')
                                ui.button(
                                    icon='close',
                                    on_click=lambda si=idx, ri=r_idx: remove_generate_parse_rule(si, ri)
                                ).props('flat round size=xs color=none').classes(
                                    '!text-slate-500 hover:!text-red-400 absolute opacity-0 group-hover/rule:opacity-100 transition-opacity'
                                )

                            # Campos
                            with ui.column().classes('flex-1 gap-2'):
                                with ui.row().classes('w-full gap-2'):
                                    ui.input(
                                        label='Nombre de variable',
                                        value=rule.get('name', ''),
                                        on_change=lambda e, si=idx, ri=r_idx: update_generate_parse_rule(si, ri, 'name', e.value)
                                    ).props('outlined dark dense color=purple').classes('w-44 input-subtle').tooltip('Nombre de la variable donde se guardará el dato extraído')

                                    ui.select(
                                        label='Modo',
                                        options=['KEYWORD', 'REGEX'],
                                        value=rule.get('mode', 'KEYWORD'),
                                        on_change=lambda e, si=idx, ri=r_idx: update_generate_parse_rule(si, ri, 'mode', e.value)
                                    ).props('outlined dark dense color=purple options-dark popup-content-class="!bg-slate-800"').classes('w-32 input-subtle').tooltip('KEYWORD: extrae texto entre dos delimitadores\nREGEX: captura con expresión regular')

                                    ui.input(
                                        label='Fallback',
                                        value=rule.get('fallback_value', ''),
                                        on_change=lambda e, si=idx, ri=r_idx: update_generate_parse_rule(si, ri, 'fallback_value', e.value)
                                    ).props('outlined dark dense color=purple').classes('w-28 input-subtle').tooltip('Valor por defecto si no se encuentran coincidencias')

                                with ui.row().classes('w-full gap-2'):
                                    is_keyword_mode = rule.get('mode', 'KEYWORD') == 'KEYWORD'
                                    ui.input(
                                        label='Extraer desde' if is_keyword_mode else 'Patrón regex',
                                        value=rule.get('pattern', ''),
                                        on_change=lambda e, si=idx, ri=r_idx: update_generate_parse_rule(si, ri, 'pattern', e.value)
                                    ).props('outlined dark dense color=purple').classes('flex-1 input-subtle').tooltip('Delimitador inicial (se excluye del resultado).\nExtrae lo que viene DESPUÉS de este texto.' if is_keyword_mode else 'Expresión regular para capturar el dato')

                                    if is_keyword_mode:
                                        ui.input(
                                            label='Hasta',
                                            value=rule.get('secondary_pattern', ''),
                                            on_change=lambda e, si=idx, ri=r_idx: update_generate_parse_rule(si, ri, 'secondary_pattern', e.value)
                                        ).props('outlined dark dense color=purple').classes('flex-1 input-subtle').tooltip('Delimitador final (se excluye del resultado).\nExtrae hasta ANTES de este texto.\nSi está vacío, extrae hasta el final.')

                    # Fila con botón añadir y filtro
                    with ui.row().classes('w-full items-center justify-between'):
                        # Botón añadir regla (icono + texto clickable)
                        ui.button('VARIABLE', icon='add', on_click=lambda i=idx: add_generate_parse_rule(i)).props('flat dense size=md color=none').classes('!text-purple-400 hover:!text-purple-300 !px-1 uppercase')

                        # Toggle de filtro (solo mostrar si hay más de una regla)
                        if len(parse_rules) > 1:
                            current_filter = params.get('parse_output_filter', 'all')
                            is_strict = current_filter == 'successful'

                            def on_strict_toggle(e, i=idx):
                                new_filter = 'successful' if e.value else 'all'
                                update_step_param(i, 'parse_output_filter', new_filter)

                            with ui.row().classes('items-center gap-2'):
                                ui.switch(
                                    value=is_strict,
                                    on_change=on_strict_toggle
                                ).props('dense size=sm color=purple')
                                ui.label('Solo completos').classes('text-xs text-slate-400 cursor-help').tooltip('Activado: solo incluye resultados donde TODAS las variables se extrajeron correctamente.\nDesactivado: incluye todos los resultados, incluso parciales.')

            # === SECCIÓN: MODELO ESPECÍFICO ===
            ui.element('div').classes('w-full h-px bg-slate-700/50 my-1')

            with ui.column().classes('w-full gap-2'):
                has_custom_model = step.get('llm_config') is not None

                def on_model_toggle(e, i=idx):
                    if e.value:
                        update_step_field(i, 'llm_config', '')
                    else:
                        update_step_field(i, 'llm_config', None)
                    refresh_builder()

                with ui.row().classes('w-full items-center gap-1.5'):
                    ui.switch(value=has_custom_model, on_change=on_model_toggle).props('dense color=purple')
                    ui.label('Usar modelo específico para este paso').classes('text-xs font-medium text-slate-300 uppercase mt-px cursor-help').tooltip('Puedes usar un modelo específico diferente al LLM seleccionado general para este paso')

                if has_custom_model:
                    ui.element('div').classes('h-1')  # Espacio entre toggle y selector
                    current_model = step.get('llm_config', '')
                    step_model_info = step.get('llm_config_info', {})

                    # Estado del selector para este paso
                    step_model_state = {'editing': not bool(current_model)}
                    step_search_state = {'last_query': '', 'results': []}
                    step_search_debounce = {'task': None}
                    step_model_refs = {'input': None, 'dropdown': None, 'indicator': None}

                    # Contenedor del selector
                    step_model_container = ui.column().classes('w-full')

                    # Diálogo para modelo local del paso
                    step_local_dialog = ui.dialog()
                    step_local_path_input = {'ref': None}

                    def step_apply_local_model(i=idx):
                        if step_local_path_input['ref'] and step_local_path_input['ref'].value:
                            path = step_local_path_input['ref'].value.strip()
                            if path:
                                update_step_field(i, 'llm_config', path)
                                update_step_field(i, 'llm_config_info', None)
                                ui.notify(f'Modelo local: {path}', type='positive')
                                step_local_dialog.close()
                                refresh_builder()

                    with step_local_dialog:
                        with ui.card().classes('w-[500px] bg-slate-800/95 border border-purple-500/30 p-4'):
                            with ui.row().classes('items-center gap-2 mb-3'):
                                with ui.element('div').classes('w-7 h-7 rounded-lg bg-purple-500/20 flex items-center justify-center'):
                                    ui.icon('folder_open', size='xs').classes('text-purple-400')
                                ui.label('Cargar Modelo Local').classes('text-base font-semibold text-purple-200')

                            def step_browse_folder():
                                try:
                                    import tkinter as tk
                                    from tkinter import filedialog
                                    root = tk.Tk()
                                    root.withdraw()
                                    root.attributes('-topmost', True)
                                    folder = filedialog.askdirectory(title='Seleccionar carpeta del modelo')
                                    root.destroy()
                                    if folder and step_local_path_input['ref']:
                                        step_local_path_input['ref'].value = folder
                                except Exception:
                                    ui.notify('Copia la ruta manualmente', type='info')

                            ui.label('Ruta completa a la carpeta del modelo:').classes('text-sm text-slate-300')
                            with ui.row().classes('w-full gap-2 mb-2 items-center'):
                                local_input = ui.input(
                                    placeholder='C:/modelos/mi-modelo o /home/user/modelos/llama'
                                ).props('dense dark outlined color=purple').classes('flex-1')
                                step_local_path_input['ref'] = local_input
                                ui.button(icon='folder_open', on_click=step_browse_folder).props('flat color=none').classes('!text-purple-400 hover:!text-purple-300').tooltip('Seleccionar carpeta')

                            ui.label('La carpeta debe contener config.json, archivos del tokenizer y pesos del modelo').classes('text-xs text-slate-400 mb-3')

                            with ui.row().classes('justify-end gap-2'):
                                ui.button('Cancelar', on_click=step_local_dialog.close).props('flat color=none').classes('!text-slate-400')
                                ui.button('Aplicar', on_click=step_apply_local_model).props('flat color=none').classes('!bg-purple-500/20 !text-purple-300')

                    def step_render_dropdown_items(results: List[Dict], i=idx):
                        dropdown = step_model_refs['dropdown']
                        if not dropdown:
                            return
                        step_search_state['results'] = results
                        dropdown.clear()
                        with dropdown:
                            for r in results:
                                def make_click(val=r['value'], info=r, step_i=i):
                                    def click():
                                        update_step_field(step_i, 'llm_config', val)
                                        update_step_field(step_i, 'llm_config_info', info)
                                        dropdown.set_visibility(False)
                                        refresh_builder()
                                    return click

                                compat = r.get('compat_status')
                                if compat == 'compatible':
                                    bg_hover = 'hover:bg-emerald-500/15'
                                    badge_class = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                                    badge_text = 'Compatible'
                                elif compat == 'limite':
                                    bg_hover = 'hover:bg-amber-500/15'
                                    badge_class = 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                                    badge_text = 'Al límite'
                                elif compat == 'incompatible':
                                    bg_hover = 'hover:bg-red-500/15'
                                    badge_class = 'bg-red-500/20 text-red-300 border-red-500/30'
                                    badge_text = 'Incompatible'
                                else:
                                    bg_hover = 'hover:bg-slate-600/30'
                                    badge_class = 'bg-slate-600/30 text-slate-400 border-slate-500/30'
                                    badge_text = '?'

                                model_id = r.get('model_id', '')
                                with ui.element('div').classes(
                                    f'w-full px-3 py-2 cursor-pointer {bg_hover} transition-all border-b border-slate-700/30'
                                ).on('click', make_click()):
                                    with ui.row().classes('items-center justify-between w-full gap-2'):
                                        ui.label(model_id).classes('text-sm font-medium text-slate-100 truncate flex-1 min-w-0')
                                        with ui.element('span').classes(f'px-2 py-0.5 text-xs rounded border flex-shrink-0 {badge_class}'):
                                            ui.label(badge_text)
                                    with ui.row().classes('items-center gap-3 mt-1'):
                                        ui.label(f"Params: {r.get('params_str', '?')}").classes('text-xs text-slate-400')
                                        ui.label(f"VRAM: ~{r.get('vram_str', '?')}").classes('text-xs text-slate-400')
                        dropdown.set_visibility(True)

                    async def step_do_search(query: str, i=idx):
                        dropdown = step_model_refs['dropdown']
                        indicator = step_model_refs['indicator']
                        if not dropdown:
                            return
                        if len(query) < 2:
                            dropdown.set_visibility(False)
                            return

                        if indicator:
                            indicator.set_visibility(True)
                        dropdown.clear()
                        with dropdown:
                            with ui.row().classes('items-center gap-2 p-3'):
                                ui.spinner('dots', size='xs').classes('text-indigo-400')
                                ui.label('Buscando...').classes('text-sm text-slate-400')
                        dropdown.set_visibility(True)

                        try:
                            if query != step_search_state['last_query']:
                                return
                            results = await search_huggingface_models(query)
                            if query == step_search_state['last_query']:
                                if results:
                                    step_render_dropdown_items(results, i)
                                else:
                                    dropdown.clear()
                                    with dropdown:
                                        ui.label('Sin resultados').classes('text-sm text-slate-400 p-3')
                                    dropdown.set_visibility(True)
                        except Exception:
                            pass
                        finally:
                            if indicator:
                                indicator.set_visibility(False)

                    async def step_on_search_debounced(i=idx):
                        inp = step_model_refs['input']
                        dropdown = step_model_refs['dropdown']
                        if not inp or not dropdown:
                            return
                        query = inp.value.strip() if inp.value else ''
                        step_search_state['last_query'] = query

                        if step_search_debounce['task'] and not step_search_debounce['task'].done():
                            step_search_debounce['task'].cancel()

                        if len(query) < 2:
                            dropdown.set_visibility(False)
                            return

                        async def debounced():
                            try:
                                await asyncio.sleep(0.3)
                                if query == step_search_state['last_query']:
                                    await step_do_search(query, i)
                            except asyncio.CancelledError:
                                pass

                        step_search_debounce['task'] = asyncio.create_task(debounced())

                    def render_step_model_selector(i=idx):
                        step_model_container.clear()
                        with step_model_container:
                            current = local_state['steps'][i].get('llm_config', '')
                            model_info = local_state['steps'][i].get('llm_config_info', {})

                            if step_model_state['editing'] or not current:
                                # Vista de edición: buscador + botón local
                                with ui.row().classes('w-1/2 items-stretch gap-2'):
                                    with ui.element('div').classes('flex-1 relative'):
                                        with ui.element('div').classes(
                                            'w-full h-10 flex items-center gap-2 px-3 bg-slate-900/50 '
                                            'border border-slate-600/50 rounded-lg hover:border-purple-500/50 '
                                            'focus-within:border-purple-500 transition-all'
                                        ):
                                            ui.icon('search', size='xs').classes('text-slate-400')
                                            model_input = ui.input(
                                                value=current,
                                                placeholder='Buscar en HuggingFace...'
                                            ).props('borderless dense dark').classes('flex-1 text-sm')
                                            step_model_refs['input'] = model_input
                                            indicator = ui.spinner('dots', size='sm').classes('text-purple-400')
                                            indicator.set_visibility(False)
                                            step_model_refs['indicator'] = indicator

                                        # Dropdown de resultados
                                        dropdown = ui.column().classes(
                                            'absolute top-full left-0 right-0 w-full mt-1 bg-slate-800/95 backdrop-blur-sm '
                                            'border border-slate-600/50 rounded-lg shadow-2xl z-50 max-h-60 overflow-y-auto'
                                        )
                                        dropdown.set_visibility(False)
                                        step_model_refs['dropdown'] = dropdown

                                        async def on_blur():
                                            await asyncio.sleep(0.15)
                                            if step_model_refs['dropdown']:
                                                step_model_refs['dropdown'].set_visibility(False)
                                        model_input.on('blur', on_blur)

                                        async def on_focus():
                                            inp = step_model_refs['input']
                                            if inp and inp.value and len(inp.value.strip()) >= 2:
                                                await step_on_search_debounced(i)
                                        model_input.on('focus', on_focus)

                                        def on_change(e, step_i=i):
                                            if e.args:
                                                update_step_field(step_i, 'llm_config', e.args)
                                        model_input.on('update:model-value', on_change)
                                        model_input.on('keyup', lambda: step_on_search_debounced(i))

                                    ui.label('o').classes('text-sm text-slate-400 self-center')

                                    with ui.button(on_click=step_local_dialog.open).props('flat dense').classes(
                                        'h-10 px-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                                        'hover:bg-purple-500/10 hover:border-purple-500/50 transition-all'
                                    ):
                                        with ui.row().classes('items-center gap-1.5'):
                                            ui.icon('folder_open', size='xs').classes('text-slate-400')
                                            ui.label('Local').classes('text-xs text-slate-300 leading-none')
                            else:
                                # Vista de modelo seleccionado
                                compat = model_info.get('compat_status') if model_info else None
                                vram_str = model_info.get('vram_str', '?') if model_info else '?'
                                params_str = model_info.get('params_str', '?') if model_info else '?'
                                available_vram = get_available_vram_gb()

                                if compat == 'compatible':
                                    bg_class = 'bg-emerald-500/10'
                                    border_class = 'border-emerald-500/30'
                                    icon_name = 'check_circle'
                                    icon_color = 'text-emerald-400'
                                    tooltip_text = f'{current}\n✓ Compatible con tu sistema\nVRAM: ~{vram_str} | Params: {params_str}'
                                elif compat == 'limite':
                                    bg_class = 'bg-amber-500/10'
                                    border_class = 'border-amber-500/30'
                                    icon_name = 'warning'
                                    icon_color = 'text-amber-400'
                                    tooltip_text = f'{current}\n⚠ Al límite de tu VRAM ({available_vram:.1f}GB)\nNecesita ~{vram_str} | Puede funcionar lento'
                                elif compat == 'incompatible':
                                    bg_class = 'bg-red-500/10'
                                    border_class = 'border-red-500/30'
                                    icon_name = 'error'
                                    icon_color = 'text-red-400'
                                    tooltip_text = f'{current}\n✗ Excede tu VRAM ({available_vram:.1f}GB)\nNecesita ~{vram_str} | Probablemente no funcione'
                                else:
                                    bg_class = 'bg-purple-500/10'
                                    border_class = 'border-purple-500/30'
                                    icon_name = 'check_circle'
                                    icon_color = 'text-purple-400'
                                    tooltip_text = f'{current}\nModelo local o sin información de compatibilidad'

                                async def switch_to_edit(step_i=i):
                                    step_model_state['editing'] = True
                                    render_step_model_selector(step_i)
                                    await asyncio.sleep(0.05)
                                    if step_model_refs['input']:
                                        step_model_refs['input'].run_method('focus')

                                with ui.row().classes('w-1/2 items-center gap-2'):
                                    with ui.element('div').classes(
                                        f'flex-1 h-10 flex items-center gap-2 px-3 rounded-lg {bg_class} border {border_class} '
                                        'cursor-pointer hover:opacity-80 transition-opacity'
                                    ).on('click', switch_to_edit):
                                        ui.icon(icon_name, size='xs').classes(icon_color)
                                        # Mostrar nombre corto del modelo
                                        model_name = current.split('/')[-1] if '/' in current else current
                                        if len(model_name) > 40:
                                            model_name = model_name[:37] + '...'
                                        ui.label(model_name).classes('text-sm text-slate-200 truncate flex-1')
                                        if vram_str != '?':
                                            ui.label(f"~{vram_str}").classes('text-xs text-slate-400')
                                        ui.tooltip(tooltip_text).classes('bg-slate-800 text-slate-200 whitespace-pre-line')

                                    ui.button(icon='edit', on_click=switch_to_edit).props('flat round size=xs color=none').classes('!text-slate-400 hover:!text-purple-400')

                    render_step_model_selector(idx)

    def render_parse_config(idx: int, step: Dict):
        """Renderiza configuración de paso parse."""
        params = step.get('parameters', {})
        rules = params.get('rules', [])

        with ui.column().classes('w-full gap-3'):
            # === SECCIÓN: REGLAS DE EXTRACCIÓN ===
            with ui.column().classes('w-full gap-2'):
                ui.label('Reglas de extracción').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Define patrones para extraer datos del texto.\nCada regla busca un fragmento y lo guarda en una variable.')

                for r_idx, rule in enumerate(rules):
                    # Separador entre reglas (no antes de la primera)
                    if r_idx > 0:
                        ui.element('div').classes('w-full h-px bg-slate-700/50 my-2')

                    # Layout: número (con X al hover) | campos
                    with ui.row().classes('w-full items-start gap-3'):
                        # Número a la izquierda (se convierte en X al hover)
                        with ui.element('div').classes('w-5 h-8 flex items-center justify-center relative group/rule'):
                            ui.label(f'#{r_idx + 1}').classes('text-xs text-slate-500 group-hover/rule:opacity-0 transition-opacity')
                            ui.button(
                                icon='close',
                                on_click=lambda si=idx, ri=r_idx: remove_parse_rule(si, ri)
                            ).props('flat round size=xs color=none').classes(
                                '!text-slate-500 hover:!text-red-400 absolute opacity-0 group-hover/rule:opacity-100 transition-opacity'
                            )

                        # Campos
                        with ui.column().classes('flex-1 gap-2'):
                            with ui.row().classes('w-full gap-2'):
                                ui.input(
                                    label='Nombre de variable',
                                    value=rule.get('name', ''),
                                    on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'name', e.value)
                                ).props('outlined dark dense').classes('w-36 input-subtle').tooltip('Nombre de la variable donde se guardará el dato extraído')

                                ui.select(
                                    label='Modo',
                                    options=['KEYWORD', 'REGEX'],
                                    value=rule.get('mode', 'KEYWORD'),
                                    on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'mode', e.value)
                                ).props('outlined dark dense').classes('w-36 input-subtle').tooltip('KEYWORD: busca texto literal\nREGEX: expresión regular')

                            with ui.row().classes('w-full gap-2'):
                                ui.input(
                                    label='Extraer desde',
                                    value=rule.get('pattern', ''),
                                    on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'pattern', e.value)
                                ).props('outlined dark dense').classes('flex-1 input-subtle').tooltip('Texto o patrón que marca DÓNDE empieza el dato')

                                ui.input(
                                    label='Hasta',
                                    value=rule.get('secondary_pattern', ''),
                                    on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'secondary_pattern', e.value)
                                ).props('outlined dark dense').classes('flex-1 input-subtle').tooltip('Solo KEYWORD: texto donde TERMINA el dato')

                                ui.input(
                                    label='Si falla',
                                    value=rule.get('fallback_value', ''),
                                    on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'fallback_value', e.value)
                                ).props('outlined dark dense').classes('w-28 input-subtle').tooltip('Valor por defecto si no encuentra')

                # Botón añadir alineado con el número
                with ui.row().classes('w-full items-center gap-3'):
                    ui.element('div').classes('w-5')  # Spacer para alinear con números
                    with ui.button(icon='add', on_click=lambda i=idx: add_parse_rule(i)).props('flat round size=sm color=none').classes('!text-purple-400 hover:!text-purple-300'):
                        ui.tooltip('Añadir regla de extracción')

            # === SECCIÓN: CONFIGURACIÓN DE SALIDA ===
            ui.element('div').classes('w-full h-px bg-slate-700/50 my-1')

            with ui.column().classes('w-full gap-2'):
                with ui.row().classes('items-center gap-1.5'):
                    ui.icon('filter_alt').classes('text-slate-400 text-sm')
                    ui.label('Configuración de salida').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Controla qué resultados se incluyen en el output del paso')

                with ui.row().classes('w-full gap-2'):
                    # Filtro de resultados (opciones correctas del backend)
                    current_filter = params.get('output_filter', 'all')
                    filter_options = {
                        'all': 'Todos',
                        'successful': 'Solo exitosos',
                        'first_n': 'Primeros N'
                    }

                    def on_filter_change(e, i=idx):
                        # Convertir label a value
                        label_to_value = {v: k for k, v in filter_options.items()}
                        value = label_to_value.get(e.value, 'all')
                        update_step_param(i, 'output_filter', value)
                        # Si cambia a first_n y no hay límite, establecer uno por defecto
                        if value == 'first_n' and not params.get('output_limit'):
                            update_step_param(i, 'output_limit', 10)
                        refresh_builder()

                    ui.select(
                        label='Filtro',
                        options=list(filter_options.values()),
                        value=filter_options.get(current_filter, 'Todos'),
                        on_change=on_filter_change
                    ).props('outlined dark dense').classes('w-40 input-subtle').tooltip('Todos: incluye todas las coincidencias\nSolo exitosos: solo cuando todas las reglas coinciden\nPrimeros N: limita a las primeras N coincidencias')

                    # Campo de límite (solo visible si filter = first_n)
                    if current_filter == 'first_n':
                        ui.number(
                            label='Límite',
                            value=params.get('output_limit', 10),
                            min=1,
                            max=1000,
                            on_change=lambda e, i=idx: update_step_param(i, 'output_limit', int(e.value) if e.value else 10)
                        ).props('outlined dark dense').classes('w-24 input-subtle').tooltip('Número máximo de coincidencias a incluir')

    def render_verify_config(idx: int, step: Dict):
        """Renderiza configuración de paso verify."""
        params = step.get('parameters', {})
        methods = params.get('methods', [])

        # Inicializar referencias para este paso
        if idx not in field_refs:
            field_refs[idx] = {}
        field_refs[idx]['methods'] = {}

        def get_step_output_variables(step_data: Dict, step_idx: int) -> List[str]:
            """Retorna lista de nombres de variables que produce un paso."""
            step_type = step_data.get('type', 'generate')
            if step_type == 'generate':
                vars_list = [f'output_{step_idx + 1}']
                # Incluir variables de parse_rules integradas
                parse_rules = step_data.get('parameters', {}).get('parse_rules', [])
                for rule in parse_rules:
                    if rule.get('name'):
                        vars_list.append(rule['name'])
                return vars_list
            elif step_type == 'parse':
                rules = step_data.get('parameters', {}).get('rules', [])
                return [rule['name'] for rule in rules if rule.get('name')]
            elif step_type == 'verify':
                return ['status', 'details']
            return []

        # Obtener variables disponibles (entrada + pasos referenciados)
        def get_available_variables() -> List[Dict[str, str]]:
            """Retorna lista de variables disponibles (entrada + pasos referenciados)."""
            variables = []

            # Variables de entrada (si están habilitadas)
            if local_state['input_vars_enabled'] and local_state['input_fields']:
                for field in local_state['input_fields']:
                    variables.append({'name': field, 'step': -1, 'type': 'input'})

            # Variables de pasos referenciados
            current_refs = step.get('reference_step_numbers', [])
            for ref_idx in current_refs:
                if 0 <= ref_idx < len(local_state['steps']):
                    ref_step = local_state['steps'][ref_idx]
                    ref_type = ref_step.get('type', 'generate')
                    if ref_type == 'generate':
                        variables.append({'name': f'output_{ref_idx + 1}', 'step': ref_idx, 'type': 'generate'})
                        # Incluir variables de parse_rules integradas
                        parse_rules = ref_step.get('parameters', {}).get('parse_rules', [])
                        for rule in parse_rules:
                            if rule.get('name'):
                                variables.append({'name': rule['name'], 'step': ref_idx, 'type': 'parse'})
                    elif ref_type == 'parse':
                        rules = ref_step.get('parameters', {}).get('rules', [])
                        for rule in rules:
                            if rule.get('name'):
                                variables.append({'name': rule['name'], 'step': ref_idx, 'type': 'parse'})
                    elif ref_type == 'verify':
                        variables.append({'name': 'status', 'step': ref_idx, 'type': 'verify'})
                        variables.append({'name': 'details', 'step': ref_idx, 'type': 'verify'})
            return variables

        def render_variable_chips(textarea_element, field: str, method_idx: int):
            """Renderiza los chips de variables clickables que insertan en la posición del cursor."""
            variables = get_available_variables()
            if not variables:
                return

            with ui.row().classes('w-full flex-wrap items-center gap-1.5 px-2 py-1.5 bg-slate-800/40 rounded-b border border-t-0 border-slate-600/50'):
                ui.icon('add_link', size='xs').classes('text-slate-400')
                ui.label('Insertar:').classes('text-xs text-slate-400 mr-1')

                for var in variables:
                    v = var['name']
                    var_type = var['type']
                    var_step = var['step']
                    var_text = f'{{{v}}}'

                    type_names = {'generate': 'Generado', 'parse': 'Detectado', 'verify': 'Verificado', 'input': 'Entrada'}

                    # Nombre descriptivo según tipo (igual que en generate)
                    if var_type == 'input':
                        display_name = f'{{{v}}}'
                        tooltip_text = f'Variable de entrada. Click para insertar {{{v}}}'
                    else:
                        step_num = var_step + 1
                        display_name = f'Paso {step_num}: {{{v}}}'
                        tooltip_text = f'{type_names.get(var_type, "Output")} del paso {step_num}. Click para insertar la variable'

                    async def insert_var_at_cursor(vt=var_text, fld=field, mi=method_idx):
                        js_code = f'''
                        (() => {{
                            const textarea = document.getElementById('{textarea_element.id}').querySelector('textarea, input');
                            if (!textarea) return null;
                            const start = textarea.selectionStart || 0;
                            const end = textarea.selectionEnd || 0;
                            const text = textarea.value || '';
                            const varText = '{vt}';
                            const newText = text.substring(0, start) + varText + text.substring(end);
                            textarea.value = newText;
                            const newCursorPos = start + varText.length;
                            textarea.selectionStart = textarea.selectionEnd = newCursorPos;
                            textarea.focus();
                            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            return newText;
                        }})()
                        '''
                        result = await ui.run_javascript(js_code)
                        if result is not None:
                            update_verify_method(idx, mi, fld, result)

                    with ui.button(on_click=insert_var_at_cursor).props('flat dense no-caps color=none').classes(
                        'px-2 py-0.5 min-h-0 hover:bg-slate-700/50 transition-all'
                    ):
                        ui.label(display_name).classes('font-mono text-xs text-slate-200 normal-case')
                        ui.tooltip(tooltip_text).classes('bg-slate-800 text-slate-200')

        with ui.column().classes('w-full gap-3'):
            # === SECCIÓN: REFERENCIAS A PASOS ANTERIORES ===
            if idx > 0:
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('items-center gap-1.5'):
                        ui.icon('data_object').classes('text-slate-400 text-sm -mt-0.5')
                        ui.label('Referenciar outputs de pasos anteriores').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Selecciona pasos anteriores para usar sus outputs como variables en tus prompts.')

                    current_refs = step.get('reference_step_numbers', [])
                    with ui.column().classes('w-full gap-0'):
                        for prev_idx in range(idx):
                            prev_step = local_state['steps'][prev_idx]
                            prev_type = prev_step.get('type', 'generate')
                            type_names = {'generate': 'Generar', 'parse': '', 'verify': 'Verificar'}
                            is_checked = prev_idx in current_refs

                            # Obtener variables que produce este paso
                            step_vars = get_step_output_variables(prev_step, prev_idx)
                            vars_display = ', '.join(f'{{{v}}}' for v in step_vars) if step_vars else '(sin variables establecidas)'

                            with ui.row().classes('w-full items-center gap-0.5 py-1 ml-2'):
                                cb = ui.checkbox(
                                    value=is_checked,
                                    on_change=lambda e, i=idx, r=prev_idx: toggle_reference_step(i, r, e.value)
                                ).props('dense size=sm')
                                ui.label(f'Paso {prev_idx + 1}:').classes('text-sm text-slate-400 cursor-pointer select-none ml-0.5').on('click', lambda _, c=cb: c.set_value(not c.value))
                                ui.label(type_names.get(prev_type, prev_type)).classes('text-sm text-slate-400 cursor-pointer select-none').on('click', lambda _, c=cb: c.set_value(not c.value))
                                ui.label('→').classes('text-slate-500 text-xs mx-0.5')
                                vars_classes = 'text-xs text-slate-500 font-mono' if step_vars else 'text-xs text-slate-500 italic'
                                ui.label(vars_display).classes(vars_classes)

                    ui.element('div').classes('w-full h-px bg-slate-700/50 my-1')

            # === SECCIÓN: MÉTODOS DE VERIFICACIÓN ===
            with ui.column().classes('w-full gap-2'):
                for m_idx, method in enumerate(methods):
                    # Inicializar referencias para este método
                    field_refs[idx]['methods'][m_idx] = {}

                    # Separador entre métodos (no antes del primero)
                    if m_idx > 0:
                        ui.element('div').classes('w-full h-px bg-slate-700/50 my-2')

                    # Layout: número (con X al hover) | campos
                    with ui.row().classes('w-full gap-3'):
                        # Número a la izquierda (se convierte en X al hover), centrado verticalmente
                        with ui.element('div').classes('w-5 flex items-center justify-center relative group/method self-center'):
                            ui.label(f'#{m_idx + 1}').classes('text-xs text-slate-500 group-hover/method:opacity-0 transition-opacity')
                            ui.button(
                                icon='close',
                                on_click=lambda si=idx, mi=m_idx: remove_verify_method(si, mi)
                            ).props('flat round size=xs color=none').classes(
                                '!text-slate-500 hover:!text-red-400 absolute opacity-0 group-hover/method:opacity-100 transition-opacity'
                            )

                        # Campos
                        with ui.column().classes('flex-1 gap-2'):
                            # Fila superior: Nombre + Consenso
                            with ui.row().classes('w-full gap-6 items-end'):
                                # Nombre con label arriba
                                with ui.column().classes('gap-0'):
                                    ui.label('Nombre del método').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Identificador único del método')

                                    name_input = ui.input(
                                        value=method.get('name', ''),
                                        placeholder='metodo_1',
                                    ).props('outlined dark dense color=purple').classes('w-40 input-subtle mt-1')
                                    name_input.on(
                                        'blur',
                                        lambda e, si=idx, mi=m_idx: validate_and_update_method_name(si, mi, e.sender.value, e.sender)
                                    )
                                    field_refs[idx]['methods'][m_idx]['name'] = name_input

                                # Consenso con label arriba
                                with ui.column().classes('gap-0'):
                                    ui.label('Consenso').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Sistema de votación.\n Se dan N respuestas, de las cuales se requieren M respuestas válidas para pasar el filtro de éste método.')

                                    # Referencias para validación cruzada
                                    current_num_seq = method.get('num_sequences', 3)
                                    current_req = method.get('required_matches', 2)

                                    def on_required_change(e, si=idx, mi=m_idx):
                                        val = int(e.value) if e.value else 1
                                        num_seq = local_state['steps'][si]['parameters']['methods'][mi].get('num_sequences', 3)
                                        # Cap required at num_sequences
                                        val = min(val, num_seq)
                                        update_verify_method(si, mi, 'required_matches', val)
                                        if e.value and int(e.value) > num_seq:
                                            e.sender.value = val

                                    def on_num_seq_change(e, si=idx, mi=m_idx):
                                        val = int(e.value) if e.value else 1
                                        update_verify_method(si, mi, 'num_sequences', val)
                                        # Reduce required_matches if it exceeds new num_sequences
                                        req = local_state['steps'][si]['parameters']['methods'][mi].get('required_matches', 2)
                                        if req > val:
                                            update_verify_method(si, mi, 'required_matches', val)
                                            refresh_builder()

                                    with ui.row().classes('items-center gap-1.5 mt-1'):
                                        ui.number(
                                            value=min(current_req, current_num_seq),
                                            min=1, max=current_num_seq,
                                            on_change=on_required_change
                                        ).props('outlined dark dense color=purple input-class="text-center !py-0.5"').classes('w-14').tooltip('Número de respuestas válidas para pasar el filtro')
                                        ui.label('de').classes('text-xs text-slate-400')
                                        ui.number(
                                            value=current_num_seq,
                                            min=1, max=10,
                                            on_change=on_num_seq_change
                                        ).props('outlined dark dense color=purple input-class="text-center !py-0.5"').classes('w-14').tooltip('Número de respuestas a generar en total')

                                # Temperatura
                                with ui.column().classes('gap-0'):
                                    ui.label('Temp.').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Controla la variabilidad.\n0 = Consistente\n2 = Variado')
                                    ui.number(
                                        value=method.get('temperature', 0.8),
                                        min=0, max=2, step=0.1,
                                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'temperature', float(e.value) if e.value is not None else 0.8)
                                    ).props('outlined dark dense color=purple input-class="text-center !py-0.5"').classes('w-16 mt-1')

                            # System Prompt con chips de variables
                            has_vars = bool(get_available_variables())
                            with ui.column().classes('w-full gap-0 mt-2'):
                                ui.label('System Prompt').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Define el rol del verificador')
                                with ui.column().classes('w-full gap-0 mt-1'):
                                    system_textarea = ui.textarea(
                                        value=method.get('system_prompt', ''),
                                        placeholder='Responde únicamente Sí o No.',
                                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'system_prompt', e.value)
                                    ).props('outlined dark dense rows=1 autogrow').classes('w-full input-subtle' + (' rounded-b-none' if has_vars else ''))
                                    field_refs[idx]['methods'][m_idx]['system_prompt'] = system_textarea
                                    render_variable_chips(system_textarea, 'system_prompt', m_idx)

                            # User Prompt con chips de variables
                            with ui.column().classes('w-full gap-0 mt-2'):
                                ui.label('User Prompt').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Pregunta para el verificador. Usa {variable} para insertar datos.')
                                with ui.column().classes('w-full gap-0 mt-1'):
                                    user_textarea = ui.textarea(
                                        value=method.get('user_prompt', ''),
                                        placeholder='¿El texto {texto} es correcto?',
                                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'user_prompt', e.value)
                                    ).props('outlined dark dense rows=1 autogrow').classes('w-full input-subtle' + (' rounded-b-none' if has_vars else ''))
                                    field_refs[idx]['methods'][m_idx]['user_prompt'] = user_textarea
                                    render_variable_chips(user_textarea, 'user_prompt', m_idx)

                            # Respuestas válidas
                            valid_responses = method.get('valid_responses', [])
                            if not valid_responses:
                                valid_responses = ['']  # Al menos un input vacío

                            with ui.column().classes('w-full gap-0 mt-2'):
                                ui.label('Respuestas válidas').classes('text-xs font-medium text-slate-300 uppercase cursor-help').tooltip('Respuestas válidas para el filtro.\nAñade tantas como desees.\nLa respuesta del modelo debe ser exactamente igual a lo especificado para ser considerada válida.')

                                with ui.row().classes('items-center gap-1 mt-1 flex-wrap'):
                                    for r_idx, response in enumerate(valid_responses):
                                        def on_response_change(e, si=idx, mi=m_idx, ri=r_idx):
                                            responses = local_state['steps'][si]['parameters']['methods'][mi].get('valid_responses', [''])
                                            if ri < len(responses):
                                                responses[ri] = e.value
                                                update_verify_method(si, mi, 'valid_responses', responses)

                                        def on_remove_response(si=idx, mi=m_idx, ri=r_idx):
                                            responses = local_state['steps'][si]['parameters']['methods'][mi].get('valid_responses', [''])
                                            if len(responses) > 1 and ri < len(responses):
                                                responses.pop(ri)
                                                update_verify_method(si, mi, 'valid_responses', responses)
                                                refresh_builder()

                                        with ui.input(
                                            value=response,
                                            placeholder='Sí',
                                            on_change=on_response_change
                                        ).props('outlined dark dense color=purple').classes('w-24 input-subtle') as inp:
                                            if len(valid_responses) > 1:
                                                with inp.add_slot('append'):
                                                    ui.button(icon='close', on_click=on_remove_response).props('flat dense round size=xs color=none').classes('!text-slate-500 hover:!text-red-400')

                                    def add_valid_response(si=idx, mi=m_idx):
                                        responses = local_state['steps'][si]['parameters']['methods'][mi].get('valid_responses', [''])
                                        responses.append('')
                                        update_verify_method(si, mi, 'valid_responses', responses)
                                        refresh_builder()

                                    ui.button(icon='add', on_click=add_valid_response).props('flat dense round size=sm color=purple').classes('ml-1')

                                    # Checkbox ignorar mayúsculas
                                    ui.element('div').classes('w-px h-4 bg-slate-600/50 mx-2')
                                    ignore_case = method.get('ignore_case', True)
                                    ui.checkbox(
                                        'Ignorar mayúsculas',
                                        value=ignore_case,
                                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'ignore_case', e.value)
                                    ).props('dense size=sm').classes('text-xs text-slate-400')

                            # Modelo específico para este método (opcional)
                            method_llm = method.get('llm_config')
                            with ui.row().classes('w-full items-center gap-2 mt-2'):
                                if method_llm:
                                    # Mostrar modelo seleccionado con opción de quitar
                                    ui.icon('smart_toy', size='xs').classes('text-purple-400')
                                    ui.label(method_llm if len(method_llm) < 40 else f'...{method_llm[-37:]}').classes('text-xs text-slate-300 font-mono truncate flex-1')
                                    ui.button(
                                        icon='close',
                                        on_click=lambda si=idx, mi=m_idx: (update_verify_method(si, mi, 'llm_config', None), refresh_builder())
                                    ).props('flat round size=xs color=none').classes('!text-slate-500 hover:!text-red-400')
                                else:
                                    # Input para añadir modelo específico
                                    def on_method_llm_set(e, si=idx, mi=m_idx):
                                        if e.value and e.value.strip():
                                            update_verify_method(si, mi, 'llm_config', e.value.strip())
                                            refresh_builder()

                                    ui.input(
                                        placeholder='Modelo específico (opcional)',
                                        on_change=on_method_llm_set
                                    ).props('outlined dark dense color=purple').classes('flex-1 input-subtle text-xs').tooltip('Ruta al modelo local o ID de HuggingFace.\nSi se deja vacío, usa el modelo del paso.')

                # Fila con botón añadir
                with ui.row().classes('w-full items-center justify-between'):
                    # Botón añadir método (icono + texto clickable)
                    ui.button('MÉTODO', icon='add', on_click=lambda i=idx: add_verify_method(i)).props('flat dense size=md color=none').classes('!text-purple-400 hover:!text-purple-300 !px-1 uppercase')


    def update_parse_rule(step_idx: int, rule_idx: int, field: str, value):
        """Actualiza un campo de una regla de parsing."""
        if 0 <= step_idx < len(local_state['steps']):
            rules = local_state['steps'][step_idx]['parameters'].get('rules', [])
            if 0 <= rule_idx < len(rules):
                rules[rule_idx][field] = value

    def remove_parse_rule(step_idx: int, rule_idx: int):
        """Elimina una regla de parsing."""
        if 0 <= step_idx < len(local_state['steps']):
            rules = local_state['steps'][step_idx]['parameters'].get('rules', [])
            if 0 <= rule_idx < len(rules):
                rules.pop(rule_idx)
                refresh_builder()

    # === Funciones para parse_rules integrado en generate ===
    def add_generate_parse_rule(step_idx: int):
        """Añade una regla de parsing a un paso generate."""
        if 0 <= step_idx < len(local_state['steps']):
            step = local_state['steps'][step_idx]
            if 'parameters' not in step:
                step['parameters'] = {}
            if 'parse_rules' not in step['parameters']:
                step['parameters']['parse_rules'] = []
            step['parameters']['parse_rules'].append({
                'name': '',
                'mode': 'KEYWORD',
                'pattern': '',
                'secondary_pattern': '',
                'fallback_value': ''
            })
            refresh_builder()

    def update_generate_parse_rule(step_idx: int, rule_idx: int, field: str, value):
        """Actualiza un campo de una regla de parsing en generate."""
        if 0 <= step_idx < len(local_state['steps']):
            rules = local_state['steps'][step_idx]['parameters'].get('parse_rules', [])
            if 0 <= rule_idx < len(rules):
                rules[rule_idx][field] = value
                # Refrescar UI cuando cambia el modo para mostrar/ocultar campos
                if field == 'mode':
                    refresh_builder()

    def remove_generate_parse_rule(step_idx: int, rule_idx: int):
        """Elimina una regla de parsing de un paso generate."""
        if 0 <= step_idx < len(local_state['steps']):
            rules = local_state['steps'][step_idx]['parameters'].get('parse_rules', [])
            if 0 <= rule_idx < len(rules):
                rules.pop(rule_idx)
                refresh_builder()

    def toggle_generate_parsing(step_idx: int, enabled: bool):
        """Activa/desactiva el parsing en un paso generate."""
        if 0 <= step_idx < len(local_state['steps']):
            step = local_state['steps'][step_idx]
            if enabled:
                # Crear regla inicial si no existe o está vacía
                if not step['parameters'].get('parse_rules'):
                    step['parameters']['parse_rules'] = [{
                        'name': '',
                        'mode': 'KEYWORD',
                        'pattern': '',
                        'secondary_pattern': '',
                        'fallback_value': ''
                    }]
            else:
                step['parameters'].pop('parse_rules', None)
            refresh_builder()

    def update_verify_method(step_idx: int, method_idx: int, field: str, value):
        """Actualiza un campo de un método de verificación."""
        if 0 <= step_idx < len(local_state['steps']):
            methods = local_state['steps'][step_idx]['parameters'].get('methods', [])
            if 0 <= method_idx < len(methods):
                methods[method_idx][field] = value

    async def validate_and_update_method_name(step_idx: int, method_idx: int, new_value: str, input_element):
        """Valida el nombre de método y muestra feedback visual si hay error."""
        import asyncio
        new_name = new_value.strip()

        if 0 > step_idx or step_idx >= len(local_state['steps']):
            return

        methods = local_state['steps'][step_idx]['parameters'].get('methods', [])
        if 0 > method_idx or method_idx >= len(methods):
            return

        old_name = methods[method_idx].get('name', '')

        # Si está vacío, restaurar nombre actual
        if not new_name:
            input_element.value = old_name
            return

        # Si no hay cambio, no hacer nada
        if new_name == old_name:
            return

        # Verificar si ya existe otro método con ese nombre
        existing_names = {
            m.get('name', '') for i, m in enumerate(methods) if i != method_idx
        }
        if new_name in existing_names:
            # Mostrar error visual: borde rojo + shake
            input_element.classes(add='input-error shake-error')
            ui.notify(f'Ya existe un método "{new_name}"', type='negative')

            # Restaurar nombre actual
            input_element.value = old_name

            # Quitar clases de error después de la animación
            await asyncio.sleep(0.5)
            input_element.classes(remove='input-error shake-error')
            return

        # Nombre válido, actualizar
        methods[method_idx]['name'] = new_name

    def remove_verify_method(step_idx: int, method_idx: int):
        """Elimina un método de verificación."""
        if 0 <= step_idx < len(local_state['steps']):
            methods = local_state['steps'][step_idx]['parameters'].get('methods', [])
            if 0 <= method_idx < len(methods):
                methods.pop(method_idx)
                refresh_builder()

    def export_data_json():
        """Exporta los datos de entrada como JSON (formato row-based)."""
        input_values = local_state['input_values']
        fields = local_state['input_fields']
        if not input_values or not fields:
            ui.notify('No hay datos para exportar', type='warning')
            return
        # Convertir de column-based a row-based
        max_entries = max((len(vals) for vals in input_values.values()), default=0)
        row_entries = []
        for i in range(max_entries):
            entry = {}
            for field in fields:
                vals = input_values.get(field, [])
                entry[field] = vals[i] if i < len(vals) else ''
            row_entries.append(entry)
        ui.download(
            json.dumps(row_entries, indent=2, ensure_ascii=False).encode('utf-8'),
            f'datos_entrada_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        ui.notify('Datos exportados', type='positive')

    def toggle_input_vars(enabled: bool):
        """Activa/desactiva las variables de entrada."""
        local_state['input_vars_enabled'] = enabled
        if enabled and not local_state['input_fields']:
            # Añadir campo por defecto al activar con placeholder (sin nombre visible)
            add_input_field(show_placeholder=True)
        else:
            refresh_data_section()
            refresh_builder()  # Actualizar chips de variables en pasos

    def render_data_section():
        """Renderiza la sección de variables de entrada - diseño de tabla."""
        is_enabled = local_state['input_vars_enabled']
        fields_list = local_state['input_fields']
        input_values = local_state['input_values']
        # Calcular número de filas (ejecuciones)
        row_count = max((len(vals) for vals in input_values.values()), default=0)

        # ════════════════════════════════════════════════════════════════
        # HEADER: Toggle + estado + botones
        # ════════════════════════════════════════════════════════════════
        with ui.row().classes('w-full items-center gap-2').style('height: 44px'):
            ui.switch(
                value=is_enabled,
                on_change=lambda e: toggle_input_vars(e.value)
            ).props('dense color=purple').classes('input-vars-toggle shrink-0')

            with ui.column().classes('gap-0 flex-1'):
                ui.label('Usar Variables de Entrada').classes('text-sm font-medium text-slate-200')

                ui.label('Añade diferentes variables y ejecuta el pipeline para cada uno de sus valores').classes('text-xs text-slate-400')

            # Acciones en header (iconos púrpura) - siempre reservar espacio
            with ui.row().classes('shrink-0 gap-1'):
                if is_enabled:
                    if row_count > 0:
                        with ui.button(
                            icon='code' if not local_state['json_view'] else 'view_list',
                            on_click=toggle_json_view
                        ).props('flat round size=md color=none').classes('!text-purple-400 hover:!text-purple-300'):
                            ui.tooltip('Ver como JSON' if not local_state['json_view'] else 'Ver como tabla')
                        with ui.button(icon='download', on_click=export_data_json).props('flat round size=md color=none').classes('!text-purple-400 hover:!text-purple-300'):
                            ui.tooltip('Exportar como JSON')

        if not is_enabled:
            return

        # ════════════════════════════════════════════════════════════════
        # CONTENIDO: Tabla de variables
        # ════════════════════════════════════════════════════════════════

        # Vista JSON (modo vista / modo edición)
        if local_state['json_view'] and fields_list and row_count > 0:
            # Generar JSON actual desde los datos
            row_entries = []
            for i in range(row_count):
                entry = {}
                for field in fields_list:
                    vals = input_values.get(field, [])
                    entry[field] = vals[i] if i < len(vals) else ''
                row_entries.append(entry)
            json_text = json.dumps(row_entries, indent=2, ensure_ascii=False)

            def enter_edit_mode():
                """Entra en modo edición."""
                local_state['json_edit_mode'] = True
                local_state['json_edit_buffer'] = json_text
                refresh_data_section()

            def cancel_edit_mode():
                """Cancela la edición y vuelve a modo vista."""
                local_state['json_edit_mode'] = False
                local_state['json_edit_buffer'] = ''
                refresh_data_section()

            def apply_json_changes():
                """Aplica los cambios del JSON al estado."""
                try:
                    parsed = json.loads(local_state['json_edit_buffer'])
                    if not isinstance(parsed, list):
                        ui.notify('El JSON debe ser una lista de objetos', type='warning')
                        return

                    # Extraer campos y valores del JSON
                    all_fields = set()
                    for entry in parsed:
                        if isinstance(entry, dict):
                            all_fields.update(entry.keys())

                    new_fields = list(all_fields)
                    new_values = {field: [] for field in new_fields}

                    for entry in parsed:
                        if isinstance(entry, dict):
                            for field in new_fields:
                                new_values[field].append(str(entry.get(field, '')))

                    # Actualizar estado
                    local_state['input_fields'] = new_fields
                    local_state['input_values'] = new_values
                    local_state['placeholder_fields'] = set()  # Las variables del JSON ya tienen nombre
                    local_state['json_edit_mode'] = False
                    local_state['json_edit_buffer'] = ''

                    ui.notify('Cambios aplicados correctamente', type='positive')
                    refresh_data_section()
                    refresh_builder()  # Actualizar chips de variables en prompts

                except json.JSONDecodeError as err:
                    ui.notify(f'JSON inválido: {err.msg}', type='negative')

            def copy_json():
                """Copia el JSON al portapapeles."""
                ui.run_javascript(f'navigator.clipboard.writeText({json.dumps(json_text)})')
                ui.notify('JSON copiado al portapapeles', type='positive')

            def update_buffer(e):
                """Actualiza el buffer de edición."""
                local_state['json_edit_buffer'] = e.value

            with ui.column().classes('w-full flex-1 gap-2'):
                if not local_state['json_edit_mode']:
                    # ═══════════════════════════════════════════════════════════
                    # MODO VISTA (solo lectura)
                    # ═══════════════════════════════════════════════════════════
                    with ui.row().classes('w-full items-center justify-start gap-2'):
                        with ui.button(on_click=copy_json).props('flat dense no-caps').classes(
                            'px-2 py-1 text-slate-200 hover:text-white hover:bg-slate-700/50 rounded'
                        ):
                            ui.icon('content_copy', size='xs').classes('mr-1 text-slate-200')
                            ui.label('Copiar').classes('text-xs text-slate-200')
                        with ui.button(on_click=enter_edit_mode).props('flat dense no-caps').classes(
                            'px-2 py-1 text-slate-200 hover:text-white hover:bg-slate-700/50 rounded'
                        ):
                            ui.icon('edit', size='xs').classes('mr-1 text-slate-200')
                            ui.label('Editar').classes('text-xs text-slate-200')

                    ui.textarea(
                        value=json_text,
                    ).props('outlined dark readonly').classes(
                        'w-full flex-1 font-mono text-sm input-subtle json-editor-textarea'
                    ).style('min-height: 280px')

                else:
                    # ═══════════════════════════════════════════════════════════
                    # MODO EDICIÓN
                    # ═══════════════════════════════════════════════════════════
                    with ui.row().classes('w-full items-center gap-2'):
                        ui.icon('edit_note', size='xs').classes('text-slate-400')
                        ui.label('Modo edición: Los cambios reemplazarán los datos actuales en la tabla').classes('text-xs text-slate-400')
                        ui.element('div').classes('flex-1')
                        with ui.button(on_click=cancel_edit_mode).props('flat dense no-caps').classes(
                            'px-2 py-1 text-slate-200 hover:text-white hover:bg-slate-700/50 rounded'
                        ):
                            ui.icon('close', size='xs').classes('mr-1 text-slate-200')
                            ui.label('Cancelar').classes('text-xs text-slate-200')
                        with ui.button(on_click=apply_json_changes).props('flat dense no-caps').classes(
                            'px-2 py-1 bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 rounded'
                        ):
                            ui.icon('check', size='xs').classes('mr-1 text-purple-300')
                            ui.label('Aplicar cambios').classes('text-xs text-purple-300')

                    ui.textarea(
                        value=local_state['json_edit_buffer'],
                        on_change=update_buffer
                    ).props('outlined dark').classes(
                        'w-full flex-1 font-mono text-sm input-subtle json-editor-textarea'
                    ).style('min-height: 280px')
            return

        # Vista Tabla con CSS Grid (todo scrollable junto)
        num_cols = len(fields_list)
        grid_cols = f'48px repeat({num_cols}, minmax(200px, auto)) auto' if num_cols > 0 else '48px auto'

        with ui.element('div').classes('w-full bg-slate-900/40 rounded-lg border border-slate-700/50 overflow-x-auto pb-4 variables-table-container'):
            with ui.element('div').classes('min-w-max').style(f'display: grid; grid-template-columns: {grid_cols}'):

                # ─────────────────────────────────────────────────────────────
                # CABECERA DE TABLA
                # ─────────────────────────────────────────────────────────────
                # Celda # (índice)
                ui.element('div').classes('bg-slate-800/60 border-b border-r border-slate-700/50 px-2 py-1 flex items-center justify-center')

                # Celdas de variables (cabecera)
                for field in fields_list:
                    # Mostrar placeholder solo si está en placeholder_fields
                    is_placeholder = field in local_state['placeholder_fields']

                    with ui.element('div').classes('bg-slate-800/60 border-b border-r border-slate-700/50 px-1 py-1'):
                        with ui.row().classes('items-center gap-1 flex-nowrap w-full'):
                            ui.input(
                                value=field,
                                placeholder='nombre_var'
                            ).props('outlined dense dark').classes(
                                'text-sm input-subtle shrink-0'
                            ).style('min-height: 18px; width: 105px').on(
                                'blur', lambda e, f=field: validate_and_update_field_name(f, e.sender.value, e.sender)
                            ).on(
                                'keydown.enter', lambda e: e.sender.run_method('blur')
                            )
                            ui.label('→').classes('text-slate-500 text-xs shrink-0')
                            ui.label(f'{{{field}}}').classes('text-slate-400 font-mono text-xs shrink-0 whitespace-nowrap')
                            ui.element('div').classes('flex-1')  # Spacer
                            ui.button(
                                icon='close',
                                on_click=lambda f=field: remove_input_field(f)
                            ).props('flat round size=xs color=none').classes(
                                '!text-slate-500 hover:!text-red-400 shrink-0'
                            )

                # Celda + Variable (alineado a la derecha)
                with ui.element('div').classes('bg-slate-800/60 border-b border-slate-700/50 px-3 py-1 flex items-center justify-end'):
                    with ui.button(on_click=lambda: add_input_field()).props('flat dense no-caps').classes(
                        'h-8 pl-2 pr-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                        'hover:bg-purple-500/20 hover:border-purple-500/50 transition-all flex items-center'
                    ):
                        ui.icon('add', size='xs').classes('text-purple-400 mr-1.5')
                        ui.label('VARIABLE').classes('text-xs text-slate-300 leading-none uppercase')

                # ─────────────────────────────────────────────────────────────
                # FILAS DE DATOS
                # ─────────────────────────────────────────────────────────────
                if not fields_list:
                    # Estado vacío (ocupa toda la fila del grid)
                    with ui.element('div').classes('py-8 px-4 flex items-center justify-center').style('grid-column: 1 / -1'):
                        with ui.column().classes('items-center gap-2'):
                            ui.icon('table_chart', size='md').classes('text-slate-600')
                            ui.label('Añade una variable para empezar').classes('text-sm text-slate-500')
                            ui.label('Usa {nombre} en los prompts para referenciarla').classes('text-xs text-slate-600')
                else:
                    for row_idx in range(row_count):
                        row_bg = 'bg-slate-800/20' if row_idx % 2 == 1 else ''

                        # Celda número de fila
                        with ui.element('div').classes(f'{row_bg} border-b border-r border-slate-700/30 px-2 py-1 flex items-center justify-center relative group/row'):
                            ui.label(str(row_idx + 1)).classes('text-xs text-slate-500 font-mono group-hover/row:opacity-0 transition-opacity')
                            ui.button(
                                icon='close',
                                on_click=lambda idx=row_idx: remove_row(idx)
                            ).props('flat round size=xs color=none').classes(
                                '!text-slate-500 hover:!text-red-400 absolute opacity-0 group-hover/row:opacity-100 transition-opacity'
                            )

                        # Celdas de valores
                        for field in fields_list:
                            vals = input_values.get(field, [])
                            cell_value = vals[row_idx] if row_idx < len(vals) else ''
                            with ui.element('div').classes(f'{row_bg} border-b border-r border-slate-700/30 px-1 py-1'):
                                ui.textarea(
                                    value=cell_value,
                                    placeholder='Escribe el valor...',
                                    on_change=lambda e, f=field, r=row_idx: update_cell(f, r, e.value)
                                ).props('outlined dense dark autogrow rows=1').classes(
                                    'w-full text-sm input-subtle'
                                ).style('min-height: 32px')

                        # Celda vacía para alinear con columna + Variable
                        ui.element('div').classes(f'{row_bg} border-b border-slate-700/30')

                    # ─────────────────────────────────────────────────────────
                    # FILA: Añadir nuevo valor
                    # ─────────────────────────────────────────────────────────
                    # Celda vacía para índice
                    ui.element('div').classes('bg-slate-800/30 border-r border-slate-700/30')

                    # Botón + Valor (ocupa resto de columnas)
                    with ui.element('div').classes('bg-slate-800/30 px-2 py-1 flex items-center').style('grid-column: 2 / -1'):
                        with ui.button(on_click=add_row).props('flat dense no-caps').classes(
                            'h-8 pl-2 pr-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                            'hover:bg-purple-500/20 hover:border-purple-500/50 transition-all flex items-center'
                        ):
                            ui.icon('add', size='xs').classes('text-purple-400 mr-1.5')
                            ui.label('VALOR').classes('text-xs text-slate-300 leading-none uppercase')

    def build_steps_from_config() -> List[PipelineStep]:
        """Construye los PipelineStep desde la configuración local."""
        built_steps = []

        for step_data in local_state['steps']:
            stype = step_data.get('type')
            params = step_data.get('parameters', {})

            if stype == 'generate':
                step_params = GenerateTextRequest(
                    system_prompt=params.get('system_prompt', ''),
                    user_prompt=params.get('user_prompt', ''),
                    num_sequences=params.get('num_sequences', 1),
                    max_tokens=params.get('max_tokens', 200),
                    temperature=params.get('temperature', 0.7)
                )
            elif stype == 'parse':
                rules = []
                for r in params.get('rules', []):
                    mode = ParseMode.KEYWORD if r.get('mode') == 'KEYWORD' else ParseMode.REGEX
                    rules.append(ParseRule(
                        name=r.get('name', 'campo'),
                        mode=mode,
                        pattern=r.get('pattern', ''),
                        secondary_pattern=r.get('secondary_pattern'),
                        fallback_value=r.get('fallback_value')
                    ))
                step_params = ParseRequest(
                    rules=rules,
                    output_filter=params.get('output_filter', 'all')
                )
            elif stype == 'verify':
                methods = []
                for m in params.get('methods', []):
                    mode = VerificationMode.CUMULATIVE if m.get('mode') == 'cumulative' else VerificationMode.ELIMINATORY
                    methods.append(VerificationMethod(
                        mode=mode,
                        name=m.get('name', 'verificar'),
                        system_prompt=m.get('system_prompt', ''),
                        user_prompt=m.get('user_prompt', ''),
                        num_sequences=m.get('num_sequences', 3),
                        valid_responses=m.get('valid_responses', ['Yes', 'yes']),
                        required_matches=m.get('required_matches', 2),
                        max_tokens=m.get('max_tokens', 5),
                        temperature=m.get('temperature', 0.8),
                        ignore_case=m.get('ignore_case', True),
                        llm_config=m.get('llm_config')
                    ))
                step_params = VerifyRequest(
                    methods=methods,
                    required_for_confirmed=params.get('required_for_confirmed', 1),
                    required_for_review=params.get('required_for_review', 0)
                )
            else:
                continue

            built_steps.append(PipelineStep(
                type=stype,
                parameters=step_params,
                uses_reference=step_data.get('uses_reference', False),
                reference_step_numbers=step_data.get('reference_step_numbers')
            ))

        return built_steps

    def validate_pipeline_fields():
        """Valida todos los campos obligatorios del pipeline.

        Returns:
            tuple: (is_valid, error_message, field_ref, step_idx)
        """
        for step_idx, step in enumerate(local_state['steps']):
            stype = step.get('type')
            params = step.get('parameters', {})

            if stype == 'generate':
                # Validar system_prompt
                if not params.get('system_prompt', '').strip():
                    field_ref = field_refs.get(step_idx, {}).get('system_prompt')
                    return (False, f'Paso {step_idx + 1}: System Prompt es obligatorio', field_ref, step_idx)
                # Validar user_prompt
                if not params.get('user_prompt', '').strip():
                    field_ref = field_refs.get(step_idx, {}).get('user_prompt')
                    return (False, f'Paso {step_idx + 1}: User Prompt es obligatorio', field_ref, step_idx)

            elif stype == 'verify':
                methods = params.get('methods', [])
                if not methods:
                    return (False, f'Paso {step_idx + 1}: Añade al menos un método de verificación', None, step_idx)

                for m_idx, method in enumerate(methods):
                    # Validar name
                    if not method.get('name', '').strip():
                        field_ref = field_refs.get(step_idx, {}).get('methods', {}).get(m_idx, {}).get('name')
                        return (False, f'Paso {step_idx + 1}, Método {m_idx + 1}: Nombre es obligatorio', field_ref, step_idx)
                    # Validar system_prompt
                    if not method.get('system_prompt', '').strip():
                        field_ref = field_refs.get(step_idx, {}).get('methods', {}).get(m_idx, {}).get('system_prompt')
                        return (False, f'Paso {step_idx + 1}, Método {m_idx + 1}: System Prompt es obligatorio', field_ref, step_idx)
                    # Validar user_prompt
                    if not method.get('user_prompt', '').strip():
                        field_ref = field_refs.get(step_idx, {}).get('methods', {}).get(m_idx, {}).get('user_prompt')
                        return (False, f'Paso {step_idx + 1}, Método {m_idx + 1}: User Prompt es obligatorio', field_ref, step_idx)
                    # Validar valid_responses
                    valid_responses = method.get('valid_responses', [])
                    if not valid_responses or (isinstance(valid_responses, list) and len(valid_responses) == 0):
                        return (False, f'Paso {step_idx + 1}, Método {m_idx + 1}: Respuestas válidas es obligatorio', None, step_idx)

            elif stype == 'parse':
                rules = params.get('rules', [])
                if not rules:
                    return (False, f'Paso {step_idx + 1}: Añade al menos una regla de parseo', None, step_idx)

                for r_idx, rule in enumerate(rules):
                    # Validar name
                    if not rule.get('name', '').strip():
                        return (False, f'Paso {step_idx + 1}, Regla {r_idx + 1}: Nombre es obligatorio', None, step_idx)
                    # Validar pattern
                    if not rule.get('pattern', '').strip():
                        return (False, f'Paso {step_idx + 1}, Regla {r_idx + 1}: Patrón es obligatorio', None, step_idx)

        return (True, None, None, None)

    async def shake_and_focus_field(field_ref, step_idx: int):
        """Aplica animación shake y focus a un campo."""
        # Expandir el paso si está colapsado
        if local_state.get('expanded_step') != step_idx:
            local_state['expanded_step'] = step_idx
            refresh_builder()
            await asyncio.sleep(0.1)  # Esperar a que se renderice

        if field_ref:
            # Shake animation
            field_ref.classes(add='shake-error input-error')
            field_ref.run_method('focus')

            # Scroll al campo
            field_ref.run_method('scrollIntoView', {'behavior': 'smooth', 'block': 'center'})

            # Quitar clases después de la animación
            async def remove_error_classes():
                await asyncio.sleep(0.5)
                if field_ref:
                    field_ref.classes(remove='shake-error input-error')
            asyncio.create_task(remove_error_classes())

    async def run_pipeline():
        """Ejecuta el pipeline."""
        if not local_state['steps']:
            ui.notify('Configura al menos un paso', type='warning')
            return

        # Validar que hay un modelo seleccionado
        if not state.model:
            ui.notify('Selecciona un modelo de lenguaje', type='warning')
            # Scroll al selector de modelo
            ui.run_javascript('document.getElementById("model-selector-section")?.scrollIntoView({behavior: "smooth", block: "center"})')
            # Shake animation
            if model_selector_ref['container']:
                model_selector_ref['container'].classes(add='shake-error')
                # Quitar clase después de la animación
                async def remove_shake():
                    await asyncio.sleep(0.5)
                    if model_selector_ref['container']:
                        model_selector_ref['container'].classes(remove='shake-error')
                asyncio.create_task(remove_shake())
            # Abrir el selector en modo edición
            if model_selector_ref['switch_to_edit']:
                await model_selector_ref['switch_to_edit']()
            return

        # Validar campos obligatorios
        is_valid, error_msg, field_ref, step_idx = validate_pipeline_fields()
        if not is_valid:
            ui.notify(error_msg, type='warning')
            await shake_and_focus_field(field_ref, step_idx)
            return

        # Determinar modo de ejecución:
        # - Con datos de entrada: ejecutar N veces (una por entrada)
        # - Sin datos de entrada: ejecutar una sola vez
        input_values = local_state['input_values']
        fields = local_state['input_fields']
        has_data_entries = bool(input_values) and bool(fields) and any(len(v) > 0 for v in input_values.values())

        # === TRANSICIÓN DE ESTADO: ready → loading ===
        local_state['ready_state'].set_visibility(False)
        local_state['loading_state'].set_visibility(True)
        local_state['results_state'].set_visibility(False)
        results_container.clear()

        # Resetear barra de progreso
        local_state['progress_bar_fill'].style('width: 0%')
        local_state['progress_text'].set_text('')

        t0 = datetime.now()

        # === SISTEMA DE CALLBACKS DE PROGRESO ===
        # Cola thread-safe para comunicar actualizaciones desde el backend
        progress_queue = queue.Queue()
        stop_monitoring = threading.Event()

        def progress_callback(update: ProgressUpdate) -> None:
            """Callback que recibe actualizaciones de progreso del backend."""
            progress_queue.put(update)

        async def monitor_progress():
            """Tarea async que monitorea la cola y actualiza la UI."""
            while not stop_monitoring.is_set():
                try:
                    # Obtener update de la cola (non-blocking)
                    update = progress_queue.get_nowait()

                    # Actualizar UI según la fase
                    if update.phase in (ProgressPhase.MODEL_DOWNLOAD, ProgressPhase.MODEL_LOADING):
                        # Carga de modelo: 0-30%
                        base_pct = (update.current / max(update.total, 1)) * 30
                        local_state['progress_bar_fill'].style(f'width: {base_pct:.0f}%')
                        local_state['loading_message'].set_text(update.message)
                        local_state['progress_text'].set_text(f'{base_pct:.0f}%')

                    elif update.phase == ProgressPhase.MODEL_READY:
                        local_state['progress_bar_fill'].style('width: 30%')
                        local_state['loading_message'].set_text('Modelo cargado')
                        local_state['progress_text'].set_text('30%')

                    elif update.phase == ProgressPhase.PIPELINE_START:
                        local_state['progress_bar_fill'].style('width: 30%')
                        local_state['loading_message'].set_text(update.message)

                    elif update.phase == ProgressPhase.PIPELINE_STEP:
                        # Pasos del pipeline: 30-90%
                        step_pct = 30 + (update.current / max(update.total, 1)) * 60
                        local_state['progress_bar_fill'].style(f'width: {step_pct:.0f}%')
                        local_state['loading_message'].set_text(update.message)
                        local_state['progress_text'].set_text(f'{step_pct:.0f}%')

                    elif update.phase == ProgressPhase.PIPELINE_COMPLETE:
                        local_state['progress_bar_fill'].style('width: 90%')
                        local_state['loading_message'].set_text('Finalizando...')
                        local_state['progress_text'].set_text('90%')

                    elif update.phase == ProgressPhase.ENTRY_START:
                        # Múltiples entradas: mostrar progreso de entrada
                        entry_pct = 30 + (update.current / max(update.total, 1)) * 60
                        local_state['progress_bar_fill'].style(f'width: {entry_pct:.0f}%')
                        local_state['loading_message'].set_text(update.message)
                        local_state['progress_text'].set_text(f'{update.current}/{update.total}')

                    elif update.phase == ProgressPhase.ENTRY_COMPLETE:
                        entry_pct = 30 + (update.current / max(update.total, 1)) * 60
                        local_state['progress_bar_fill'].style(f'width: {entry_pct:.0f}%')
                        local_state['progress_text'].set_text(f'{update.current}/{update.total}')

                except queue.Empty:
                    pass

                # Pequeña pausa para no consumir CPU
                await asyncio.sleep(0.1)

        # Iniciar monitoreo de progreso
        monitor_task = asyncio.create_task(monitor_progress())

        try:
            steps = build_steps_from_config()
            if not steps:
                raise ValueError("No se pudieron construir los pasos")

            request = PipelineRequest(steps=steps, global_references={})

            PipelineUseCase = _get_pipeline_use_case()
            use_case = PipelineUseCase(state.model, progress_callback=progress_callback)

            if has_data_entries:
                # Convertir input_values (column-based) a entries (row-based) para el backend
                max_entries = max((len(vals) for vals in input_values.values()), default=0)
                data_entries = []
                for i in range(max_entries):
                    entry = {}
                    for field in fields:
                        vals = input_values.get(field, [])
                        entry[field] = vals[i] if i < len(vals) else ''
                    data_entries.append(entry)

                # Actualizar mensaje de loading
                local_state['loading_message'].set_text(f'Procesando {len(data_entries)} entradas...')

                # Modo múltiple: ejecutar para cada entrada de datos
                response = await run.io_bound(
                    use_case.execute_with_references,
                    request,
                    data_entries
                )
                elapsed = (datetime.now() - t0).total_seconds()

                # Completar barra de progreso
                local_state['progress_bar_fill'].style('width: 100%')
                local_state['progress_text'].set_text('100%')

                # === TRANSICIÓN DE ESTADO: loading → results ===
                local_state['loading_state'].set_visibility(False)
                local_state['results_state'].set_visibility(True)
                show_results(response, elapsed)
                ui.notify(f'Completado: {response.successful_entries}/{response.total_entries}', type='positive')
            else:
                # Actualizar mensaje de loading
                local_state['loading_message'].set_text('Ejecutando pipeline...')

                # Modo único: ejecutar una sola vez sin datos de entrada
                response = await run.io_bound(
                    use_case._execute,
                    request
                )
                elapsed = (datetime.now() - t0).total_seconds()

                # Completar barra de progreso
                local_state['progress_bar_fill'].style('width: 100%')
                local_state['progress_text'].set_text('100%')

                # === TRANSICIÓN DE ESTADO: loading → results ===
                local_state['loading_state'].set_visibility(False)
                local_state['results_state'].set_visibility(True)
                show_single_result(response, elapsed)
                ui.notify('Pipeline ejecutado correctamente', type='positive')

        except Exception as ex:
            # === TRANSICIÓN DE ESTADO: loading → error (dentro de results) ===
            local_state['loading_state'].set_visibility(False)
            local_state['results_state'].set_visibility(True)
            ui.notify(f'Error: {str(ex)[:100]}', type='negative')
            show_error_result(str(ex))

        finally:
            # Detener monitoreo de progreso
            stop_monitoring.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    def reset_to_ready_state():
        """Vuelve al estado inicial para ejecutar de nuevo."""
        local_state['ready_state'].set_visibility(True)
        local_state['loading_state'].set_visibility(False)
        local_state['results_state'].set_visibility(False)
        results_container.clear()

    def show_error_result(error_message: str):
        """Muestra un error en el panel de resultados con mensajes amigables."""
        results_container.clear()

        # Detectar tipos de error comunes y dar mensajes amigables
        friendly_title = 'Error en la ejecución'
        friendly_msg = error_message[:150]
        suggestion = None

        error_lower = error_message.lower()
        if 'gguf' in error_lower or 'unrecognized model' in error_lower:
            friendly_title = 'Modelo no compatible'
            friendly_msg = 'Este modelo no es compatible con el sistema.'
            suggestion = 'Usa modelos estándar de HuggingFace (no GGUF/cuantizados). Ejemplo: Qwen/Qwen2.5-0.5B-Instruct'
        elif 'connection' in error_lower or 'timeout' in error_lower:
            friendly_title = 'Error de conexión'
            friendly_msg = 'No se pudo conectar con el modelo.'
            suggestion = 'Verifica tu conexión a internet y que el modelo exista en HuggingFace.'
        elif 'out of memory' in error_lower or 'cuda' in error_lower:
            friendly_title = 'Sin memoria suficiente'
            friendly_msg = 'No hay memoria GPU/RAM suficiente para este modelo.'
            suggestion = 'Prueba con un modelo más pequeño o cierra otras aplicaciones.'
        elif 'token' in error_lower and 'auth' in error_lower:
            friendly_title = 'Error de autenticación'
            friendly_msg = 'Se requiere autenticación para este modelo.'
            suggestion = 'Algunos modelos requieren aceptar licencia en HuggingFace.'

        with results_container:
            # Panel de error centrado
            with ui.column().classes('w-full items-center justify-center py-8 px-6 gap-4'):
                # Icono de error
                with ui.element('div').classes(
                    'w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/30 '
                    'flex items-center justify-center'
                ):
                    ui.icon('error_outline', size='lg').classes('text-red-400')

                # Mensaje amigable
                ui.label(friendly_title).classes('text-lg font-medium text-red-300')
                ui.label(friendly_msg).classes('text-sm text-slate-400 text-center max-w-md')

                # Sugerencia (si hay)
                if suggestion:
                    with ui.element('div').classes('flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 max-w-md'):
                        ui.icon('lightbulb', size='xs').classes('text-amber-400 mt-0.5')
                        ui.label(suggestion).classes('text-xs text-amber-200')

                # Botón para reintentar
                with ui.button('Intentar de nuevo', icon='refresh', on_click=reset_to_ready_state).props('outline').classes('mt-2'):
                    pass

    def show_single_result(response, elapsed: float):
        """Muestra el resultado de una ejecución única del pipeline."""
        results_container.clear()
        with results_container:
            # === HEADER DE ÉXITO ===
            with ui.row().classes('w-full items-center justify-between p-4 bg-emerald-500/10 border-b border-emerald-500/20'):
                with ui.row().classes('items-center gap-3'):
                    with ui.element('div').classes(
                        'w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center'
                    ):
                        ui.icon('check_circle', size='sm').classes('text-emerald-400')
                    with ui.column().classes('gap-0'):
                        ui.label('Pipeline completado').classes('text-base font-semibold text-emerald-300')
                        ui.label(f'{len(response.step_results)} pasos ejecutados en {elapsed:.1f}s').classes('text-xs text-slate-400')

                # Acciones del header
                with ui.row().classes('items-center gap-2'):
                    def do_export():
                        export = {
                            'timestamp': datetime.now().isoformat(),
                            'model': state.model,
                            'steps': len(response.step_results),
                            'time': elapsed,
                            'results': response.step_results
                        }
                        ui.download(
                            json.dumps(export, indent=2, ensure_ascii=False, default=str).encode('utf-8'),
                            f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                        )
                    ui.button(icon='download', on_click=do_export).props('flat dense').classes(
                        'text-slate-400 hover:text-emerald-400'
                    ).tooltip('Exportar JSON')
                    ui.button(icon='refresh', on_click=reset_to_ready_state).props('flat dense').classes(
                        'text-slate-400 hover:text-emerald-400'
                    ).tooltip('Ejecutar de nuevo')

            # === CONTENIDO DE RESULTADOS ===
            with ui.column().classes('w-full p-4 gap-4'):
                # Métricas en fila
                with ui.row().classes('gap-3 flex-wrap'):
                    for label, value, color, icon_name in [
                        ('Pasos', str(len(response.step_results)), 'emerald', 'route'),
                        ('Tiempo', f'{elapsed:.1f}s', 'amber', 'schedule'),
                    ]:
                        with ui.element('div').classes(f'flex items-center gap-2 px-3 py-2 rounded-lg bg-{color}-500/10 border border-{color}-500/20'):
                            ui.icon(icon_name, size='xs').classes(f'text-{color}-400')
                            ui.label(value).classes(f'text-sm font-semibold text-{color}-300')
                            ui.label(label).classes('text-xs text-slate-400')

                # Verificación (si hay datos)
                confirmed = response.verification_references.get('confirmed', [])
                to_verify = response.verification_references.get('to_verify', [])

                if confirmed or to_verify:
                    with ui.row().classes('gap-3 w-full'):
                        # Confirmados
                        with ui.element('div').classes(
                            'flex-1 p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20'
                        ):
                            with ui.row().classes('items-center gap-2 mb-2'):
                                ui.icon('verified', size='xs').classes('text-emerald-400')
                                ui.label(f'{len(confirmed)} confirmados').classes('text-sm font-medium text-emerald-300')
                            if confirmed:
                                for ref in confirmed[:3]:
                                    ui.label(str(ref)[:80]).classes('text-xs text-slate-500 truncate')
                                if len(confirmed) > 3:
                                    ui.label(f'+{len(confirmed) - 3} más').classes('text-xs text-slate-500 italic')

                        # Por revisar
                        with ui.element('div').classes(
                            'flex-1 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20'
                        ):
                            with ui.row().classes('items-center gap-2 mb-2'):
                                ui.icon('pending', size='xs').classes('text-amber-400')
                                ui.label(f'{len(to_verify)} por revisar').classes('text-sm font-medium text-amber-300')
                            if to_verify:
                                for ref in to_verify[:3]:
                                    ui.label(str(ref)[:80]).classes('text-xs text-slate-500 truncate')
                                if len(to_verify) > 3:
                                    ui.label(f'+{len(to_verify) - 3} más').classes('text-xs text-slate-500 italic')

                # Detalles de cada paso (colapsable)
                if response.step_results:
                    with ui.expansion('Ver detalles de pasos', icon='code').classes('w-full').props('dense'):
                        for step_idx, step_result in enumerate(response.step_results):
                            step_type = step_result.get('type', 'generate')
                            step_icons = {'generate': 'auto_awesome', 'parse': 'find_in_page', 'verify': 'verified'}

                            with ui.element('div').classes('p-2 mb-2 rounded bg-slate-800/30 border border-slate-700/50'):
                                with ui.row().classes('items-center gap-2 mb-1'):
                                    ui.icon(step_icons.get(step_type, 'check'), size='xs').classes('text-slate-400')
                                    ui.label(f'Paso {step_idx + 1}: {step_type}').classes('text-xs font-medium text-slate-300')
                                ui.label(json.dumps(step_result, indent=2, ensure_ascii=False, default=str)[:500]).classes(
                                    'whitespace-pre font-mono text-xs text-slate-500 overflow-auto max-h-32'
                                )

    def show_results(response, elapsed: float):
        """Muestra los resultados del pipeline (múltiples entradas)."""
        results_container.clear()

        # Calcular estadísticas
        total = response.total_entries
        successful = response.successful_entries
        failed = response.failed_entries
        success_rate = (successful / total * 100) if total > 0 else 0
        is_success = failed == 0

        with results_container:
            # === HEADER DE RESULTADOS ===
            header_bg = 'bg-emerald-500/10 border-emerald-500/20' if is_success else 'bg-amber-500/10 border-amber-500/20'
            header_icon_bg = 'bg-emerald-500/20' if is_success else 'bg-amber-500/20'
            header_icon_color = 'text-emerald-400' if is_success else 'text-amber-400'
            header_text_color = 'text-emerald-300' if is_success else 'text-amber-300'

            with ui.row().classes(f'w-full items-center justify-between p-4 {header_bg} border-b'):
                with ui.row().classes('items-center gap-3'):
                    with ui.element('div').classes(f'w-10 h-10 rounded-xl {header_icon_bg} flex items-center justify-center'):
                        icon_name = 'check_circle' if is_success else 'warning'
                        ui.icon(icon_name, size='sm').classes(header_icon_color)
                    with ui.column().classes('gap-0'):
                        status_text = 'Procesamiento completado' if is_success else f'Completado con {failed} errores'
                        ui.label(status_text).classes(f'text-base font-semibold {header_text_color}')
                        ui.label(f'{total} entradas procesadas en {elapsed:.1f}s').classes('text-xs text-slate-400')

                # Acciones del header
                confirmed = response.verification_references.get('confirmed', [])
                to_verify = response.verification_references.get('to_verify', [])

                with ui.row().classes('items-center gap-2'):
                    def do_export():
                        export = {
                            'timestamp': datetime.now().isoformat(),
                            'model': state.model,
                            'total': total,
                            'successful': successful,
                            'failed': failed,
                            'success_rate': success_rate,
                            'time': elapsed,
                            'confirmed': confirmed,
                            'to_verify': to_verify,
                            'results': response.step_results[:50] if response.step_results else []
                        }
                        ui.download(
                            json.dumps(export, indent=2, ensure_ascii=False).encode('utf-8'),
                            f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                        )
                    ui.button(icon='download', on_click=do_export).props('flat dense').classes(
                        'text-slate-400 hover:text-emerald-400'
                    ).tooltip('Exportar JSON')
                    ui.button(icon='refresh', on_click=reset_to_ready_state).props('flat dense').classes(
                        'text-slate-400 hover:text-emerald-400'
                    ).tooltip('Ejecutar de nuevo')

            # === MÉTRICAS PRINCIPALES ===
            with ui.column().classes('w-full p-4 gap-4'):
                with ui.row().classes('gap-3 flex-wrap justify-center'):
                    # Total
                    with ui.element('div').classes('flex flex-col items-center px-5 py-3 rounded-xl bg-slate-500/10 border border-slate-500/20 min-w-24'):
                        ui.label(str(total)).classes('text-2xl font-bold text-slate-200')
                        ui.label('Total').classes('text-xs text-slate-400 uppercase')

                    # Exitosos
                    with ui.element('div').classes('flex flex-col items-center px-5 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 min-w-24'):
                        ui.label(str(successful)).classes('text-2xl font-bold text-emerald-400')
                        ui.label('Exitosos').classes('text-xs text-slate-400 uppercase')

                    # Fallidos
                    with ui.element('div').classes('flex flex-col items-center px-5 py-3 rounded-xl bg-red-500/10 border border-red-500/20 min-w-24'):
                        ui.label(str(failed)).classes('text-2xl font-bold text-red-400')
                        ui.label('Fallidos').classes('text-xs text-slate-400 uppercase')

                    # Tasa de éxito
                    with ui.element('div').classes('flex flex-col items-center px-5 py-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 min-w-24'):
                        ui.label(f'{success_rate:.0f}%').classes('text-2xl font-bold text-indigo-400')
                        ui.label('Tasa').classes('text-xs text-slate-400 uppercase')

                    # Tiempo
                    with ui.element('div').classes('flex flex-col items-center px-5 py-3 rounded-xl bg-amber-500/10 border border-amber-500/20 min-w-24'):
                        ui.label(f'{elapsed:.1f}s').classes('text-2xl font-bold text-amber-400')
                        ui.label('Tiempo').classes('text-xs text-slate-400 uppercase')

                # === VERIFICACIÓN ===
                if confirmed or to_verify:
                    with ui.row().classes('gap-3 w-full'):
                        # Confirmados
                        with ui.element('div').classes(
                            'flex-1 p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20'
                        ):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('verified', size='xs').classes('text-emerald-400')
                                ui.label(f'{len(confirmed)} confirmados').classes('text-sm font-medium text-emerald-300')

                        # Por revisar
                        with ui.element('div').classes(
                            'flex-1 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20'
                        ):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('pending', size='xs').classes('text-amber-400')
                                ui.label(f'{len(to_verify)} por revisar').classes('text-sm font-medium text-amber-300')

                # === DETALLES (colapsable) ===
                if response.step_results:
                    with ui.expansion(f'Ver {len(response.step_results)} resultados detallados', icon='list').classes('w-full').props('dense'):
                        for i, sr in enumerate(response.step_results[:20]):
                            stype = sr.get('step_type', '?')
                            sdata = sr.get('step_data', [])
                            with ui.element('div').classes('p-2 mb-2 rounded bg-slate-800/30 border border-slate-700/50'):
                                with ui.row().classes('items-center gap-2 mb-1'):
                                    ui.label(f'#{i+1}').classes('text-xs font-mono text-slate-500 bg-slate-700/50 px-1.5 py-0.5 rounded')
                                    ui.label(stype).classes('text-xs font-medium text-slate-300 bg-indigo-500/20 px-2 py-0.5 rounded')
                                if sdata:
                                    txt = str(sdata[0])[:200]
                                    ui.label(txt + ('...' if len(str(sdata[0])) > 200 else '')).classes(
                                        'text-xs text-slate-500 font-mono'
                                    )
                        if len(response.step_results) > 20:
                            ui.label(f'... y {len(response.step_results) - 20} más (exportar para ver todos)').classes(
                                'text-xs text-slate-500 italic text-center py-2'
                            )

    # === LAYOUT PRINCIPAL ===
    with ui.column().classes('w-full max-w-6xl mx-auto gap-4 p-6'):

        # Nota: La precarga de modelos se inicia en main() para que empiece
        # antes de que el usuario llegue a esta pestaña (lazy loading)

        # Header: Título
        with ui.row().classes('items-center gap-3 mb-2'):
            with ui.element('div').classes('w-10 h-10 rounded-xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center'):
                ui.icon('account_tree', size='sm').classes('text-purple-400')
            with ui.column().classes('gap-0'):
                ui.label('Pipeline').classes('text-xl font-bold')
                ui.label('Constructor visual de flujos').classes('text-xs text-slate-400')

        # Referencia al input de modelo (se asigna más abajo)
        model_input_ref = {'input': None}

        def select_custom_pipeline():
            """Selecciona pipeline personalizado y enfoca el buscador de modelo."""
            select_template('personalizado')
            # Enfocar el input de modelo después de un pequeño delay para que se actualice la UI
            if model_input_ref['input']:
                model_input_ref['input'].run_method('focus')

        # Sección de Plantillas - Compacta en una sola fila
        with ui.row().classes('w-full items-center gap-2 flex-wrap'):
            with ui.row().classes('items-center gap-2 mr-5'):
                ui.icon('widgets', size='xs').classes('text-purple-400')
                with ui.column().classes('gap-0'):
                    ui.label('Inicio rápido').classes('text-sm font-medium text-slate-300')
                    ui.label('Plantillas predefinidas').classes('text-xs text-slate-400')

            # Plantillas predefinidas (excluye 'personalizado')
            for key, tmpl in PIPELINE_TEMPLATES.items():
                if key == 'personalizado':
                    continue
                icon = tmpl.get('icon', 'build')
                is_selected = local_state['selected_template'] == key

                if is_selected:
                    btn_style = 'bg-purple-500/20 border-purple-500 text-purple-200'
                    icon_color = 'text-purple-300'
                else:
                    btn_style = 'bg-slate-700/30 border-slate-600/50 text-slate-300 hover:bg-purple-500/10 hover:border-purple-500/50'
                    icon_color = 'text-slate-400'

                with ui.element('button').classes(f'flex items-center gap-1.5 px-2 py-1 rounded border {btn_style} transition-all cursor-pointer').on('click', lambda k=key: select_template(k)):
                    ui.icon(icon, size='xs').classes(icon_color)
                    ui.label(tmpl['name']).classes('text-xs font-medium')
                    if is_selected:
                        ui.icon('check', size='xs').classes('text-purple-400')

            # Separador vertical
            ui.element('div').classes('w-px h-5 bg-slate-600/50 mx-1')

            # Botón "Desde cero" - estilo diferente
            is_custom_selected = local_state['selected_template'] == 'personalizado'
            if is_custom_selected:
                custom_style = 'bg-purple-500/20 border-purple-500 text-purple-200'
                custom_icon = 'text-purple-300'
            else:
                custom_style = 'bg-slate-800/50 border-dashed border-slate-500/50 text-slate-400 hover:bg-purple-500/10 hover:border-purple-500/50 hover:text-slate-300'
                custom_icon = 'text-slate-400'

            with ui.element('button').classes(f'flex items-center gap-1.5 px-2 py-1 rounded border {custom_style} transition-all cursor-pointer').on('click', select_custom_pipeline):
                ui.icon('add', size='xs').classes(custom_icon)
                ui.label('Desde cero').classes('text-xs font-medium')
                if is_custom_selected:
                    ui.icon('check', size='xs').classes('text-purple-400')

        # Sección: Variables de Entrada (compacta, con toggle)
        with ui.card().props('flat').classes('w-full bg-slate-800/30 border border-purple-500/20 px-4 py-3'):
            data_container = ui.column().classes('w-full')
            entries_counter_label = None  # Ya no se usa, el contador está integrado en render_data_section
            with data_container:
                render_data_section()

        # Sección: Constructor de Pipeline - Layout 2 columnas (púrpura - paso 2 Configura)
        with ui.card().props('flat').classes('w-full bg-slate-800/30 border border-purple-500/20 p-4'):
            with ui.row().classes('w-full justify-between items-center mb-1'):
                with ui.row().classes('items-center gap-2'):
                    with ui.element('div').classes('w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center'):
                        ui.icon('build', size='xs').classes('text-purple-400')
                    with ui.column().classes('gap-0'):
                        ui.label('Constructor de Pipeline').classes('text-base font-semibold text-purple-300')
                        steps_counter_label = ui.label(f'{len(local_state["steps"])} pasos configurados').classes('text-xs text-slate-400')

                # Botones de acción (JSON view, exportar)
                with ui.row().classes('shrink-0 gap-1'):
                    with ui.button(
                        icon='code' if not local_state['pipeline_json_view'] else 'view_list',
                        on_click=toggle_pipeline_json_view
                    ).props('flat round size=md color=none').classes('!text-purple-400 hover:!text-purple-300'):
                        ui.tooltip('Ver como JSON' if not local_state['pipeline_json_view'] else 'Ver como constructor')
                    with ui.button(icon='download', on_click=export_pipeline_json).props('flat round size=md color=none').classes('!text-purple-400 hover:!text-purple-300'):
                        ui.tooltip('Exportar como JSON')

            # Selector de modelo LLM (dentro del constructor)
            with ui.column().classes('w-full gap-2 mb-1'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('psychology', size='xs').classes('text-slate-400')
                    ui.label('Modelo de Lenguaje').classes('text-sm font-medium text-slate-300 uppercase')

                # Estado del selector: si hay modelo seleccionado, mostrar vista compacta
                model_selector_state = {'editing': not bool(state.model)}

                # Contenedor principal del selector
                model_selector_container = ui.column().classes('w-full').props('id="model-selector-section"')

                # Guardar referencia global para acceso desde run_pipeline
                model_selector_ref['container'] = model_selector_container

                # Referencias compartidas
                model_input_local = {'ref': None}
                search_indicator_local = {'ref': None}
                model_dropdown_local = {'ref': None}
                search_state = {'last_query': '', 'results': []}
                search_debounce = {'task': None}

                # Diálogo para modelo local
                local_model_dialog = ui.dialog()
                local_path_input = None

                def switch_to_selected_view():
                    """Cambia a vista de modelo seleccionado."""
                    model_selector_state['editing'] = False
                    render_model_selector()
                    # Actualizar checklist de ejecución si existe
                    if 'update_execution_checklist' in local_state and local_state['update_execution_checklist']:
                        local_state['update_execution_checklist']()

                async def switch_to_editing_view():
                    """Cambia a vista de edición/selección."""
                    model_selector_state['editing'] = True
                    render_model_selector()
                    # Esperar a que se renderice el input y poner foco al final
                    await asyncio.sleep(0.05)
                    inp = model_input_local['ref']
                    if inp:
                        inp.run_method('focus')
                        if inp.value:
                            length = len(inp.value)
                            inp.run_method('setSelectionRange', length, length)

                # Guardar referencia a la función para acceso desde run_pipeline
                model_selector_ref['switch_to_edit'] = switch_to_editing_view

                def apply_local_model():
                    """Aplica la ruta del modelo local."""
                    if local_path_input and local_path_input.value:
                        path = local_path_input.value.strip()
                        if path:
                            state.model = path
                            ui.notify(f'Modelo local: {path}', type='positive')
                            local_model_dialog.close()
                            switch_to_selected_view()

                with local_model_dialog:
                    with ui.card().classes('w-[500px] bg-slate-800/95 border border-purple-500/30 p-4'):
                        with ui.row().classes('items-center gap-2 mb-3'):
                            with ui.element('div').classes('w-7 h-7 rounded-lg bg-purple-500/20 flex items-center justify-center'):
                                ui.icon('folder_open', size='xs').classes('text-purple-400')
                            ui.label('Cargar Modelo Local').classes('text-base font-semibold text-purple-200')

                        def browse_folder():
                            """Abre un diálogo para seleccionar carpeta."""
                            try:
                                import tkinter as tk
                                from tkinter import filedialog
                                root = tk.Tk()
                                root.withdraw()
                                root.attributes('-topmost', True)
                                folder = filedialog.askdirectory(title='Seleccionar carpeta del modelo')
                                root.destroy()
                                if folder and local_path_input:
                                    local_path_input.value = folder
                            except Exception:
                                ui.notify('Copia la ruta manualmente', type='info')

                        ui.label('Ruta completa a la carpeta del modelo:').classes('text-sm text-slate-300')
                        with ui.row().classes('w-full gap-2 mb-2 items-center'):
                            local_path_input = ui.input(
                                placeholder='C:/modelos/mi-modelo o /home/user/modelos/llama'
                            ).props('dense dark outlined color=purple').classes('flex-1')
                            ui.button(icon='folder_open', on_click=browse_folder).props('flat color=none').classes('!text-purple-400 hover:!text-purple-300').tooltip('Seleccionar carpeta')

                        ui.label('La carpeta debe contener:').classes('text-xs text-slate-400 -mb-1')
                        with ui.column().classes('gap-0 mb-3 pl-3'):
                            ui.label('• config.json').classes('text-xs text-slate-400')
                            ui.label('• Tokenizer (tokenizer.json, tokenizer.model o tokenizer_config.json)').classes('text-xs text-slate-400')
                            ui.label('• Pesos del modelo (.safetensors o .bin)').classes('text-xs text-slate-400')

                        with ui.row().classes('justify-end gap-2'):
                            ui.button('Cancelar', on_click=local_model_dialog.close).props('flat color=none').classes('!text-slate-400 hover:!text-slate-300')
                            ui.button('Aplicar', on_click=apply_local_model).props('flat color=none').classes('!bg-purple-500/20 !text-purple-300 hover:!bg-purple-500/30')

                def open_local_model_dialog():
                    if local_path_input:
                        local_path_input.value = ''
                    local_model_dialog.open()

                def render_dropdown_items(results: List[Dict]):
                    """Renderiza items del dropdown de búsqueda."""
                    dropdown = model_dropdown_local['ref']
                    if not dropdown:
                        return
                    search_state['results'] = results  # Guardar resultados para selección con Enter
                    dropdown.clear()
                    with dropdown:
                        for r in results:
                            def make_click(val=r['value'], info=r):
                                def click():
                                    state.model = val
                                    state.model_info = info  # Guardar info completa del modelo
                                    dropdown.set_visibility(False)
                                    switch_to_selected_view()
                                return click

                            compat = r.get('compat_status')
                            if compat == 'compatible':
                                bg_hover = 'hover:bg-emerald-500/15'
                                badge_class = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                                badge_text = 'Compatible'
                                vram_color = 'text-emerald-400'
                            elif compat == 'limite':
                                bg_hover = 'hover:bg-amber-500/15'
                                badge_class = 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                                badge_text = 'Al límite'
                                vram_color = 'text-amber-400'
                            elif compat == 'incompatible':
                                bg_hover = 'hover:bg-red-500/15'
                                badge_class = 'bg-red-500/20 text-red-300 border-red-500/30'
                                badge_text = 'Incompatible'
                                vram_color = 'text-red-400'
                            else:
                                bg_hover = 'hover:bg-slate-600/30'
                                badge_class = 'bg-slate-600/30 text-slate-400 border-slate-500/30'
                                badge_text = '?'
                                vram_color = 'text-slate-400'

                            model_id = r.get('model_id', '')
                            hf_url = f'https://huggingface.co/{model_id}'

                            with ui.element('div').classes(
                                f'w-full px-3 py-2 cursor-pointer {bg_hover} transition-all border-b border-slate-700/30'
                            ).on('click', make_click()):
                                with ui.row().classes('items-center justify-between w-full gap-2'):
                                    ui.label(model_id).classes('text-sm font-medium text-slate-100 truncate flex-1 min-w-0')
                                    with ui.element('span').classes(f'px-2 py-0.5 text-xs rounded border flex-shrink-0 {badge_class}'):
                                        ui.label(badge_text)
                                with ui.row().classes('items-center justify-between w-full mt-1'):
                                    with ui.row().classes('items-center gap-3'):
                                        ui.label(f"Params: {r.get('params_str', '?')}").classes('text-xs text-slate-400')
                                        ui.label(f"VRAM: ~{r.get('vram_str', '?')}").classes(f'text-xs {vram_color}')
                                        if r.get('date'):
                                            ui.label(f"Fecha: {r.get('date')}").classes('text-xs text-slate-400')
                                        ui.label(f"Descargas: {r.get('downloads', '?')}").classes('text-xs text-slate-400')
                                    with ui.link(target=hf_url, new_tab=True).classes('flex items-center gap-1 text-xs text-indigo-400 no-underline hover:text-indigo-300'):
                                        ui.label('Ver en HuggingFace')
                                        ui.icon('open_in_new', size='xs')
                    dropdown.set_visibility(True)

                async def do_model_search(query: str):
                    """Ejecuta la búsqueda en HuggingFace."""
                    dropdown = model_dropdown_local['ref']
                    indicator = search_indicator_local['ref']
                    if not dropdown:
                        return
                    if len(query) < 2:
                        dropdown.set_visibility(False)
                        return

                    # Mostrar indicador de búsqueda
                    if indicator:
                        indicator.set_visibility(True)
                    dropdown.clear()
                    with dropdown:
                        with ui.row().classes('items-center gap-2 p-3'):
                            ui.spinner('dots', size='xs').classes('text-indigo-400')
                            ui.label('Buscando...').classes('text-sm text-slate-400')
                    dropdown.set_visibility(True)

                    try:
                        if query != search_state['last_query']:
                            return
                        results = await search_huggingface_models(query)
                        if query == search_state['last_query']:
                            if results:
                                render_dropdown_items(results)
                            else:
                                dropdown.clear()
                                with dropdown:
                                    ui.label('Sin resultados').classes('text-sm text-slate-400 p-3')
                                dropdown.set_visibility(True)
                    except Exception:
                        pass
                    finally:
                        if indicator:
                            indicator.set_visibility(False)

                async def on_model_search_debounced():
                    """Búsqueda con debouncing real de 300ms."""
                    model_input = model_input_local['ref']
                    dropdown = model_dropdown_local['ref']
                    if not model_input or not dropdown:
                        return
                    query = model_input.value.strip() if model_input.value else ''
                    search_state['last_query'] = query

                    # Cancelar búsqueda anterior si existe
                    if search_debounce['task'] and not search_debounce['task'].done():
                        search_debounce['task'].cancel()

                    if len(query) < 2:
                        dropdown.set_visibility(False)
                        return

                    async def debounced_search():
                        try:
                            await asyncio.sleep(0.3)  # Esperar 300ms de inactividad
                            if query == search_state['last_query']:
                                await do_model_search(query)
                        except asyncio.CancelledError:
                            pass  # Búsqueda cancelada por nueva tecla

                    search_debounce['task'] = asyncio.create_task(debounced_search())

                def render_model_selector():
                    """Renderiza el selector de modelo según el estado."""
                    model_selector_container.clear()
                    with model_selector_container:
                        if model_selector_state['editing'] or not state.model:
                            # Vista de edición: buscador + botón local
                            with ui.row().classes('w-2/3 items-stretch gap-2'):
                                # Buscador de HuggingFace
                                with ui.element('div').classes('flex-1 relative'):
                                    with ui.element('div').classes(
                                        'w-full h-11 flex items-center gap-3 px-3 bg-slate-900/50 '
                                        'border border-slate-600/50 rounded-lg hover:border-purple-500/50 '
                                        'focus-within:border-purple-500 transition-all'
                                    ):
                                        ui.icon('search', size='xs').classes('text-slate-400')
                                        model_input = ui.input(
                                            value=state.model,
                                            placeholder='Buscar modelo en HuggingFace...'
                                        ).props('borderless dense dark').classes('flex-1 text-sm')
                                        model_input_local['ref'] = model_input
                                        model_input_ref['input'] = model_input
                                        search_indicator = ui.spinner('dots', size='sm').classes('text-purple-400')
                                        search_indicator.set_visibility(False)
                                        search_indicator_local['ref'] = search_indicator

                                    # Dropdown de resultados
                                    model_dropdown = ui.column().classes(
                                        'absolute top-full left-0 right-0 w-full mt-1 bg-slate-800/95 backdrop-blur-sm '
                                        'border border-slate-600/50 rounded-lg shadow-2xl z-50 max-h-72 overflow-y-auto'
                                    )
                                    model_dropdown.set_visibility(False)
                                    model_dropdown_local['ref'] = model_dropdown

                                # Event handlers
                                async def on_blur():
                                    await asyncio.sleep(0.15)
                                    if model_dropdown_local['ref']:
                                        model_dropdown_local['ref'].set_visibility(False)
                                model_input.on('blur', on_blur)

                                async def on_focus():
                                    """Al recuperar focus, re-mostrar dropdown si hay texto."""
                                    inp = model_input_local['ref']
                                    if inp and inp.value and len(inp.value.strip()) >= 2:
                                        await on_model_search_debounced()
                                model_input.on('focus', on_focus)

                                def on_model_change(e):
                                    if e.args:
                                        state.model = e.args
                                model_input.on('update:model-value', on_model_change)
                                model_input.on('keyup', lambda: on_model_search_debounced())

                                def on_enter(e):
                                    """Seleccionar modelo si hay coincidencia exacta al presionar Enter."""
                                    if e.args.get('key') != 'Enter':
                                        return
                                    inp = model_input_local['ref']
                                    if not inp or not inp.value:
                                        return
                                    query = inp.value.strip().lower()
                                    # Buscar coincidencia exacta en los resultados actuales
                                    for r in search_state.get('results', []):
                                        model_id = r.get('model_id', '').lower()
                                        # Coincidencia exacta con ID completo o nombre corto
                                        short_name = model_id.split('/')[-1] if '/' in model_id else model_id
                                        if query == model_id or query == short_name:
                                            state.model = r['value']
                                            state.model_info = r
                                            if model_dropdown_local['ref']:
                                                model_dropdown_local['ref'].set_visibility(False)
                                            switch_to_selected_view()
                                            return
                                model_input.on('keydown', on_enter)

                                # Separador
                                ui.label('o').classes('text-sm text-slate-400 px-2 self-center')

                                # Botón para modelo local
                                with ui.button(on_click=open_local_model_dialog).props('flat dense').classes(
                                    'h-11 px-4 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                                    'hover:bg-purple-500/10 hover:border-purple-500/50 transition-all'
                                ):
                                    with ui.row().classes('items-center gap-2'):
                                        ui.icon('folder_open', size='xs').classes('text-purple-400')
                                        ui.label('Modelo Local').classes('text-xs text-slate-300 leading-none')
                        else:
                            # Vista de modelo seleccionado
                            # Determinar estilo según compatibilidad
                            compat = state.model_info.get('compat_status') if state.model_info else None
                            vram_str = state.model_info.get('vram_str', '?') if state.model_info else '?'
                            params_str = state.model_info.get('params_str', '?') if state.model_info else '?'
                            available_vram = get_available_vram_gb()

                            if compat == 'compatible':
                                bg_class = 'bg-emerald-500/10'
                                border_class = 'border-emerald-500/30'
                                icon_name = 'check_circle'
                                icon_color = 'text-emerald-400'
                                text_color = 'text-emerald-200'
                                badge_text = None  # No mostrar badge si es compatible
                                badge_class = ''
                                tooltip_text = f'{state.model}\n✓ Compatible con tu sistema\nVRAM: ~{vram_str} | Params: {params_str}'
                            elif compat == 'limite':
                                bg_class = 'bg-amber-500/10'
                                border_class = 'border-amber-500/30'
                                icon_name = 'warning'
                                icon_color = 'text-amber-400'
                                text_color = 'text-amber-200'
                                badge_text = 'LÍMITE'
                                badge_class = 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                                tooltip_text = f'{state.model}\n⚠ Al límite de tu VRAM ({available_vram:.1f}GB)\nNecesita ~{vram_str} | Puede funcionar lento'
                            elif compat == 'incompatible':
                                bg_class = 'bg-red-500/10'
                                border_class = 'border-red-500/30'
                                icon_name = 'error'
                                icon_color = 'text-red-400'
                                text_color = 'text-red-200'
                                badge_text = 'NO RECOMENDADO'
                                badge_class = 'bg-red-500/20 text-red-300 border border-red-500/30'
                                tooltip_text = f'{state.model}\n✗ Excede tu VRAM ({available_vram:.1f}GB)\nNecesita ~{vram_str} | Probablemente no funcione'
                            else:
                                # Sin info de compatibilidad (modelo local u otro)
                                bg_class = 'bg-emerald-500/10'
                                border_class = 'border-emerald-500/30'
                                icon_name = 'check_circle'
                                icon_color = 'text-emerald-400'
                                text_color = 'text-emerald-200'
                                badge_text = None
                                badge_class = ''
                                tooltip_text = state.model

                            with ui.row().classes('w-2/3 items-center gap-3'):
                                with ui.element('div').classes(
                                    f'flex-1 h-11 flex items-center gap-3 px-4 {bg_class} '
                                    f'border {border_class} rounded-lg cursor-pointer '
                                    'hover:opacity-80 transition-opacity'
                                ).on('click', switch_to_editing_view):
                                    ui.icon(icon_name, size='xs').classes(icon_color)
                                    # Mostrar nombre corto del modelo
                                    model_name = state.model.split('/')[-1] if '/' in state.model else state.model
                                    if len(model_name) > 65:
                                        model_name = model_name[:62] + '...'
                                    ui.label(model_name).classes(f'text-sm font-medium {text_color} truncate flex-1')
                                    # Badge de estado si aplica
                                    if badge_text:
                                        with ui.element('span').classes(
                                            f'px-2 py-0.5 text-xs rounded {badge_class} flex-shrink-0'
                                        ):
                                            ui.label(badge_text).classes('text-xs')
                                    # Tooltip con detalles
                                    ui.tooltip(tooltip_text).classes('bg-slate-800 text-slate-200 whitespace-pre-line')

                                with ui.button(on_click=switch_to_editing_view).props('flat dense').classes(
                                    'h-11 px-4 bg-slate-700/50 border border-slate-500/50 rounded-lg '
                                    'hover:bg-purple-500/10 hover:border-purple-500/50 transition-all'
                                ):
                                    with ui.row().classes('items-center gap-2'):
                                        ui.icon('swap_horiz', size='xs').classes('text-purple-400')
                                        ui.label('Cambiar').classes('text-xs text-slate-300')

                # Renderizar vista inicial
                render_model_selector()

            # Sección: Pasos del Pipeline (misma estructura que Selecciona Modelo)
            with ui.column().classes('w-full gap-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('route', size='xs').classes('text-slate-400')
                    ui.label('Pasos del Pipeline').classes('text-sm font-medium text-slate-300 uppercase')

                # Contenedor de pasos (accordion)
                steps_list_container = ui.column().classes('w-full gap-2')
                with steps_list_container:
                    render_steps_accordion()

        # =================================================================
        # SECCIÓN DE EJECUCIÓN Y RESULTADOS (Rediseño UX Profesional)
        # =================================================================
        # Panel unificado con 3 estados: ready → loading → results

        def get_pipeline_status_summary():
            """Obtiene un resumen del estado actual del pipeline."""
            num_steps = len(local_state['steps'])
            has_model = bool(state.model)
            input_values = local_state.get('input_values', {})
            fields = local_state.get('input_fields', [])
            input_vars_enabled = local_state.get('input_vars_enabled', False)
            num_entries = 0
            # Solo contar entradas si las variables de entrada están habilitadas
            if input_vars_enabled and input_values and fields:
                num_entries = max((len(v) for v in input_values.values()), default=0)
            return {
                'steps': num_steps,
                'has_model': has_model,
                'model_name': state.model if has_model else None,
                'entries': num_entries,
                'is_ready': num_steps > 0 and has_model
            }

        # Card principal (mismo estilo que Constructor de Pipeline)
        with ui.card().props('flat').classes('w-full bg-slate-800/30 border border-emerald-500/20 p-4'):
            # Header dentro de la caja
            with ui.row().classes('w-full justify-between items-center mb-4'):
                with ui.row().classes('items-center gap-2'):
                    with ui.element('div').classes('w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center'):
                        ui.icon('play_circle', size='xs').classes('text-emerald-400')
                    with ui.column().classes('gap-0'):
                        ui.label('Ejecutar Pipeline').classes('text-base font-semibold text-emerald-300')
                        execution_subtitle = ui.label('Configura el pipeline para ejecutar').classes('text-xs text-slate-400')

            # --- ESTADO: LISTO PARA EJECUTAR ---
            ready_state = ui.column().classes('w-full gap-4')
            with ready_state:
                config_checklist = ui.column().classes('w-full gap-2 p-4 rounded-lg bg-slate-900/50 border border-slate-700/50')

                def update_config_checklist():
                    """Actualiza la checklist de configuración."""
                    status = get_pipeline_status_summary()
                    config_checklist.clear()
                    with config_checklist:
                        # Modelo
                        with ui.row().classes('items-center gap-3'):
                            if status['has_model']:
                                ui.icon('check_circle', size='xs').classes('text-emerald-400')
                                model_display = status['model_name']
                                if len(model_display) > 100:
                                    model_display = model_display[:97] + '...'
                                ui.label(f'Modelo: {model_display}').classes('text-sm text-slate-300')
                            else:
                                ui.icon('radio_button_unchecked', size='xs').classes('text-slate-500')
                                ui.label('Modelo: No seleccionado').classes('text-sm text-slate-500')

                        # Pasos
                        with ui.row().classes('items-center gap-3'):
                            if status['steps'] > 0:
                                ui.icon('check_circle', size='xs').classes('text-emerald-400')
                                ui.label(f'Pipeline: {status["steps"]} paso{"s" if status["steps"] != 1 else ""} configurado{"s" if status["steps"] != 1 else ""}').classes('text-sm text-slate-300')
                            else:
                                ui.icon('radio_button_unchecked', size='xs').classes('text-slate-500')
                                ui.label('Pipeline: Sin pasos configurados').classes('text-sm text-slate-500')

                        # Datos de entrada
                        with ui.row().classes('items-center gap-3'):
                            if status['entries'] > 0:
                                ui.icon('check_circle', size='xs').classes('text-emerald-400')
                                ui.label(f'Variables: {status["entries"]} entrada{"s" if status["entries"] != 1 else ""}').classes('text-sm text-slate-300')
                            else:
                                ui.icon('info', size='xs').classes('text-slate-500')
                                ui.label('Sin variables establecidas (ejecución simple)').classes('text-sm text-slate-500 italic')

                    # Actualizar subtítulo
                    if status['is_ready']:
                        if status['entries'] > 0:
                            execution_subtitle.set_text(f'Listo para procesar {status["entries"]} entrada{"s" if status["entries"] != 1 else ""}')
                        else:
                            execution_subtitle.set_text('Listo para ejecutar')
                    else:
                        missing = []
                        if not status['has_model']:
                            missing.append('modelo')
                        if status['steps'] == 0:
                            missing.append('pasos')
                        execution_subtitle.set_text(f'Falta: {", ".join(missing)}')

                update_config_checklist()
                local_state['update_execution_checklist'] = update_config_checklist

                # Botón CTA principal
                with ui.row().classes('w-full justify-center pt-2'):
                    run_btn = ui.button('Ejecutar Pipeline', icon='play_arrow', on_click=run_pipeline).props('unelevated size=lg')
                    run_btn.style(
                        'background: linear-gradient(135deg, #059669, #047857) !important; '
                        'color: white !important; font-weight: 600; '
                        'border: none; padding: 14px 40px !important; font-size: 16px !important; '
                        'border-radius: 12px !important; box-shadow: 0 4px 14px rgba(5, 150, 105, 0.4) !important;'
                    )
                    run_btn.classes('hover:scale-105 transition-transform')

            # --- ESTADO: EJECUTANDO ---
            loading_state = ui.column().classes('w-full gap-4 py-2')
            loading_state.set_visibility(False)
            with loading_state:
                with ui.row().classes('items-center gap-3'):
                    ui.spinner('dots', size='md').classes('text-emerald-400')
                    with ui.column().classes('gap-0 flex-1'):
                        loading_title = ui.label('Ejecutando pipeline...').classes('text-base font-medium text-emerald-300')
                        loading_message = ui.label('Iniciando proceso...').classes('text-sm text-slate-400')

                with ui.column().classes('w-full gap-1'):
                    progress_bar_container = ui.element('div').classes('w-full h-2 bg-slate-700 rounded-full overflow-hidden')
                    with progress_bar_container:
                        progress_bar_fill = ui.element('div').classes(
                            'h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-300'
                        ).style('width: 0%;')
                    progress_text = ui.label('').classes('text-xs text-slate-500 text-right')

                intermediate_results = ui.column().classes('w-full gap-2 max-h-48 overflow-y-auto')

                local_state['loading_title'] = loading_title
                local_state['progress_bar_fill'] = progress_bar_fill
                local_state['progress_text'] = progress_text
                local_state['intermediate_results'] = intermediate_results

            # --- ESTADO: RESULTADOS ---
            results_state = ui.column().classes('w-full')
            results_state.set_visibility(False)
            results_container = ui.column().classes('w-full')

        # Guardar referencias en local_state
        local_state['ready_state'] = ready_state
        local_state['loading_state'] = loading_state
        local_state['results_state'] = results_state
        local_state['loading_message'] = loading_message
        local_state['execution_subtitle'] = execution_subtitle

        loading_box = loading_state


# =============================================================================
# BENCHMARK PAGE
# =============================================================================
def benchmark_page():
    """Página de benchmark."""

    config_content = None
    entries_content = None
    run_btn = None
    loading_row = None
    results_box = None

    def update_ui():
        has_config = state.benchmark_config is not None
        has_entries = len(state.benchmark_entries) > 0

        config_content.clear()
        with config_content:
            if has_config:
                c = state.benchmark_config
                steps = c.get('pipeline_steps', c.get('steps', []))
                ui.label(f'{len(steps)} pasos').classes('text-sm text-green-400')
                ui.label(f'{c.get("label_key")} = {c.get("label_value")}').classes('text-xs text-slate-400')
            else:
                ui.label('Sin configuración').classes('text-sm text-slate-400')

        entries_content.clear()
        with entries_content:
            if has_entries:
                n = len(state.benchmark_entries)
                ui.label(f'{n} entrada{"s" if n != 1 else ""}').classes('text-sm text-green-400')
                with ui.element('div').classes('json-preview mt-2'):
                    ui.label(format_json_preview(state.benchmark_entries[0]))
            else:
                ui.label('Sin entradas').classes('text-sm text-slate-400')

        if has_config and has_entries and not state.benchmark_running:
            run_btn.enable()
        else:
            run_btn.disable()

    def load_config():
        data = load_json_file(DEFAULT_BENCHMARK_CONFIG)
        if data:
            state.benchmark_config = data
            update_ui()
            ui.notify('Configuración cargada', type='positive')

    def load_entries():
        data = load_json_file(DEFAULT_BENCHMARK_ENTRIES)
        if data:
            state.benchmark_entries = data if isinstance(data, list) else [data]
            update_ui()
            ui.notify(f'{len(state.benchmark_entries)} entrada(s)', type='positive')

    async def run_benchmark():
        if not state.benchmark_config or not state.benchmark_entries:
            return

        state.benchmark_running = True
        run_btn.disable()
        loading_row.set_visibility(True)
        results_box.clear()

        t0 = datetime.now()

        try:
            cfg = state.benchmark_config
            steps = build_pipeline_steps(cfg)

            bc = BenchmarkConfig(
                model_name=cfg.get('model_name', state.model),
                pipeline_steps=steps,
                label_key=cfg['label_key'],
                label_value=cfg['label_value']
            )

            entries = [
                BenchmarkEntry(
                    input_data={k: v for k, v in e.items() if k != bc.label_key},
                    expected_label=str(e.get(bc.label_key, ''))
                )
                for e in state.benchmark_entries
            ]

            BenchmarkUseCase = _get_benchmark_use_case()
            use_case = BenchmarkUseCase(state.model)
            metrics = await run.io_bound(use_case.run_benchmark, bc, entries)

            state.benchmark_results = metrics
            elapsed = (datetime.now() - t0).total_seconds()
            show_results(elapsed)
            ui.notify('Benchmark completado', type='positive')

        except Exception as ex:
            ui.notify(f'Error: {str(ex)[:80]}', type='negative')
        finally:
            state.benchmark_running = False
            run_btn.enable()
            loading_row.set_visibility(False)

    def show_results(elapsed: float):
        results_box.clear()
        m = state.benchmark_results
        if not m:
            return

        with results_box:
            ui.label('Resultados').classes('text-xl font-semibold mb-4')

            # Metrics
            with ui.row().classes('gap-3 mb-4 flex-wrap'):
                for label, value, color in [
                    ('Accuracy', m.accuracy, 'text-indigo-400'),
                    ('Precision', m.precision, 'text-green-400'),
                    ('Recall', m.recall, 'text-amber-400'),
                    ('F1', m.f1_score, 'text-rose-400'),
                    ('Tiempo', elapsed, 'text-slate-400'),
                ]:
                    with ui.column().classes('metric-box'):
                        if isinstance(value, float) and label != 'Tiempo':
                            ui.label(f'{value:.1%}').classes(f'metric-value {color}')
                        else:
                            ui.label(f'{value:.1f}s' if label == 'Tiempo' else str(value)).classes(f'metric-value {color}')
                        ui.label(label).classes('metric-label')

            # Confusion Matrix
            with ui.column().classes('card mb-4 gap-3'):
                ui.label('Matriz de Confusión').classes('font-semibold')
                cm = m.confusion_matrix
                with ui.element('div').classes('grid grid-cols-3 gap-2 w-56'):
                    ui.element('div')
                    ui.label('Pred +').classes('text-xs text-slate-400 text-center')
                    ui.label('Pred -').classes('text-xs text-slate-400 text-center')

                    ui.label('Real +').classes('text-xs text-slate-400 self-center')
                    with ui.element('div').classes('cm-cell cm-good'):
                        ui.label(str(cm.get('true_positive', 0))).classes('text-lg font-bold text-green-400')
                    with ui.element('div').classes('cm-cell cm-bad'):
                        ui.label(str(cm.get('false_negative', 0))).classes('text-lg font-bold text-red-400')

                    ui.label('Real -').classes('text-xs text-slate-400 self-center')
                    with ui.element('div').classes('cm-cell cm-bad'):
                        ui.label(str(cm.get('false_positive', 0))).classes('text-lg font-bold text-red-400')
                    with ui.element('div').classes('cm-cell cm-good'):
                        ui.label(str(cm.get('true_negative', 0))).classes('text-lg font-bold text-green-400')

            # Misclassified
            if m.misclassified:
                with ui.expansion(f'{len(m.misclassified)} mal clasificados', icon='warning').classes('w-full'):
                    for case in m.misclassified[:10]:
                        with ui.row().classes('items-center gap-2 p-2 bg-slate-800/50 rounded mb-1'):
                            ui.label(f'Pred: {case.predicted_label}').classes('status-pill pill-error')
                            ui.icon('arrow_forward', size='xs').classes('text-slate-600')
                            ui.label(f'Real: {case.actual_label}').classes('status-pill pill-success')

            # Export
            def do_export():
                export = {
                    'timestamp': datetime.now().isoformat(),
                    'model': state.model,
                    'metrics': m.to_dict(),
                    'time': elapsed
                }
                ui.download(
                    json.dumps(export, indent=2, ensure_ascii=False).encode('utf-8'),
                    f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )

            ui.button('Exportar JSON', icon='download', on_click=do_export).props('flat').classes('mt-2')

    # === LAYOUT ===
    with ui.column().classes('w-full max-w-5xl mx-auto gap-5 p-6'):
        ui.label('Benchmark').classes('text-2xl font-bold')

        # Cards (púrpura - paso 2 Configura)
        with ui.row().classes('w-full gap-4'):
            with ui.column().classes('card flex-1 gap-2 border border-purple-500/20'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('settings', size='sm').classes('text-purple-400')
                    ui.label('Configuración').classes('font-semibold text-purple-300')
                config_content = ui.column().classes('w-full min-h-12')
                ui.button('Cargar ejemplo', icon='folder', on_click=load_config).props('flat dense size=sm').classes('mt-2')

            with ui.column().classes('card flex-1 gap-2 border border-purple-500/20'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('dataset', size='sm').classes('text-purple-400')
                    ui.label('Entradas').classes('font-semibold text-purple-300')
                entries_content = ui.column().classes('w-full min-h-12')
                ui.button('Cargar ejemplo', icon='folder', on_click=load_entries).props('flat dense size=sm').classes('mt-2')

        # Run (emerald oscuro - paso 3)
        with ui.row().classes('items-center gap-4'):
            run_btn = ui.button('Ejecutar Benchmark', icon='analytics', on_click=run_benchmark).props('unelevated')
            run_btn.style('background: linear-gradient(135deg, #0f4342, #0d3a39) !important; color: #6ee7b7 !important; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.5);')
            run_btn.disable()

            with ui.row().classes('items-center gap-2') as lr:
                loading_row = lr
                loading_row.set_visibility(False)
                ui.spinner('dots', size='md').classes('text-emerald-400')
                ui.label('Evaluando...').classes('text-slate-400')

        # Results
        results_box = ui.column().classes('w-full')

        # Init
        update_ui()


# =============================================================================
# MAIN
# =============================================================================
@ui.page('/')
def main():
    """Aplicación principal."""
    ui.dark_mode().enable()
    ui.add_head_html(CUSTOM_CSS)

    # Contenedor para tabs (permite referencia antes de definir)
    tabs_ref = {'tabs': None}

    # Header con logo y tabs en la misma barra
    with ui.header().classes('bg-slate-900/95 backdrop-blur border-b border-slate-700 px-6 py-0'):
        with ui.row().classes('w-full max-w-5xl mx-auto items-center justify-between'):
            # Logo (izquierda) - clickable para ir a Inicio
            with ui.row().classes('items-center gap-3 cursor-pointer').on('click', lambda: tabs_ref['tabs'].set_value('Inicio') if tabs_ref['tabs'] else None):
                with ui.element('div').classes('w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md shadow-indigo-500/30'):
                    ui.icon('psychology', size='sm').classes('text-white')
                ui.label('EsencIA').classes('text-xl font-bold')

            # Tabs (derecha)
            with ui.tabs().classes('') as tabs:
                ui.tab('Inicio', icon='home')
                ui.tab('Pipeline', icon='account_tree')
                ui.tab('Benchmark', icon='analytics')

    # Guardar referencia
    tabs_ref['tabs'] = tabs

    # Estado de pestañas renderizadas (lazy loading)
    rendered_tabs = {'Inicio': False, 'Pipeline': False, 'Benchmark': False}
    current_tab = {'value': 'Inicio'}

    # Content con lazy loading - solo renderiza la pestaña activa
    with ui.column().classes('w-full') as content_area:
        with ui.tab_panels(tabs, value='Inicio').classes('w-full') as panels:
            # Inicio - se renderiza siempre al inicio
            with ui.tab_panel('Inicio') as inicio_panel:
                home_page(tabs)
                rendered_tabs['Inicio'] = True

            # Pipeline - lazy load
            with ui.tab_panel('Pipeline') as pipeline_panel:
                pipeline_placeholder = ui.column().classes('w-full')

            # Benchmark - lazy load
            with ui.tab_panel('Benchmark') as benchmark_panel:
                benchmark_placeholder = ui.column().classes('w-full')

    async def render_tab_content(tab_name: str):
        """Renderiza el contenido de una pestaña si no está renderizada."""
        if tab_name == 'Pipeline' and not rendered_tabs['Pipeline']:
            # Marcar como renderizado INMEDIATAMENTE para evitar doble renderizado
            rendered_tabs['Pipeline'] = True

            # Mostrar loading inmediatamente
            pipeline_placeholder.clear()
            with pipeline_placeholder:
                with ui.column().classes('w-full items-center justify-center py-20'):
                    ui.spinner('dots', size='lg').classes('text-indigo-400')
                    ui.label('Cargando Pipeline...').classes('text-slate-400 mt-4')

            # Ceder control brevemente para que se muestre el loading
            await asyncio.sleep(0)

            # Renderizar el contenido real
            pipeline_placeholder.clear()
            with pipeline_placeholder:
                pipeline_page()

        elif tab_name == 'Benchmark' and not rendered_tabs['Benchmark']:
            # Marcar como renderizado INMEDIATAMENTE
            rendered_tabs['Benchmark'] = True

            # Mostrar loading inmediatamente
            benchmark_placeholder.clear()
            with benchmark_placeholder:
                with ui.column().classes('w-full items-center justify-center py-20'):
                    ui.spinner('dots', size='lg').classes('text-indigo-400')
                    ui.label('Cargando Benchmark...').classes('text-slate-400 mt-4')

            # Ceder control brevemente
            await asyncio.sleep(0)

            # Renderizar el contenido real
            benchmark_placeholder.clear()
            with benchmark_placeholder:
                benchmark_page()

    def on_tab_change(e):
        """Handler para cambio de pestaña - dispara renderizado async."""
        tab_name = e.args if isinstance(e.args, str) else e.args
        current_tab['value'] = tab_name
        asyncio.create_task(render_tab_content(tab_name))

    # Escuchar cambios de pestaña de múltiples formas para mayor confiabilidad
    tabs.on('update:model-value', on_tab_change)
    tabs.on_value_change(lambda e: asyncio.create_task(render_tab_content(e.value)))

    # Nota: La precarga de modelos HuggingFace se hace lazy (al entrar en Pipeline)


# =============================================================================
# STARTUP TASKS - Pre-conexión para reducir latencia
# =============================================================================
async def on_startup():
    """Tareas de inicialización al arrancar la app."""
    # Ejecutar en paralelo para máxima eficiencia
    await asyncio.gather(
        init_system_resources_async(),  # Detectar GPU/VRAM en background
        warmup_hf_connection(),          # Pre-establecer conexión a HuggingFace
        return_exceptions=True
    )


app.on_startup(on_startup)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ in {'__main__', '__mp_main__'}:
    ui.run(title='EsencIA', dark=True, port=8080, reload=True)
