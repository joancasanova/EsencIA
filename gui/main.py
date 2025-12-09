"""
EsencIA - GUI Profesional
Interfaz para procesamiento de textos con LLMs.
Arquitectura: NiceGUI + Backend asíncrono
"""

import sys
import json
import platform
import asyncio
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
                model_id_lower = model_id.lower()
                skip_patterns = [
                    'embed', 'embedding', 'sentence-transformer', 'bge-', 'e5-',
                    'gte-', 'instructor', 'encoder', 'retriev', 'rerank',
                    'clip', 'bert-base', 'bert-large', 'roberta', 'xlm-roberta',
                    'minilm', 'mpnet', 'contriever', 'colbert', 'splade'
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
            },
            {
                'type': 'parse',
                'parameters': {
                    'rules': [
                        {'name': 'adaptacion', 'mode': 'REGEX', 'pattern': r'.*', 'fallback_value': ''}
                    ],
                    'output_filter': 'all'
                },
                'uses_reference': True,
                'reference_step_numbers': [0]
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
                    'temperature': 0.2
                },
                'uses_reference': True
            },
            {
                'type': 'parse',
                'parameters': {
                    'rules': [
                        {'name': 'dato', 'mode': 'KEYWORD', 'pattern': 'Dato:', 'secondary_pattern': '\n', 'fallback_value': 'no_encontrado'}
                    ],
                    'output_filter': 'successful'
                },
                'uses_reference': True,
                'reference_step_numbers': [0]
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
                    'temperature': 0.7
                },
                'uses_reference': True
            },
            {
                'type': 'parse',
                'parameters': {
                    'rules': [
                        {'name': 'contenido', 'mode': 'KEYWORD', 'pattern': 'Contenido:', 'secondary_pattern': '"', 'fallback_value': ''}
                    ],
                    'output_filter': 'successful'
                },
                'uses_reference': True,
                'reference_step_numbers': [0]
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
                'reference_step_numbers': [1]
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
                    'rules': [],
                    'output_filter': 'all'
                },
                'uses_reference': True,
                'reference_step_numbers': [len(local_state['steps']) - 1] if local_state['steps'] else []
            }
        elif step_type == 'verify':
            new_step = {
                'type': 'verify',
                'parameters': {
                    'methods': [],
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
            rules = local_state['steps'][step_idx]['parameters'].get('rules', [])
            rules.append({
                'name': f'campo_{len(rules)+1}',
                'mode': 'KEYWORD',
                'pattern': '',
                'secondary_pattern': '',
                'fallback_value': ''
            })
            local_state['steps'][step_idx]['parameters']['rules'] = rules
            refresh_builder()

    def add_verify_method(step_idx: int):
        """Añade un método de verificación."""
        if 0 <= step_idx < len(local_state['steps']):
            methods = local_state['steps'][step_idx]['parameters'].get('methods', [])
            methods.append({
                'mode': 'cumulative',
                'name': f'verificar_{len(methods)+1}',
                'system_prompt': 'Responde Yes o No.',
                'user_prompt': '',
                'num_sequences': 3,
                'valid_responses': ['Yes', 'yes', 'Si', 'si'],
                'required_matches': 2,
                'max_tokens': 5,
                'temperature': 0.8
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

    def refresh_data_section():
        """Refresca la sección de datos."""
        data_container.clear()
        with data_container:
            render_data_section()
        update_counters()

    def render_steps_accordion():
        """Renderiza los pasos como barras desplegables (accordion)."""
        step_colors = {
            'generate': ('purple', 'auto_awesome', 'Generar'),
            'parse': ('purple', 'find_in_page', 'Detectar'),
            'verify': ('purple', 'verified', 'Verificar')
        }

        # Botones para añadir pasos con etiqueta clara
        with ui.row().classes('w-full items-center gap-3'):
            ui.label('Añadir:').classes('text-xs text-slate-400 font-medium')

            with ui.button(on_click=lambda: add_step('generate')).props('flat dense no-caps').classes(
                'h-8 pl-2 pr-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                'hover:bg-purple-500/20 hover:border-purple-500/50 transition-all flex items-center'
            ):
                ui.icon('add', size='xs').classes('text-purple-400 mr-1.5')
                ui.label('Generar').classes('text-xs text-slate-300 leading-none')

            with ui.button(on_click=lambda: add_step('parse')).props('flat dense no-caps').classes(
                'h-8 pl-2 pr-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                'hover:bg-purple-500/20 hover:border-purple-500/50 transition-all flex items-center'
            ):
                ui.icon('add', size='xs').classes('text-purple-400 mr-1.5')
                ui.label('Detectar').classes('text-xs text-slate-300 leading-none')

            with ui.button(on_click=lambda: add_step('verify')).props('flat dense no-caps').classes(
                'h-8 pl-2 pr-3 bg-slate-700/50 border border-dashed border-slate-500/50 rounded-lg '
                'hover:bg-purple-500/20 hover:border-purple-500/50 transition-all flex items-center'
            ):
                ui.icon('add', size='xs').classes('text-purple-400 mr-1.5')
                ui.label('Verificar').classes('text-xs text-slate-300 leading-none')

        if not local_state['steps']:
            with ui.column().classes('w-full items-center justify-center py-8 gap-2'):
                ui.icon('touch_app', size='lg').classes('text-slate-600')
                ui.label('Sin pasos configurados').classes('text-slate-400 text-sm')
                ui.label('Añade pasos con los botones de arriba').classes('text-xs text-slate-600')
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
                    summary = f"{rules_count} regla{'s' if rules_count != 1 else ''} de extracción"
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
                        'cursor-pointer hover:bg-purple-500/10 hover:border-purple-500/50 transition-all overflow-hidden'
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
            'parse': ('purple', 'find_in_page', 'Detectar'),
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
                return [f'output_{step_idx + 1}']
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
                ui.label('Insertar variable:').classes('text-xs text-slate-400 mr-1')

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
                        field_name = var['name'].split('_')[0] if '_' in var['name'] else var['name']
                        if field_name in ('content', 'output'):
                            display_name = f'Paso {step_num}'
                        else:
                            display_name = f'Paso {step_num}: {field_name}'
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
                        ui.label(display_name).classes('font-mono text-xs text-slate-400 normal-case')
                        ui.tooltip(tooltip_text).classes('bg-slate-800 text-slate-200')

        with ui.column().classes('w-full gap-3'):
            # === SECCIÓN: PROMPTS ===
            with ui.column().classes('w-full gap-2'):
                # System Prompt
                with ui.column().classes('w-full gap-0'):
                    with ui.row().classes('items-center gap-1.5'):
                        ui.label('System Prompt').classes('text-xs font-medium text-slate-300 uppercase')
                        with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                            ui.tooltip('Define el rol y comportamiento del modelo').classes('bg-slate-800 text-slate-200')

                    # Contenedor para textarea + chips (textarea primero para obtener su referencia)
                    has_vars = bool(get_available_variables())
                    with ui.column().classes('w-full gap-0 mt-1'):
                        system_textarea = ui.textarea(
                            value=params.get('system_prompt', ''),
                            placeholder='Eres un asistente capaz de...',
                            on_change=lambda e, i=idx: update_step_param(i, 'system_prompt', e.value)
                        ).props('outlined dark dense rows=1 autogrow').classes('w-full input-subtle' + (' rounded-b-none' if has_vars else ''))
                        render_variable_chips(system_textarea, 'system_prompt')

                # User Prompt
                with ui.column().classes('w-full gap-0 mt-2'):
                    with ui.row().classes('items-center gap-1.5'):
                        ui.label('User Prompt').classes('text-xs font-medium text-slate-300 uppercase')
                        with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                            ui.tooltip('Instrucciones específicas para cada entrada').classes('bg-slate-800 text-slate-200')

                    # Contenedor para textarea + chips
                    has_vars = bool(get_available_variables())
                    with ui.column().classes('w-full gap-0 mt-1'):
                        user_textarea = ui.textarea(
                            value=params.get('user_prompt', ''),
                            placeholder='Analiza el siguiente texto: {texto}',
                            on_change=lambda e, i=idx: update_step_param(i, 'user_prompt', e.value)
                        ).props('outlined dark dense rows=1 autogrow').classes('w-full input-subtle' + (' rounded-b-none' if has_vars else ''))
                        render_variable_chips(user_textarea, 'user_prompt')

            # === SECCIÓN: REFERENCIAS + PARÁMETROS (en fila) ===
            ui.element('div').classes('w-full h-px bg-slate-700/50 my-1')

            with ui.row().classes('w-full gap-6 items-start'):
                # --- COLUMNA IZQUIERDA: REFERENCIAS ---
                with ui.column().classes('flex-1 gap-2'):
                    # === REFERENCIAS A PASOS ANTERIORES ===
                    with ui.row().classes('items-center gap-1.5'):
                        ui.icon('data_object').classes('text-slate-400 text-sm')
                        ui.label('Referenciar outputs de pasos anteriores').classes('text-xs font-medium text-slate-300 uppercase')
                        with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                            ui.tooltip('Selecciona pasos anteriores para usar sus outputs como variables en tus prompts.\nPodrás insertar su valor en el prompt usando llaves como {output} ó pulsando el botón "Insertar variable".').classes('bg-slate-800 text-slate-200 whitespace-pre-line')

                    if idx > 0:
                        current_refs = step.get('reference_step_numbers', [])
                        with ui.column().classes('w-full gap-0'):
                            for prev_idx in range(idx):
                                prev_step = local_state['steps'][prev_idx]
                                prev_type = prev_step.get('type', 'generate')
                                type_icons = {'generate': 'auto_awesome', 'parse': 'find_in_page', 'verify': 'verified'}
                                type_names = {'generate': 'Generar', 'parse': 'Detectar', 'verify': 'Verificar'}
                                is_checked = prev_idx in current_refs

                                # Obtener variables que produce este paso
                                step_vars = get_step_output_variables(prev_step, prev_idx)
                                vars_display = ', '.join(f'{{{v}}}' for v in step_vars) if step_vars else '(sin variables establecidas)'

                                with ui.row().classes('w-full items-center gap-1 py-2'):
                                    cb = ui.checkbox(
                                        value=is_checked,
                                        on_change=lambda e, i=idx, r=prev_idx: toggle_reference_step(i, r, e.value)
                                    ).props('dense color=purple')
                                    ui.label(f'Paso {prev_idx + 1}:').classes('text-base text-slate-300 cursor-pointer select-none').style('margin-top: -1px').on('click', lambda _, c=cb: c.set_value(not c.value))
                                    ui.label(type_names.get(prev_type, prev_type)).classes('text-base text-slate-300 cursor-pointer select-none').style('margin-top: -1px').on('click', lambda _, c=cb: c.set_value(not c.value))
                                    # Separador visual
                                    ui.label('→').classes('text-slate-500 mx-1')
                                    # Variables disponibles
                                    vars_classes = 'text-sm text-slate-400 font-mono' if step_vars else 'text-sm text-slate-400 italic'
                                    ui.label(vars_display).classes(vars_classes)
                    else:
                        ui.label('(primer paso, sin referencias disponibles)').classes('text-xs text-slate-400 italic mt-1')

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
                                with ui.row().classes('items-center gap-1'):
                                    ui.label('Temperatura').classes('text-xs text-slate-400')
                                    with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                                        ui.tooltip('Controla la creatividad del modelo.\n0 = Respuestas más predecibles\n2 = Respuestas más variadas').classes('bg-slate-800 text-slate-200 whitespace-pre-line')
                                temp_label = ui.label(f'{params.get("temperature", 0.7):.1f}').classes('text-xs text-slate-300 font-mono')

                            def on_temp_change(e, i=idx):
                                temp_label.text = f'{e.value:.1f}'
                                update_step_param(i, 'temperature', float(e.value))

                            ui.slider(
                                value=params.get('temperature', 0.7),
                                min=0, max=2, step=0.1,
                                on_change=on_temp_change
                            ).props('color=purple').classes('w-full')
                            with ui.row().classes('w-full justify-between -mt-1'):
                                ui.label('Preciso').classes('text-[10px] text-slate-500')
                                ui.label('Creativo').classes('text-[10px] text-slate-500')

                        # Max Tokens
                        with ui.column().classes('gap-1 w-21 ml-4'):
                            with ui.row().classes('items-center gap-1'):
                                ui.label('Max Tokens').classes('text-xs text-slate-400')
                                with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                                    ui.tooltip('Longitud máxima de la respuesta generada (1-4096)').classes('bg-slate-800 text-slate-200')
                            ui.number(
                                value=params.get('max_tokens', 200),
                                min=1, max=4096,
                                on_change=lambda e, i=idx: update_step_param(i, 'max_tokens', int(e.value) if e.value else 200)
                            ).props('outlined dark dense color=purple input-class="text-center !py-1"').classes('w-full')

                        # Secuencias
                        with ui.column().classes('gap-1 w-20'):
                            with ui.row().classes('items-center gap-1'):
                                ui.label('Secuencias').classes('text-xs text-slate-400')
                                with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                                    ui.tooltip('Número de variaciones a generar por cada entrada (1-10)').classes('bg-slate-800 text-slate-200')
                            ui.number(
                                value=params.get('num_sequences', 1),
                                min=1, max=10,
                                on_change=lambda e, i=idx: update_step_param(i, 'num_sequences', int(e.value) if e.value else 1)
                            ).props('outlined dark dense color=purple input-class="text-center !py-1"').classes('w-full')

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
                    ui.label('Usar modelo específico para este paso').classes('text-xs font-medium text-slate-300 uppercase')
                    with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                        ui.tooltip('Puedes usar un modelo específico diferente al LLM seleccionado general para este paso').classes('bg-slate-800 text-slate-200')

                if has_custom_model:
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
                                        with ui.row().classes('items-center gap-1'):
                                            ui.icon('folder_open', size='xs').classes('text-purple-400')
                                            ui.label('Local').classes('text-xs text-slate-300')
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

        with ui.column().classes('w-full gap-4'):
            # Reglas de parsing
            ui.label('Reglas de extracción').classes('text-sm text-slate-400')
            ui.label('Define cómo extraer datos del texto generado').classes('text-xs text-slate-400 -mt-2')

            if not rules:
                with ui.row().classes('items-center gap-2 p-3 bg-slate-900/30 rounded'):
                    ui.icon('info', size='xs').classes('text-slate-400')
                    ui.label('No hay reglas. Añade una regla para extraer datos.').classes('text-slate-400 text-sm')

            for r_idx, rule in enumerate(rules):
                with ui.card().classes('w-full bg-slate-900/30 p-3'):
                    # Primera fila: Nombre, Modo, Eliminar
                    with ui.row().classes('w-full gap-2 items-center justify-between'):
                        with ui.row().classes('gap-2 items-end'):
                            ui.input(
                                label='Nombre variable',
                                value=rule.get('name', ''),
                                placeholder='ej: contenido',
                                on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'name', e.value)
                            ).props('outlined dark dense').classes('w-36').tooltip('Nombre de la variable extraída')

                            ui.select(
                                label='Modo',
                                options=['KEYWORD', 'REGEX'],
                                value=rule.get('mode', 'KEYWORD'),
                                on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'mode', e.value)
                            ).props('outlined dark dense').classes('w-28').tooltip('KEYWORD: busca texto literal\nREGEX: usa expresión regular')

                        ui.button(icon='delete', on_click=lambda si=idx, ri=r_idx: remove_parse_rule(si, ri)).props('flat round size=sm color=red').tooltip('Eliminar regla')

                    # Segunda fila: Patrón principal
                    ui.input(
                        label='Patrón principal',
                        value=rule.get('pattern', ''),
                        placeholder='ej: Contenido: o regex: .*',
                        on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'pattern', e.value)
                    ).props('outlined dark dense').classes('w-full mt-2').tooltip('Texto o regex para encontrar el inicio del dato')

                    # Tercera fila: Patrón secundario y Valor por defecto
                    with ui.row().classes('gap-2 mt-2'):
                        ui.input(
                            label='Patrón fin (secundario)',
                            value=rule.get('secondary_pattern', ''),
                            placeholder='ej: \\n o "',
                            on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'secondary_pattern', e.value)
                        ).props('outlined dark dense').classes('flex-1').tooltip('Texto que marca el final del dato a extraer')

                        ui.input(
                            label='Valor por defecto',
                            value=rule.get('fallback_value', ''),
                            placeholder='ej: no_encontrado',
                            on_change=lambda e, si=idx, ri=r_idx: update_parse_rule(si, ri, 'fallback_value', e.value)
                        ).props('outlined dark dense').classes('flex-1').tooltip('Valor si no se encuentra el patrón')

            ui.button('+ Añadir regla', icon='add', on_click=lambda i=idx: add_parse_rule(i)).props('flat size=sm').classes('text-amber-400')

            # Filtro de salida
            ui.label('Configuración de salida').classes('text-sm text-slate-400 mt-2')
            with ui.row().classes('gap-4'):
                ui.select(
                    label='Filtro de resultados',
                    options=['all', 'successful', 'failed'],
                    value=params.get('output_filter', 'all'),
                    on_change=lambda e, i=idx: update_step_param(i, 'output_filter', e.value)
                ).props('outlined dark dense').classes('w-40').tooltip('all: todos los resultados\nsuccessful: solo exitosos\nfailed: solo fallidos')

    def render_verify_config(idx: int, step: Dict):
        """Renderiza configuración de paso verify."""
        params = step.get('parameters', {})
        methods = params.get('methods', [])

        with ui.column().classes('w-full gap-4'):
            ui.label('Métodos de verificación').classes('text-sm text-slate-400')
            ui.label('Define criterios para validar los datos extraídos').classes('text-xs text-slate-400 -mt-2')

            if not methods:
                with ui.row().classes('items-center gap-2 p-3 bg-slate-900/30 rounded'):
                    ui.icon('info', size='xs').classes('text-slate-400')
                    ui.label('No hay métodos. Añade uno para verificar resultados.').classes('text-slate-400 text-sm')

            for m_idx, method in enumerate(methods):
                with ui.card().classes('w-full bg-slate-900/30 p-3'):
                    # Primera fila: Nombre, Modo, Eliminar
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('gap-2 items-end'):
                            ui.input(
                                label='Nombre',
                                value=method.get('name', ''),
                                placeholder='ej: validar_contenido',
                                on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'name', e.value)
                            ).props('outlined dark dense').classes('w-40').tooltip('Identificador del método')

                            ui.select(
                                label='Modo',
                                options=['cumulative', 'eliminatory'],
                                value=method.get('mode', 'cumulative'),
                                on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'mode', e.value)
                            ).props('outlined dark dense').classes('w-32').tooltip('cumulative: suma votos positivos\neliminatory: un No elimina')

                        ui.button(icon='delete', on_click=lambda si=idx, mi=m_idx: remove_verify_method(si, mi)).props('flat round size=sm color=red').tooltip('Eliminar método')

                    # System Prompt
                    ui.label('System Prompt').classes('text-xs text-slate-400 mt-2')
                    ui.textarea(
                        value=method.get('system_prompt', 'Responde Yes o No.'),
                        placeholder='Instrucciones para el verificador...',
                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'system_prompt', e.value)
                    ).props('outlined dark dense rows=2 autogrow').classes('w-full input-subtle').tooltip('Instrucciones del sistema para la verificación')

                    # User Prompt
                    ui.label('User Prompt').classes('text-xs text-slate-400 mt-2')
                    ui.textarea(
                        value=method.get('user_prompt', ''),
                        placeholder='Pregunta de verificación. Usa {variable} para datos.',
                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'user_prompt', e.value)
                    ).props('outlined dark dense rows=2 autogrow').classes('w-full input-subtle').tooltip('Pregunta para verificar el contenido')

                    # Valid Responses
                    ui.label('Respuestas válidas').classes('text-xs text-slate-400 mt-2')
                    valid_responses = method.get('valid_responses', ['Yes', 'yes', 'Si', 'si'])
                    ui.input(
                        value=', '.join(valid_responses) if isinstance(valid_responses, list) else str(valid_responses),
                        placeholder='Yes, yes, Si, si',
                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'valid_responses', [r.strip() for r in e.value.split(',')] if e.value else [])
                    ).props('outlined dark dense').classes('w-full').tooltip('Respuestas que cuentan como verificación positiva (separadas por coma)')

                    # Parámetros numéricos
                    with ui.row().classes('gap-3 mt-3 flex-wrap'):
                        ui.number(
                            label='Secuencias',
                            value=method.get('num_sequences', 3),
                            min=1, max=10,
                            on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'num_sequences', int(e.value) if e.value else 3)
                        ).props('outlined dark dense').classes('w-24').tooltip('Número de verificaciones a realizar')

                        ui.number(
                            label='Requeridos',
                            value=method.get('required_matches', 2),
                            min=1, max=10,
                            on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'required_matches', int(e.value) if e.value else 2)
                        ).props('outlined dark dense').classes('w-24').tooltip('Mínimo de respuestas positivas necesarias')

                        ui.number(
                            label='Max Tokens',
                            value=method.get('max_tokens', 5),
                            min=1, max=50,
                            on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'max_tokens', int(e.value) if e.value else 5)
                        ).props('outlined dark dense').classes('w-24').tooltip('Tokens máximos en respuesta')

                        ui.number(
                            label='Temperatura',
                            value=method.get('temperature', 0.8),
                            min=0, max=2, step=0.1,
                            on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'temperature', float(e.value) if e.value else 0.8)
                        ).props('outlined dark dense').classes('w-24').tooltip('Variabilidad en respuestas')

            ui.button('+ Añadir método', icon='add', on_click=lambda i=idx: add_verify_method(i)).props('flat size=sm').classes('text-emerald-400')

            # Configuración global de verificación
            ui.label('Configuración global').classes('text-sm text-slate-400 mt-2')
            with ui.row().classes('gap-4'):
                ui.number(
                    label='Mín. métodos para confirmar',
                    value=params.get('required_for_confirmed', 1),
                    min=0, max=10,
                    on_change=lambda e, i=idx: update_step_param(i, 'required_for_confirmed', int(e.value) if e.value else 1)
                ).props('outlined dark dense').classes('w-48').tooltip('Métodos que deben pasar para confirmar')

                ui.number(
                    label='Mín. métodos para revisión',
                    value=params.get('required_for_review', 0),
                    min=0, max=10,
                    on_change=lambda e, i=idx: update_step_param(i, 'required_for_review', int(e.value) if e.value else 0)
                ).props('outlined dark dense').classes('w-48').tooltip('Métodos que deben pasar para marcar como revisar')

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

    def update_verify_method(step_idx: int, method_idx: int, field: str, value):
        """Actualiza un campo de un método de verificación."""
        if 0 <= step_idx < len(local_state['steps']):
            methods = local_state['steps'][step_idx]['parameters'].get('methods', [])
            if 0 <= method_idx < len(methods):
                methods[method_idx][field] = value

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
                with ui.row().classes('items-center gap-1'):
                    ui.label('Utilizar Variables de Entrada').classes('text-sm font-medium text-slate-200')
                    with ui.icon('help_outline').classes('text-slate-400 cursor-help text-[12px]'):
                        ui.tooltip(
                            'Define variables con múltiples valores. El pipeline se ejecutará una vez por cada fila.'
                        ).classes('text-xs')

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
                                value='' if is_placeholder else field,
                                placeholder='nombre_var' if is_placeholder else 'nombre'
                            ).props('outlined dense dark').classes(
                                'text-sm input-subtle shrink-0'
                            ).style('min-height: 18px; width: 105px').on(
                                'blur', lambda e, f=field: validate_and_update_field_name(f, e.sender.value, e.sender)
                            ).on(
                                'keydown.enter', lambda e: e.sender.run_method('blur')
                            )
                            ui.label('→').classes('text-slate-500 text-xs shrink-0')
                            ui.label('{sin nombre}' if is_placeholder else f'{{{field}}}').classes('text-slate-400 font-mono text-xs shrink-0 whitespace-nowrap')
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
                        ui.label('Variable').classes('text-xs text-slate-300 leading-none')

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
                            ui.label('Valor').classes('text-xs text-slate-300 leading-none')

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
                        temperature=m.get('temperature', 0.8)
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

    async def run_pipeline():
        """Ejecuta el pipeline."""
        if not local_state['steps']:
            ui.notify('Configura al menos un paso', type='warning')
            return

        # Determinar modo de ejecución:
        # - Con datos de entrada: ejecutar N veces (una por entrada)
        # - Sin datos de entrada: ejecutar una sola vez
        input_values = local_state['input_values']
        fields = local_state['input_fields']
        has_data_entries = bool(input_values) and bool(fields) and any(len(v) > 0 for v in input_values.values())

        run_btn.disable()
        loading_box.set_visibility(True)
        results_container.clear()

        t0 = datetime.now()

        try:
            steps = build_steps_from_config()
            if not steps:
                raise ValueError("No se pudieron construir los pasos")

            request = PipelineRequest(steps=steps, global_references={})

            PipelineUseCase = _get_pipeline_use_case()
            use_case = PipelineUseCase(state.model)

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

                # Modo múltiple: ejecutar para cada entrada de datos
                response = await run.io_bound(
                    use_case.execute_with_references,
                    request,
                    data_entries
                )
                elapsed = (datetime.now() - t0).total_seconds()
                show_results(response, elapsed)
                ui.notify(f'Completado: {response.successful_entries}/{response.total_entries}', type='positive')
            else:
                # Modo único: ejecutar una sola vez sin datos de entrada
                response = await run.io_bound(
                    use_case._execute,
                    request
                )
                elapsed = (datetime.now() - t0).total_seconds()
                show_single_result(response, elapsed)
                ui.notify('Pipeline ejecutado correctamente', type='positive')

        except Exception as ex:
            ui.notify(f'Error: {str(ex)[:100]}', type='negative')
            with results_container:
                with ui.row().classes('items-center gap-2 p-4 bg-red-900/20 rounded-lg'):
                    ui.icon('error', size='sm').classes('text-red-400')
                    ui.label(f'Error: {str(ex)}').classes('text-red-300 text-sm')
        finally:
            run_btn.enable()
            loading_box.set_visibility(False)

    def show_single_result(response, elapsed: float):
        """Muestra el resultado de una ejecución única del pipeline."""
        results_container.clear()
        with results_container:
            ui.label('Resultado').classes('text-xl font-semibold mb-4')

            # Métricas simples
            with ui.row().classes('gap-3 mb-4 flex-wrap'):
                metrics = [
                    ('Pasos', str(len(response.step_results)), 'text-slate-200'),
                    ('Tiempo', f'{elapsed:.1f}s', 'text-amber-400'),
                ]
                for label, value, color in metrics:
                    with ui.column().classes('metric-box'):
                        ui.label(value).classes(f'metric-value {color}')
                        ui.label(label).classes('metric-label')

            # Confirmados vs a revisar
            confirmed = response.verification_references.get('confirmed', [])
            to_verify = response.verification_references.get('to_verify', [])

            if confirmed or to_verify:
                with ui.row().classes('gap-4 mb-4'):
                    with ui.column().classes('flex-1 p-3 bg-emerald-900/20 rounded-lg border border-emerald-500/30'):
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.icon('check_circle', size='sm').classes('text-emerald-400')
                            ui.label(f'Confirmados ({len(confirmed)})').classes('text-emerald-300 font-medium')
                        for ref in confirmed[:5]:
                            ui.label(str(ref)[:100] + ('...' if len(str(ref)) > 100 else '')).classes('text-xs text-slate-400')

                    with ui.column().classes('flex-1 p-3 bg-amber-900/20 rounded-lg border border-amber-500/30'):
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.icon('pending', size='sm').classes('text-amber-400')
                            ui.label(f'A revisar ({len(to_verify)})').classes('text-amber-300 font-medium')
                        for ref in to_verify[:5]:
                            ui.label(str(ref)[:100] + ('...' if len(str(ref)) > 100 else '')).classes('text-xs text-slate-400')

            # Mostrar resultados de cada paso
            for step_idx, step_result in enumerate(response.step_results):
                step_type = step_result.get('type', 'generate')
                step_icons = {'generate': 'auto_awesome', 'parse': 'find_in_page', 'verify': 'verified'}

                with ui.expansion(f'Paso {step_idx + 1}: {step_type.capitalize()}', icon=step_icons.get(step_type, 'check')).classes('w-full bg-slate-800/30'):
                    with ui.element('div').classes('p-3 bg-slate-900/50 rounded'):
                        ui.label(json.dumps(step_result, indent=2, ensure_ascii=False, default=str)).classes('whitespace-pre font-mono text-xs text-slate-300')

    def show_results(response, elapsed: float):
        """Muestra los resultados del pipeline."""
        results_container.clear()
        with results_container:
            ui.label('Resultados').classes('text-xl font-semibold mb-4')

            # Métricas
            with ui.row().classes('gap-3 mb-4 flex-wrap'):
                metrics = [
                    ('Total', str(response.total_entries), 'text-slate-200'),
                    ('Exitosos', str(response.successful_entries), 'text-green-400'),
                    ('Fallidos', str(response.failed_entries), 'text-red-400'),
                ]
                if response.total_entries > 0:
                    pct = response.successful_entries / response.total_entries
                    metrics.append(('Tasa', f'{pct:.0%}', 'text-indigo-400'))
                metrics.append(('Tiempo', f'{elapsed:.1f}s', 'text-amber-400'))

                for label, value, color in metrics:
                    with ui.column().classes('metric-box'):
                        ui.label(value).classes(f'metric-value {color}')
                        ui.label(label).classes('metric-label')

            # Confirmados vs a revisar
            confirmed = response.verification_references.get('confirmed', [])
            to_verify = response.verification_references.get('to_verify', [])

            if confirmed or to_verify:
                with ui.row().classes('gap-4 mb-4'):
                    with ui.column().classes('flex-1 p-3 bg-emerald-900/20 rounded-lg border border-emerald-500/30'):
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.icon('check_circle', size='sm').classes('text-emerald-400')
                            ui.label(f'{len(confirmed)} Confirmados').classes('font-medium text-emerald-300')

                    with ui.column().classes('flex-1 p-3 bg-amber-900/20 rounded-lg border border-amber-500/30'):
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.icon('pending', size='sm').classes('text-amber-400')
                            ui.label(f'{len(to_verify)} Por revisar').classes('font-medium text-amber-300')

            # Detalles
            if response.step_results:
                with ui.expansion('Ver detalles', icon='visibility').classes('w-full'):
                    for i, sr in enumerate(response.step_results[:20]):
                        stype = sr.get('step_type', '?')
                        sdata = sr.get('step_data', [])
                        with ui.column().classes('p-3 bg-slate-800/50 rounded mb-2'):
                            with ui.row().classes('items-center gap-2 mb-1'):
                                ui.label(f'#{i+1}').classes('text-xs text-slate-400')
                                ui.label(stype).classes('status-pill pill-info')
                            if sdata:
                                txt = str(sdata[0])[:300]
                                ui.label(txt + ('...' if len(str(sdata[0])) > 300 else '')).classes('text-xs text-slate-400 font-mono')

            # Exportar
            def do_export():
                export = {
                    'timestamp': datetime.now().isoformat(),
                    'model': state.model,
                    'total': response.total_entries,
                    'successful': response.successful_entries,
                    'failed': response.failed_entries,
                    'time': elapsed,
                    'confirmed': confirmed,
                    'to_verify': to_verify,
                    'results': response.step_results[:50] if response.step_results else []
                }
                ui.download(
                    json.dumps(export, indent=2, ensure_ascii=False).encode('utf-8'),
                    f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )

            ui.button('Exportar JSON', icon='download', on_click=do_export).props('flat').classes('mt-3')

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
        with ui.card().classes('w-full bg-slate-800/30 border border-purple-500/20 px-4 py-3'):
            data_container = ui.column().classes('w-full')
            entries_counter_label = None  # Ya no se usa, el contador está integrado en render_data_section
            with data_container:
                render_data_section()

        # Sección: Constructor de Pipeline - Layout 2 columnas (púrpura - paso 2 Configura)
        with ui.card().classes('w-full bg-slate-800/30 border border-purple-500/20 p-4'):
            with ui.row().classes('w-full justify-between items-center mb-1'):
                with ui.row().classes('items-center gap-2'):
                    with ui.element('div').classes('w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center pt-1'):
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
                model_selector_container = ui.column().classes('w-full')

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
                                        ui.label('Modelo Local').classes('text-xs text-slate-300')
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

        # Sección: Ejecutar (emerald oscuro - paso 3 - CTA principal)
        with ui.row().classes('w-full justify-center gap-4 mt-2'):
            run_btn = ui.button('Ejecutar Pipeline', icon='play_arrow', on_click=run_pipeline).props('unelevated size=lg')
            run_btn.style('background: linear-gradient(135deg, #0f4342, #0d3a39) !important; color: #6ee7b7 !important; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.5); padding: 12px 32px !important; font-size: 16px !important;')

        # Loading (verde - paso 3 Ejecuta)
        with ui.row().classes('loading-box items-center gap-3 justify-center') as lb:
            loading_box = lb
            loading_box.set_visibility(False)
            ui.spinner('dots', size='lg').classes('text-emerald-400')
            ui.label('Ejecutando pipeline...').classes('text-slate-300')

        # Resultados
        results_container = ui.column().classes('w-full')


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
