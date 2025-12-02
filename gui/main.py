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
from datetime import datetime
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

from nicegui import ui, run

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


def _add_to_hf_cache(key: str, value: List[Dict]):
    """Añade al cache LRU, eliminando entradas antiguas si excede el límite."""
    # Si ya existe, moverlo al final (más reciente)
    if key in _hf_search_cache:
        _hf_search_cache.move_to_end(key)
    _hf_search_cache[key] = value
    # Eliminar entradas más antiguas si excede el límite
    while len(_hf_search_cache) > _HF_CACHE_MAX_ENTRIES:
        _hf_search_cache.popitem(last=False)


# Modelos precargados (se cargan al iniciar la página)
_preloaded_models: List[Dict] = []
_preload_status: str = 'pending'  # 'pending', 'loading', 'ready', 'error'

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

async def search_huggingface_models(query: str, limit: int = 8) -> List[Dict]:
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
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    try:
        # Pedir más resultados para filtrar mejor
        async with httpx.AsyncClient(timeout=2.0) as client:
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
    except Exception as e:
        print(f"HuggingFace search error: {e}")
    return []


async def preload_recommended_models():
    """Precarga modelos recomendados en background al iniciar la página."""
    global _preloaded_models, _preload_status

    if _preload_status == 'ready' and _preloaded_models:
        return  # Ya cargados

    _preload_status = 'loading'

    # Primero inicializar recursos del sistema (async, no bloquea)
    await init_system_resources_async()
    available_vram = get_available_vram_gb()
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    all_models = {}  # Usar dict para evitar duplicados por model_id

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Buscar modelos populares de text-generation ordenados por descargas
            response = await client.get(
                "https://huggingface.co/api/models",
                params={
                    "filter": "text-generation",
                    "sort": "downloads",
                    "direction": -1,
                    "limit": 25,  # Reducido de 50 para carga más rápida
                }
            )

            if response.status_code == 200:
                models = response.json()
                for m in models:
                    model_id = m.get('id', '')
                    if model_id in all_models:
                        continue

                    # Extraer parámetros
                    params_b = None
                    safetensors = m.get('safetensors', {})
                    if safetensors and 'total' in safetensors:
                        params_b = safetensors['total'] / 1_000_000_000
                    if params_b is None:
                        params_b = extract_params_from_name(model_id)

                    # Estimar VRAM
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

                    # Fecha
                    last_modified = m.get('lastModified', '')
                    date_str = last_modified[:10] if last_modified else ''

                    # Downloads
                    downloads_num = m.get('downloads', 0)
                    if downloads_num >= 1_000_000:
                        dl_str = f"{downloads_num/1_000_000:.1f}M"
                    elif downloads_num >= 1_000:
                        dl_str = f"{downloads_num/1_000:.0f}K"
                    else:
                        dl_str = str(downloads_num)

                    all_models[model_id] = {
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
                    }

        # Ordenar por compatibilidad + descargas
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
            return -score

        _preloaded_models = sorted(all_models.values(), key=sort_key)
        _preload_status = 'ready'
        print(f"Precargados {len(_preloaded_models)} modelos")

    except Exception as e:
        print(f"Error precargando modelos: {e}")
        _preload_status = 'error'


def filter_preloaded_models(query: str, limit: int = 8) -> List[Dict]:
    """Filtra modelos precargados por query con early stopping."""
    if not _preloaded_models or not query:
        return []

    query_lower = query.lower()
    matches = []
    for m in _preloaded_models:
        if query_lower in m['model_id'].lower():
            matches.append(m)
            if len(matches) >= limit:  # Early stop cuando alcanza el límite
                break
    return matches


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
        with ui.column().classes('w-full items-center gap-6 mb-4'):
            ui.label('Cómo funciona').classes('text-base text-slate-500 uppercase tracking-widest mb-2')

            with ui.row().classes('w-full justify-center items-start gap-0'):
                # Paso 1
                with ui.column().classes('items-center gap-2 flex-1 max-w-[200px]'):
                    with ui.element('div').classes('w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500/30 to-indigo-600/20 border-2 border-indigo-500/50 flex items-center justify-center'):
                        ui.label('1').classes('text-indigo-400 font-bold text-sm')
                    ui.label('Elige Modo').classes('text-xl font-semibold text-white')
                    ui.label('Pipeline o Benchmark').classes('text-sm text-slate-500 text-center')

                # Conector
                ui.element('div').classes('h-0.5 w-12 bg-gradient-to-r from-indigo-500/50 to-purple-500/50 mt-5')

                # Paso 2
                with ui.column().classes('items-center gap-2 flex-1 max-w-[200px]'):
                    with ui.element('div').classes('w-8 h-8 rounded-full bg-gradient-to-br from-purple-500/30 to-purple-600/20 border-2 border-purple-500/50 flex items-center justify-center'):
                        ui.label('2').classes('text-purple-400 font-bold text-sm')
                    ui.label('Configura').classes('text-xl font-semibold text-white')
                    with ui.column().classes('items-center gap-0'):
                        ui.label('Elige modelos de lenguaje').classes('text-sm text-slate-500 text-center')
                        ui.label('Crea un flujo de procesamiento').classes('text-sm text-slate-500 text-center')
                        ui.label('Guarda configuraciones').classes('text-sm text-slate-500 text-center')

                # Conector
                ui.element('div').classes('h-0.5 w-12 bg-gradient-to-r from-purple-500/50 to-emerald-500/50 mt-5')

                # Paso 3
                with ui.column().classes('items-center gap-2 flex-1 max-w-[200px]'):
                    with ui.element('div').classes('w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500/30 to-emerald-600/20 border-2 border-emerald-500/50 flex items-center justify-center'):
                        ui.label('3').classes('text-emerald-400 font-bold text-sm')
                    ui.label('Ejecuta').classes('text-xl font-semibold text-white')
                    ui.label('Visualiza y exporta resultados').classes('text-sm text-slate-500 text-center')

        # Título MODOS + Cards juntos para reducir espacio
        with ui.column().classes('w-full items-center gap-4 mt-2'):
            ui.label('Modos').classes('text-base text-slate-500 uppercase tracking-widest')

            # Features - Cards lado a lado (borde indigo para conectar con paso 1 "Elige Modo")
            with ui.row().classes('w-full gap-6 justify-center'):
                # Pipeline - clickable
                with ui.column().classes('feature-box w-[360px] gap-4 cursor-pointer items-center relative border-2 border-indigo-500/50').on('click', lambda: tabs.set_value('Pipeline') if tabs else None):
                    # Icono centrado
                    with ui.element('div').classes('w-14 h-14 rounded-xl bg-indigo-500/20 flex items-center justify-center'):
                        ui.icon('account_tree', size='lg').classes('text-indigo-400')

                    # Descripción centrada
                    with ui.row().classes('justify-center gap-1'):
                        ui.label('Diseña y Ejecuta').classes('text-lg text-slate-400')
                        ui.label('Pipelines').classes('text-lg text-indigo-400 font-bold')

                    # Features list - alineado a la izquierda
                    with ui.column().classes('gap-2 mt-2 w-full'):
                        for txt in ['Ejecución con múltiples inputs', 'Genera, extrae, y verifica', 'Exporta tus configuraciones']:
                            with ui.row().classes('items-center gap-3'):
                                ui.icon('check_circle', size='xs').classes('text-emerald-400')
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

                    # Features list - alineado a la izquierda
                    with ui.column().classes('gap-2 mt-2 w-full'):
                        for txt in ['Accuracy, precision, recall', 'Matriz de confusión', 'Análisis de errores']:
                            with ui.row().classes('items-center gap-3'):
                                ui.icon('check_circle', size='xs').classes('text-emerald-400')
                                ui.label(txt).classes('text-base text-slate-300')

                    # Flecha abajo derecha
                    ui.icon('arrow_forward', size='sm').classes('absolute bottom-4 right-4 text-indigo-400')

        # Recursos del sistema - diseño compacto con barras (carga asíncrona)
        with ui.column().classes('w-full items-center gap-3 mt-4'):
            ui.label('RECURSOS DEL SISTEMA').classes('text-sm text-slate-500 uppercase tracking-widest')

            # Contenedor para actualización asíncrona
            resources_container = ui.column().classes('w-full items-center')

            def render_resources_ui(resources: Dict[str, Any]):
                """Renderiza la UI de recursos del sistema."""
                resources_container.clear()
                with resources_container:
                    with ui.row().classes('items-center gap-8 px-6 py-4 rounded-xl bg-slate-800/30 border border-slate-700/50'):
                        # GPU status
                        with ui.column().classes('gap-1 min-w-[200px]'):
                            with ui.row().classes('items-center gap-2'):
                                if resources['gpu_detected']:
                                    ui.icon('memory', size='xs').classes('text-emerald-400')
                                    ui.label(resources['gpu_name'] or 'GPU').classes('text-sm text-slate-300 truncate')
                                else:
                                    ui.icon('memory', size='xs').classes('text-slate-500')
                                    ui.label('Sin GPU').classes('text-sm text-slate-500')

                            if resources['gpu_detected'] and resources['vram_total'] > 0:
                                with ui.row().classes('w-full items-center gap-2'):
                                    with ui.element('div').classes('flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden'):
                                        ui.element('div').classes('h-full bg-emerald-500 rounded-full').style(f"width: {resources['vram_used_percent']:.0f}%")
                                    vram_used = resources['vram_total'] - resources['vram_available']
                                    ui.label(f"{format_bytes(vram_used)}/{format_bytes(resources['vram_total'])}").classes('text-xs text-slate-500 whitespace-nowrap')

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
                                ui.label(f"{format_bytes(ram_used)}/{format_bytes(resources['ram_total'])}").classes('text-xs text-slate-500 whitespace-nowrap')

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
                                        ui.label(w['msg']).classes('text-xs text-slate-500')

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
                        with ui.row().classes('items-center gap-2 text-slate-500'):
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
        'data_entries': [],
        'detected_vars': set(),
        'json_view': False  # Toggle para vista JSON de datos
    }

    # Referencias UI
    steps_list_container = None
    step_config_container = None
    data_container = None
    results_container = None
    run_btn = None
    loading_box = None
    steps_counter_label = None
    entries_counter_label = None

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
                'uses_reference': True
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

    def add_data_entry():
        """Añade una entrada de datos vacía."""
        entry = {var: '' for var in local_state['detected_vars']}
        local_state['data_entries'].append(entry)
        refresh_data_section()

    def remove_data_entry(index: int):
        """Elimina una entrada de datos."""
        if 0 <= index < len(local_state['data_entries']):
            local_state['data_entries'].pop(index)
            refresh_data_section()

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

    def update_counters():
        """Actualiza los contadores dinámicos."""
        if steps_counter_label:
            steps_counter_label.set_text(f'{len(local_state["steps"])} pasos configurados')
        if entries_counter_label:
            entries_counter_label.set_text(f'{len(local_state["data_entries"])} entradas')

    def update_data_entry(entry_idx: int, var_name: str, value: str):
        """Actualiza un campo de una entrada de datos."""
        if 0 <= entry_idx < len(local_state['data_entries']):
            local_state['data_entries'][entry_idx][var_name] = value

    def on_data_upload(e):
        """Maneja la carga de datos desde archivo JSON."""
        try:
            content = e.content.read().decode('utf-8')
            data = json.loads(content)
            local_state['data_entries'] = data if isinstance(data, list) else [data]
            refresh_data_section()
            ui.notify(f'{len(local_state["data_entries"])} entrada(s) cargadas', type='positive')
        except Exception as ex:
            ui.notify(f'Error al cargar: {ex}', type='negative')

    def load_example_data():
        """Carga datos de ejemplo."""
        data = load_json_file(DEFAULT_PIPELINE_REFERENCE_DATA)
        if data:
            local_state['data_entries'] = data if isinstance(data, list) else [data]
            refresh_data_section()
            ui.notify(f'{len(local_state["data_entries"])} entrada(s) cargadas', type='positive')

    def refresh_builder():
        """Refresca el constructor de pasos con layout 2 columnas."""
        nonlocal steps_list_container, step_config_container
        if steps_list_container:
            steps_list_container.clear()
            with steps_list_container:
                render_steps_list()
        if step_config_container:
            step_config_container.clear()
            with step_config_container:
                render_step_config()
        update_counters()

    def refresh_data_section():
        """Refresca la sección de datos."""
        data_container.clear()
        with data_container:
            render_data_section()
        update_counters()

    def render_steps_list():
        """Renderiza la lista de pasos (columna izquierda)."""
        step_colors = {
            'generate': ('indigo', 'auto_awesome'),
            'parse': ('amber', 'find_in_page'),
            'verify': ('emerald', 'verified')
        }

        if not local_state['steps']:
            with ui.column().classes('w-full items-center justify-center py-6 gap-2'):
                ui.icon('construction', size='lg').classes('text-slate-600')
                ui.label('Sin pasos').classes('text-slate-500 text-sm')
                ui.label('Usa una plantilla o añade pasos').classes('text-xs text-slate-600')
        else:
            for idx, step in enumerate(local_state['steps']):
                stype = step.get('type', 'generate')
                color, icon = step_colors.get(stype, ('slate', 'help'))
                is_selected = idx == local_state['selected_step']

                # Card del paso (clickable para seleccionar)
                border_class = f'border-2 border-{color}-500' if is_selected else f'border border-{color}-500/30'
                bg_class = f'bg-{color}-500/10' if is_selected else 'bg-slate-800/50'

                with ui.card().classes(f'w-full {bg_class} {border_class} p-2 cursor-pointer').on('click', lambda i=idx: select_step(i)):
                    with ui.row().classes('w-full items-center justify-between'):
                        # Info del paso
                        with ui.row().classes('items-center gap-2 flex-1'):
                            # Número
                            with ui.element('div').classes(f'w-6 h-6 rounded-full bg-{color}-500/30 flex items-center justify-center'):
                                ui.label(str(idx + 1)).classes(f'text-xs font-bold text-{color}-400')
                            ui.icon(icon, size='xs').classes(f'text-{color}-400')
                            ui.label(stype.capitalize()).classes(f'text-sm text-{color}-300')

                        # Botones de acción terciarios (pequeños y sutiles)
                        with ui.row().classes('gap-0'):
                            ui.button(icon='arrow_upward', on_click=lambda i=idx: move_step_up(i)).props('flat round size=xs').classes('text-slate-500 hover:text-slate-300').tooltip('Subir')
                            ui.button(icon='arrow_downward', on_click=lambda i=idx: move_step_down(i)).props('flat round size=xs').classes('text-slate-500 hover:text-slate-300').tooltip('Bajar')
                            ui.button(icon='content_copy', on_click=lambda i=idx: duplicate_step(i)).props('flat round size=xs').classes('text-slate-500 hover:text-slate-300').tooltip('Duplicar')
                            ui.button(icon='delete', on_click=lambda i=idx: remove_step(i)).props('flat round size=xs').classes('text-red-500/50 hover:text-red-400').tooltip('Eliminar')

        # Botones para añadir pasos (con texto explicativo)
        ui.label('Añadir paso:').classes('text-xs text-slate-500 mt-4 mb-1')
        with ui.row().classes('w-full gap-2 flex-wrap'):
            # Generar
            with ui.button(on_click=lambda: add_step('generate')).props('flat dense no-caps').classes('bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 px-3'):
                with ui.row().classes('items-center gap-1'):
                    ui.icon('auto_awesome', size='xs')
                    ui.label('Generar').classes('text-xs')

            # Parsear
            with ui.button(on_click=lambda: add_step('parse')).props('flat dense no-caps').classes('bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 px-3'):
                with ui.row().classes('items-center gap-1'):
                    ui.icon('find_in_page', size='xs')
                    ui.label('Extraer').classes('text-xs')

            # Verificar
            with ui.button(on_click=lambda: add_step('verify')).props('flat dense no-caps').classes('bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 px-3'):
                with ui.row().classes('items-center gap-1'):
                    ui.icon('verified', size='xs')
                    ui.label('Verificar').classes('text-xs')

    def render_step_config():
        """Renderiza el panel de configuración del paso seleccionado (columna derecha)."""
        if not local_state['steps']:
            with ui.column().classes('w-full items-center justify-center py-12 gap-3'):
                ui.icon('touch_app', size='xl').classes('text-slate-600')
                ui.label('Selecciona una plantilla o añade un paso').classes('text-slate-500')
            return

        idx = local_state['selected_step']
        if idx >= len(local_state['steps']):
            idx = len(local_state['steps']) - 1
            local_state['selected_step'] = idx

        step = local_state['steps'][idx]
        stype = step.get('type', 'generate')

        step_colors = {
            'generate': ('indigo', 'auto_awesome', 'Generación de Texto'),
            'parse': ('amber', 'find_in_page', 'Extracción de Datos'),
            'verify': ('emerald', 'verified', 'Verificación')
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

        with ui.column().classes('w-full gap-4'):
            # System Prompt - textarea más grande
            ui.label('System Prompt').classes('text-sm text-slate-400 -mb-2')
            ui.textarea(
                value=params.get('system_prompt', ''),
                placeholder='Define el rol y comportamiento del modelo...',
                on_change=lambda e, i=idx: update_step_param(i, 'system_prompt', e.value)
            ).props('outlined dark rows=4 autogrow').classes('w-full')

            # User Prompt - textarea más grande
            ui.label('User Prompt').classes('text-sm text-slate-400 -mb-2')
            with ui.column().classes('gap-1'):
                ui.textarea(
                    value=params.get('user_prompt', ''),
                    placeholder='Instrucciones para el modelo. Usa {variable} para datos dinámicos.',
                    on_change=lambda e, i=idx: update_step_param(i, 'user_prompt', e.value)
                ).props('outlined dark rows=4 autogrow').classes('w-full')
                ui.label('Tip: Usa {variable} para insertar datos de entrada').classes('text-xs text-slate-500')

            # Parámetros numéricos
            ui.label('Parámetros de generación').classes('text-sm text-slate-400')
            with ui.row().classes('gap-4 flex-wrap'):
                ui.number(
                    label='Secuencias',
                    value=params.get('num_sequences', 1),
                    min=1, max=10,
                    on_change=lambda e, i=idx: update_step_param(i, 'num_sequences', int(e.value) if e.value else 1)
                ).props('outlined dark dense').classes('w-28').tooltip('Número de respuestas a generar')

                ui.number(
                    label='Max Tokens',
                    value=params.get('max_tokens', 200),
                    min=10, max=2000,
                    on_change=lambda e, i=idx: update_step_param(i, 'max_tokens', int(e.value) if e.value else 200)
                ).props('outlined dark dense').classes('w-28').tooltip('Longitud máxima de respuesta')

                ui.number(
                    label='Temperatura',
                    value=params.get('temperature', 0.7),
                    min=0, max=2, step=0.1,
                    on_change=lambda e, i=idx: update_step_param(i, 'temperature', float(e.value) if e.value else 0.7)
                ).props('outlined dark dense').classes('w-28').tooltip('0=determinista, 2=muy creativo')

    def render_parse_config(idx: int, step: Dict):
        """Renderiza configuración de paso parse."""
        params = step.get('parameters', {})
        rules = params.get('rules', [])

        with ui.column().classes('w-full gap-4'):
            # Reglas de parsing
            ui.label('Reglas de extracción').classes('text-sm text-slate-400')
            ui.label('Define cómo extraer datos del texto generado').classes('text-xs text-slate-500 -mt-2')

            if not rules:
                with ui.row().classes('items-center gap-2 p-3 bg-slate-900/30 rounded'):
                    ui.icon('info', size='xs').classes('text-slate-500')
                    ui.label('No hay reglas. Añade una regla para extraer datos.').classes('text-slate-500 text-sm')

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
            ui.label('Define criterios para validar los datos extraídos').classes('text-xs text-slate-500 -mt-2')

            if not methods:
                with ui.row().classes('items-center gap-2 p-3 bg-slate-900/30 rounded'):
                    ui.icon('info', size='xs').classes('text-slate-500')
                    ui.label('No hay métodos. Añade uno para verificar resultados.').classes('text-slate-500 text-sm')

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
                    ui.label('System Prompt').classes('text-xs text-slate-500 mt-2')
                    ui.textarea(
                        value=method.get('system_prompt', 'Responde Yes o No.'),
                        placeholder='Instrucciones para el verificador...',
                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'system_prompt', e.value)
                    ).props('outlined dark dense rows=2 autogrow').classes('w-full').tooltip('Instrucciones del sistema para la verificación')

                    # User Prompt
                    ui.label('User Prompt').classes('text-xs text-slate-500 mt-2')
                    ui.textarea(
                        value=method.get('user_prompt', ''),
                        placeholder='Pregunta de verificación. Usa {variable} para datos.',
                        on_change=lambda e, si=idx, mi=m_idx: update_verify_method(si, mi, 'user_prompt', e.value)
                    ).props('outlined dark dense rows=2 autogrow').classes('w-full').tooltip('Pregunta para verificar el contenido')

                    # Valid Responses
                    ui.label('Respuestas válidas').classes('text-xs text-slate-500 mt-2')
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

    def render_data_section():
        """Renderiza la sección de datos de entrada."""
        vars_list = sorted(local_state['detected_vars'])

        if not vars_list:
            with ui.column().classes('w-full items-center py-4'):
                ui.label('Configura los pasos para detectar variables').classes('text-slate-500 text-sm')
            return

        # Header con variables y toggle JSON
        with ui.row().classes('w-full justify-between items-center mb-3'):
            with ui.row().classes('gap-2 items-center'):
                ui.label('Variables:').classes('text-sm text-slate-400')
                for var in vars_list:
                    ui.label(f'{{{var}}}').classes('px-2 py-0.5 bg-indigo-500/20 text-indigo-300 rounded text-xs')

            # Toggle JSON/Form
            with ui.row().classes('gap-1 items-center'):
                ui.button(
                    icon='list' if local_state['json_view'] else 'code',
                    on_click=toggle_json_view
                ).props('flat round size=sm').tooltip('Alternar vista JSON/Formulario')

        # Vista JSON
        if local_state['json_view']:
            with ui.column().classes('w-full gap-2'):
                if local_state['data_entries']:
                    json_text = json.dumps(local_state['data_entries'], indent=2, ensure_ascii=False)
                    with ui.element('div').classes('json-preview w-full max-h-64 overflow-auto'):
                        ui.label(json_text).classes('whitespace-pre font-mono text-xs')
                else:
                    ui.label('[]').classes('text-slate-500 font-mono')
        else:
            # Vista formulario
            if not local_state['data_entries']:
                with ui.column().classes('w-full items-center py-4 gap-2'):
                    ui.icon('inbox', size='lg').classes('text-slate-600')
                    ui.label('No hay entradas de datos').classes('text-slate-500')
            else:
                # Mostrar entradas en grid compacto
                for e_idx, entry in enumerate(local_state['data_entries']):
                    with ui.card().classes('w-full bg-slate-800/30 p-3'):
                        with ui.row().classes('w-full justify-between items-center mb-2'):
                            ui.label(f'Entrada {e_idx + 1}').classes('text-sm font-medium text-slate-300')
                            ui.button(icon='delete', on_click=lambda i=e_idx: remove_data_entry(i)).props('flat round size=xs color=red').tooltip('Eliminar entrada')

                        with ui.row().classes('gap-3 flex-wrap'):
                            for var in vars_list:
                                ui.textarea(
                                    label=var,
                                    value=entry.get(var, ''),
                                    on_change=lambda e, ei=e_idx, v=var: update_data_entry(ei, v, e.value)
                                ).props('outlined dark dense rows=2 autogrow').classes('flex-1 min-w-[200px]')

        # Botones de acción (más claros y visibles)
        with ui.row().classes('gap-3 mt-4 flex-wrap items-center'):
            # Añadir entrada manual
            ui.button('Añadir entrada', icon='add', on_click=add_data_entry).props('flat dense no-caps').classes('bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 px-3')

            ui.label('o').classes('text-xs text-slate-600')

            # Cargar desde archivo
            with ui.row().classes('items-center gap-1 px-3 py-1 rounded bg-slate-700/50 hover:bg-slate-700 transition-colors'):
                ui.icon('upload_file', size='xs').classes('text-slate-400')
                ui.upload(on_upload=on_data_upload, auto_upload=True).props('flat dense accept=.json label="Cargar JSON"').classes('text-slate-400 text-xs')

            # Cargar ejemplo
            ui.button('Usar ejemplo', icon='science', on_click=load_example_data).props('flat dense no-caps').classes('bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 px-3')

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
                uses_reference=step_data.get('uses_reference', True),
                reference_step_numbers=step_data.get('reference_step_numbers')
            ))

        return built_steps

    async def run_pipeline():
        """Ejecuta el pipeline."""
        if not local_state['steps']:
            ui.notify('Configura al menos un paso', type='warning')
            return
        if not local_state['data_entries']:
            ui.notify('Añade al menos una entrada de datos', type='warning')
            return

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
            response = await run.io_bound(
                use_case.execute_with_references,
                request,
                local_state['data_entries']
            )

            elapsed = (datetime.now() - t0).total_seconds()
            show_results(response, elapsed)
            ui.notify(f'Completado: {response.successful_entries}/{response.total_entries}', type='positive')

        except Exception as ex:
            ui.notify(f'Error: {str(ex)[:100]}', type='negative')
            with results_container:
                with ui.row().classes('items-center gap-2 p-4 bg-red-900/20 rounded-lg'):
                    ui.icon('error', size='sm').classes('text-red-400')
                    ui.label(f'Error: {str(ex)}').classes('text-red-300 text-sm')
        finally:
            run_btn.enable()
            loading_box.set_visibility(False)

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
                                ui.label(f'#{i+1}').classes('text-xs text-slate-500')
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
            with ui.element('div').classes('w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 border border-indigo-500/30 flex items-center justify-center'):
                ui.icon('account_tree', size='sm').classes('text-indigo-400')
            with ui.column().classes('gap-0'):
                ui.label('Pipeline').classes('text-xl font-bold')
                ui.label('Constructor visual de flujos').classes('text-xs text-slate-500')

        # Sección de Plantillas - Cards visuales con indicador de selección
        with ui.column().classes('w-full gap-3'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('widgets', size='xs').classes('text-amber-400')
                ui.label('Inicio rápido').classes('text-sm font-medium text-slate-300')
                ui.label('Selecciona una plantilla para comenzar').classes('text-xs text-slate-500')

            with ui.row().classes('w-full gap-3 flex-wrap'):
                for key, tmpl in PIPELINE_TEMPLATES.items():
                    color = tmpl.get('color', 'slate')
                    icon = tmpl.get('icon', 'build')
                    is_selected = local_state['selected_template'] == key

                    # Card de plantilla (más visual)
                    border_style = f'border-2 border-{color}-500' if is_selected else f'border border-{color}-500/30'
                    bg_style = f'bg-{color}-500/10' if is_selected else 'bg-slate-800/50'

                    with ui.card().classes(f'{bg_style} {border_style} p-3 cursor-pointer hover:border-{color}-500/60 transition-all min-w-[140px]').on('click', lambda k=key: select_template(k)):
                        with ui.column().classes('items-center gap-2'):
                            # Icono
                            with ui.element('div').classes(f'w-8 h-8 rounded-lg bg-{color}-500/20 flex items-center justify-center'):
                                ui.icon(icon, size='xs').classes(f'text-{color}-400')
                            # Nombre
                            ui.label(tmpl['name']).classes(f'text-sm font-medium text-{color}-300 text-center')
                            # Descripción corta
                            if tmpl.get('description'):
                                ui.label(tmpl['description'][:40] + ('...' if len(tmpl.get('description', '')) > 40 else '')).classes('text-xs text-slate-500 text-center')
                            # Indicador de selección
                            if is_selected:
                                ui.icon('check_circle', size='xs').classes(f'text-{color}-400')

        # Sección 2: Constructor de Pipeline - Layout 2 columnas (púrpura - paso 2 Configura)
        with ui.card().classes('w-full bg-slate-800/30 border border-purple-500/20 p-4'):
            with ui.row().classes('w-full justify-between items-center mb-3'):
                with ui.row().classes('items-center gap-2'):
                    with ui.element('div').classes('w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center'):
                        ui.icon('build', size='xs').classes('text-purple-400')
                    with ui.column().classes('gap-0'):
                        ui.label('Constructor de Pipeline').classes('text-base font-semibold text-purple-300')
                        steps_counter_label = ui.label(f'{len(local_state["steps"])} pasos configurados').classes('text-xs text-slate-500')

            # Selector de modelo LLM (dentro del constructor)
            with ui.column().classes('gap-2 mb-3'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('smart_toy', size='xs').classes('text-indigo-400')
                    ui.label('Modelo de Lenguaje').classes('text-sm font-medium text-slate-300')

                with ui.element('div').classes('w-3/4 relative'):
                    with ui.element('div').classes(
                        'w-full flex items-center gap-3 px-3 py-2 bg-slate-900/50 '
                        'border border-slate-600/50 rounded-lg hover:border-indigo-500/50 '
                        'focus-within:border-indigo-500 transition-all'
                    ):
                        ui.icon('search', size='xs').classes('text-slate-500')
                        model_input = ui.input(
                            value=state.model,
                            placeholder='Buscar modelo en HuggingFace...'
                        ).props('borderless dense dark').classes('flex-1 text-sm')
                        search_indicator = ui.spinner('dots', size='sm').classes('text-indigo-400')
                        search_indicator.set_visibility(False)

                    # Dropdown de resultados (más alto para mejor visibilidad)
                    model_dropdown = ui.column().classes(
                        'absolute top-full left-0 right-0 w-full mt-1 bg-slate-800/95 backdrop-blur-sm '
                        'border border-slate-600/50 rounded-lg shadow-2xl z-50 max-h-72 overflow-y-auto'
                    )
                    model_dropdown.set_visibility(False)

                # Event handlers para el selector de modelo
                async def on_blur():
                    await asyncio.sleep(0.15)
                    model_dropdown.set_visibility(False)
                model_input.on('blur', on_blur)

                def on_model_change(e):
                    if e.args:
                        state.model = e.args
                model_input.on('update:model-value', on_model_change)

                search_state = {'last_query': ''}

                def render_dropdown_items(results: List[Dict]):
                    model_dropdown.clear()
                    with model_dropdown:
                        for r in results:
                            def make_click(val=r['value']):
                                def click():
                                    model_input.value = val
                                    state.model = val
                                    model_dropdown.set_visibility(False)
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

                            with ui.element('div').classes(
                                f'w-full px-3 py-2 cursor-pointer {bg_hover} transition-all border-b border-slate-700/30'
                            ).on('click', make_click()):
                                with ui.row().classes('items-center justify-between w-full gap-2'):
                                    ui.label(r.get('model_id', '')).classes('text-sm font-medium text-slate-100 truncate flex-1 min-w-0')
                                    with ui.element('span').classes(f'px-2 py-0.5 text-xs rounded border flex-shrink-0 {badge_class}'):
                                        ui.label(badge_text)
                                with ui.row().classes('items-center gap-3 mt-1'):
                                    ui.label(f"Params: {r.get('params_str', '?')}").classes('text-xs text-slate-400')
                                    ui.label(f"VRAM: ~{r.get('vram_str', '?')}").classes('text-xs text-slate-400')
                                    if r.get('date'):
                                        ui.label(f"Fecha: {r.get('date')}").classes('text-xs text-slate-500')
                                    ui.label(f"Descargas: {r.get('downloads', '?')}").classes('text-xs text-slate-500')
                    model_dropdown.set_visibility(True)

                def show_loading_message():
                    model_dropdown.clear()
                    with model_dropdown:
                        with ui.row().classes('items-center gap-2 p-3'):
                            ui.spinner('dots', size='sm').classes('text-indigo-400')
                            ui.label('Cargando modelos...').classes('text-sm text-slate-400')
                    model_dropdown.set_visibility(True)

                # Debouncing real para búsqueda de modelos
                search_debounce = {'task': None}

                async def do_model_search(query: str):
                    """Ejecuta la búsqueda real después del debounce."""
                    if len(query) < 2:
                        model_dropdown.set_visibility(False)
                        return

                    # Primero intentar con modelos precargados (instantáneo)
                    if _preload_status == 'ready' and _preloaded_models:
                        preloaded_results = filter_preloaded_models(query, limit=8)
                        if preloaded_results:
                            render_dropdown_items(preloaded_results)
                            return

                    # Esperar si está cargando precarga
                    if _preload_status == 'loading':
                        show_loading_message()
                        while _preload_status == 'loading':
                            await asyncio.sleep(0.1)
                            if query != search_state['last_query']:
                                return
                        if _preload_status == 'ready':
                            preloaded_results = filter_preloaded_models(query, limit=8)
                            if preloaded_results:
                                render_dropdown_items(preloaded_results)
                                return

                    # Búsqueda en API de HuggingFace
                    search_indicator.set_visibility(True)
                    model_dropdown.clear()
                    with model_dropdown:
                        with ui.row().classes('items-center gap-2 p-3'):
                            ui.spinner('dots', size='xs').classes('text-indigo-400')
                            ui.label('Buscando...').classes('text-sm text-slate-400')
                    model_dropdown.set_visibility(True)

                    try:
                        if query != search_state['last_query']:
                            return
                        results = await search_huggingface_models(query)
                        if query == search_state['last_query']:
                            if results:
                                render_dropdown_items(results)
                            else:
                                model_dropdown.clear()
                                with model_dropdown:
                                    ui.label('Sin resultados').classes('text-sm text-slate-500 p-3')
                                model_dropdown.set_visibility(True)
                    except Exception:
                        pass
                    finally:
                        search_indicator.set_visibility(False)

                async def on_model_search_debounced():
                    """Búsqueda con debouncing real de 300ms."""
                    query = model_input.value.strip() if model_input.value else ''
                    search_state['last_query'] = query

                    # Cancelar búsqueda anterior si existe
                    if search_debounce['task'] and not search_debounce['task'].done():
                        search_debounce['task'].cancel()

                    if len(query) < 2:
                        model_dropdown.set_visibility(False)
                        return

                    async def debounced_search():
                        try:
                            await asyncio.sleep(0.3)  # Esperar 300ms de inactividad
                            if query == search_state['last_query']:
                                await do_model_search(query)
                        except asyncio.CancelledError:
                            pass  # Búsqueda cancelada por nueva tecla

                    search_debounce['task'] = asyncio.create_task(debounced_search())

                model_input.on('keyup', lambda: on_model_search_debounced())

            # Layout 2 columnas: Lista de pasos (izq) + Config del paso seleccionado (der)
            with ui.row().classes('w-full gap-3'):
                # Columna izquierda: Lista de pasos
                with ui.column().classes('w-64 flex-shrink-0'):
                    ui.label('Pasos del Pipeline').classes('text-xs text-slate-500 uppercase tracking-wider mb-1')
                    steps_list_container = ui.column().classes('w-full gap-2')
                    with steps_list_container:
                        render_steps_list()

                # Columna derecha: Configuración del paso seleccionado
                with ui.column().classes('flex-1 bg-slate-900/30 rounded-lg p-3 min-h-[200px]'):
                    ui.label('Configuración del Paso').classes('text-xs text-slate-500 uppercase tracking-wider mb-1')
                    step_config_container = ui.column().classes('w-full')
                    with step_config_container:
                        render_step_config()

        # Sección 3: Datos de Entrada (púrpura - paso 2 Configura)
        with ui.card().classes('w-full bg-slate-800/30 border border-purple-500/20 p-4'):
            with ui.row().classes('w-full justify-between items-center mb-3'):
                with ui.row().classes('items-center gap-2'):
                    with ui.element('div').classes('w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center'):
                        ui.icon('input', size='xs').classes('text-purple-400')
                    with ui.column().classes('gap-0'):
                        ui.label('Datos de Entrada').classes('text-base font-semibold text-purple-300')
                        entries_counter_label = ui.label(f'{len(local_state["data_entries"])} entradas').classes('text-xs text-slate-500')

            data_container = ui.column().classes('w-full')
            with data_container:
                render_data_section()

        # Sección 4: Ejecutar (emerald oscuro - paso 3 - CTA principal)
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
                ui.label(f'{c.get("label_key")} = {c.get("label_value")}').classes('text-xs text-slate-500')
            else:
                ui.label('Sin configuración').classes('text-sm text-slate-500')

        entries_content.clear()
        with entries_content:
            if has_entries:
                n = len(state.benchmark_entries)
                ui.label(f'{n} entrada{"s" if n != 1 else ""}').classes('text-sm text-green-400')
                with ui.element('div').classes('json-preview mt-2'):
                    ui.label(format_json_preview(state.benchmark_entries[0]))
            else:
                ui.label('Sin entradas').classes('text-sm text-slate-500')

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
                    ui.label('Pred +').classes('text-xs text-slate-500 text-center')
                    ui.label('Pred -').classes('text-xs text-slate-500 text-center')

                    ui.label('Real +').classes('text-xs text-slate-500 self-center')
                    with ui.element('div').classes('cm-cell cm-good'):
                        ui.label(str(cm.get('true_positive', 0))).classes('text-lg font-bold text-green-400')
                    with ui.element('div').classes('cm-cell cm-bad'):
                        ui.label(str(cm.get('false_negative', 0))).classes('text-lg font-bold text-red-400')

                    ui.label('Real -').classes('text-xs text-slate-500 self-center')
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
            # Mostrar loading inmediatamente
            pipeline_placeholder.clear()
            with pipeline_placeholder:
                with ui.column().classes('w-full items-center justify-center py-20'):
                    ui.spinner('dots', size='lg').classes('text-indigo-400')
                    ui.label('Cargando Pipeline...').classes('text-slate-400 mt-4')

            # Ceder control al event loop para que se muestre el loading
            await asyncio.sleep(0.05)

            # Renderizar el contenido real
            pipeline_placeholder.clear()
            with pipeline_placeholder:
                pipeline_page()
            rendered_tabs['Pipeline'] = True

        elif tab_name == 'Benchmark' and not rendered_tabs['Benchmark']:
            # Mostrar loading inmediatamente
            benchmark_placeholder.clear()
            with benchmark_placeholder:
                with ui.column().classes('w-full items-center justify-center py-20'):
                    ui.spinner('dots', size='lg').classes('text-indigo-400')
                    ui.label('Cargando Benchmark...').classes('text-slate-400 mt-4')

            # Ceder control al event loop
            await asyncio.sleep(0.05)

            # Renderizar el contenido real
            benchmark_placeholder.clear()
            with benchmark_placeholder:
                benchmark_page()
            rendered_tabs['Benchmark'] = True

    def on_tab_change(e):
        """Handler para cambio de pestaña - dispara renderizado async."""
        tab_name = e.args if isinstance(e.args, str) else e.args
        current_tab['value'] = tab_name
        asyncio.create_task(render_tab_content(tab_name))

    # Escuchar cambios de pestaña de múltiples formas para mayor confiabilidad
    tabs.on('update:model-value', on_tab_change)
    tabs.on_value_change(lambda e: asyncio.create_task(render_tab_content(e.value)))

    # Iniciar precarga de modelos HuggingFace en background (no bloquea)
    asyncio.create_task(preload_recommended_models())


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ in {'__main__', '__mp_main__'}:
    ui.run(title='EsencIA', dark=True, port=8080, reload=True)
