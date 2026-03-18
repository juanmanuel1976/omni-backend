# ==============================================================================
# COSTS TRACKER — Crisalia LLM Cost Monitor v2.0
# Registra tokens y costo estimado de cada llamada a los modelos de IA.
# Persistencia: Supabase (tabla llm_costs_log) — no usa SQLite.
# ==============================================================================

import os
import logging
import httpx
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Precios por 1M tokens (USD) — actualizar si los proveedores cambian precios
PRICING = {
    'gemini':   {'input': 0.075,  'output': 0.30},
    'deepseek': {'input': 0.14,   'output': 0.28},
    'claude':   {'input': 0.25,   'output': 1.25},
}

TABLE = "llm_costs_log"


def _read_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }


def _write_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def calc_cost(model: str, tokens_input: int, tokens_output: int) -> float:
    pricing = PRICING.get(model, {'input': 0.0, 'output': 0.0})
    return (tokens_input * pricing['input'] + tokens_output * pricing['output']) / 1_000_000


async def log_cost(model: str, endpoint: str, tokens_input: int, tokens_output: int) -> float:
    cost = calc_cost(model, tokens_input, tokens_output)
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("[costs_tracker] Supabase no configurado — costo no guardado.")
        return cost
    try:
        payload = {
            "timestamp":     datetime.now().isoformat(),
            "model":         model,
            "endpoint":      endpoint,
            "tokens_input":  tokens_input,
            "tokens_output": tokens_output,
            "cost_usd":      round(cost, 8),
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/{TABLE}",
                headers=_write_headers(),
                json=payload,
            )
    except Exception as e:
        logger.error(f"[costs_tracker] Error al guardar costo: {e}")
    return cost


async def get_summary() -> dict:
    """
    Retorna el resumen agregado de costos para el dashboard.

    TODO (próxima iteración — ESCALABILIDAD):
    El procesamiento actual descarga hasta 5000 filas a la RAM del servidor
    y agrega en Python. Migrar a VIEW o función RPC en Supabase cuando el
    volumen de registros supere los 10.000.
    """
    rows = await _fetch_all_rows()
    if not rows:
        return _empty_summary()

    today_str  = date.today().isoformat()
    cutoff_30d = (date.today() - timedelta(days=30)).isoformat()

    total_cost   = 0.0
    total_tokens = 0
    total_calls  = 0
    cost_today   = 0.0
    calls_today  = 0

    by_model    = {}
    by_endpoint = {}
    daily_dict  = {}

    for r in rows:
        cost     = r.get('cost_usd', 0) or 0
        tin      = r.get('tokens_input', 0) or 0
        tout     = r.get('tokens_output', 0) or 0
        model    = r.get('model', 'unknown')
        endpoint = r.get('endpoint', 'unknown')
        ts       = r.get('timestamp', '')
        day      = ts[:10] if ts else ''

        total_cost   += cost
        total_tokens += tin + tout
        total_calls  += 1

        if day == today_str:
            cost_today  += cost
            calls_today += 1

        if model not in by_model:
            by_model[model] = {'cost_usd': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'calls': 0}
        by_model[model]['cost_usd']   += cost
        by_model[model]['tokens_in']  += tin
        by_model[model]['tokens_out'] += tout
        by_model[model]['calls']      += 1

        if endpoint not in by_endpoint:
            by_endpoint[endpoint] = {'cost_usd': 0.0, 'calls': 0}
        by_endpoint[endpoint]['cost_usd'] += cost
        by_endpoint[endpoint]['calls']    += 1

        if day >= cutoff_30d:
            key = (day, model)
            daily_dict[key] = daily_dict.get(key, 0) + cost

    daily_list = [
        {'date': d, 'model': m, 'cost_usd': round(c, 6)}
        for (d, m), c in sorted(daily_dict.items())
    ]

    recent_list = [
        {
            'timestamp':     r.get('timestamp'),
            'model':         r.get('model'),
            'endpoint':      r.get('endpoint'),
            'tokens_input':  r.get('tokens_input'),
            'tokens_output': r.get('tokens_output'),
            'cost_usd':      round(r.get('cost_usd', 0), 6),
        }
        for r in rows[:50]
    ]

    return {
        'totals': {
            'total_usd':    round(total_cost, 6),
            'total_tokens': total_tokens,
            'total_calls':  total_calls,
        },
        'today': {
            'cost_today':  round(cost_today, 6),
            'calls_today': calls_today,
        },
        'by_model': [
            {'model': m, 'cost_usd': round(v['cost_usd'], 6),
             'tokens_in': v['tokens_in'], 'tokens_out': v['tokens_out'], 'calls': v['calls']}
            for m, v in sorted(by_model.items(), key=lambda x: -x[1]['cost_usd'])
        ],
        'by_endpoint': [
            {'endpoint': ep, 'cost_usd': round(v['cost_usd'], 6), 'calls': v['calls']}
            for ep, v in sorted(by_endpoint.items(), key=lambda x: -x[1]['cost_usd'])
        ],
        'daily':   daily_list,
        'recent':  recent_list,
        'pricing': PRICING,
    }


async def _fetch_all_rows() -> list:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/{TABLE}",
                headers={**_read_headers(), "Range": "0-4999"},
                params={"select": "*", "order": "id.desc"},
            )
            return r.json() if r.status_code == 200 else []
    except Exception as e:
        logger.error(f"[costs_tracker] Error al obtener datos: {e}")
        return []


def _empty_summary() -> dict:
    return {
        'totals':      {'total_usd': 0.0, 'total_tokens': 0, 'total_calls': 0},
        'today':       {'cost_today': 0.0, 'calls_today': 0},
        'by_model':    [],
        'by_endpoint': [],
        'daily':       [],
        'recent':      [],
        'pricing':     PRICING,
    }


def init_db():
    """No-op: mantenido por compatibilidad con omni_app.py. La tabla se crea en Supabase."""
    pass
