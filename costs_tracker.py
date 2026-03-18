# ==============================================================================
# COSTS TRACKER — Crisalia LLM Cost Monitor v1.0
# Registra tokens y costo estimado de cada llamada a los modelos de IA.
# Se integra con omni_app.py modificando call_ai_model_no_stream.
# ==============================================================================

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

COSTS_DB_PATH = os.path.join(os.path.dirname(__file__), 'crisalia_costs.db')

# Precios por 1M tokens (USD) — actualizar si los proveedores cambian precios
PRICING = {
    'gemini':   {'input': 0.075,  'output': 0.30},
    'deepseek': {'input': 0.14,   'output': 0.28},
    'claude':   {'input': 0.25,   'output': 1.25},
}


@contextmanager
def get_db():
    conn = sqlite3.connect(COSTS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS llm_costs_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                model          TEXT    NOT NULL,
                endpoint       TEXT    NOT NULL DEFAULT 'unknown',
                tokens_input   INTEGER DEFAULT 0,
                tokens_output  INTEGER DEFAULT 0,
                cost_usd       REAL    DEFAULT 0.0
            )
        ''')


def calc_cost(model: str, tokens_input: int, tokens_output: int) -> float:
    pricing = PRICING.get(model, {'input': 0.0, 'output': 0.0})
    return (tokens_input * pricing['input'] + tokens_output * pricing['output']) / 1_000_000


def log_cost(model: str, endpoint: str, tokens_input: int, tokens_output: int) -> float:
    cost = calc_cost(model, tokens_input, tokens_output)
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO llm_costs_log '
                '(timestamp, model, endpoint, tokens_input, tokens_output, cost_usd) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (datetime.now().isoformat(), model, endpoint, tokens_input, tokens_output, cost)
            )
    except Exception as e:
        print(f"[costs_tracker] Error al guardar costo: {e}")
    return cost


def get_summary() -> dict:
    with get_db() as conn:
        # Totales globales
        totals = conn.execute('''
            SELECT
                ROUND(SUM(cost_usd), 6)              AS total_usd,
                SUM(tokens_input + tokens_output)    AS total_tokens,
                COUNT(*)                             AS total_calls
            FROM llm_costs_log
        ''').fetchone()

        # Totales de hoy
        today = conn.execute('''
            SELECT
                ROUND(SUM(cost_usd), 6) AS cost_today,
                COUNT(*)                AS calls_today
            FROM llm_costs_log
            WHERE DATE(timestamp) = DATE('now')
        ''').fetchone()

        # Por modelo
        by_model = conn.execute('''
            SELECT
                model,
                ROUND(SUM(cost_usd), 6)  AS cost_usd,
                SUM(tokens_input)        AS tokens_in,
                SUM(tokens_output)       AS tokens_out,
                COUNT(*)                 AS calls
            FROM llm_costs_log
            GROUP BY model
            ORDER BY cost_usd DESC
        ''').fetchall()

        # Por endpoint
        by_endpoint = conn.execute('''
            SELECT
                endpoint,
                ROUND(SUM(cost_usd), 6) AS cost_usd,
                COUNT(*)                AS calls
            FROM llm_costs_log
            GROUP BY endpoint
            ORDER BY cost_usd DESC
        ''').fetchall()

        # Evolución diaria últimos 30 días, desglosada por modelo
        daily = conn.execute('''
            SELECT
                DATE(timestamp)          AS date,
                model,
                ROUND(SUM(cost_usd), 6) AS cost_usd,
                COUNT(*)                AS calls
            FROM llm_costs_log
            WHERE timestamp >= DATE('now', '-30 days')
            GROUP BY DATE(timestamp), model
            ORDER BY date ASC
        ''').fetchall()

        # Últimas 50 llamadas
        recent = conn.execute('''
            SELECT timestamp, model, endpoint, tokens_input, tokens_output,
                   ROUND(cost_usd, 6) AS cost_usd
            FROM llm_costs_log
            ORDER BY id DESC
            LIMIT 50
        ''').fetchall()

        return {
            'totals':       dict(totals),
            'today':        dict(today),
            'by_model':     [dict(r) for r in by_model],
            'by_endpoint':  [dict(r) for r in by_endpoint],
            'daily':        [dict(r) for r in daily],
            'recent':       [dict(r) for r in recent],
            'pricing':      PRICING,
        }
