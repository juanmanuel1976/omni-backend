-- ==============================================================
-- CRISALIA — Setup de tablas en Supabase
-- Ejecutar en: Supabase Dashboard > SQL Editor > New Query
-- ==============================================================

-- Tabla: feedback de calidad de respuestas
-- Recibe rating 1-5 + comentario opcional del usuario final
CREATE TABLE IF NOT EXISTS llm_feedback (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rating      SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment     TEXT DEFAULT '',
    query_id    TEXT DEFAULT '',
    -- Para análisis futuro: detectar si la mejora fue por el modo "Mejorar respuesta IA"
    improve_mode BOOLEAN DEFAULT FALSE
);

-- Índice para consultas por fecha
CREATE INDEX IF NOT EXISTS idx_llm_feedback_timestamp ON llm_feedback (timestamp DESC);

-- RLS: deshabilitar para acceso desde backend con service key
ALTER TABLE llm_feedback DISABLE ROW LEVEL SECURITY;

-- ==============================================================
-- VERIFICAR que llm_costs_log ya existe (creada en iteración anterior)
-- Si no existe, crearla:
-- ==============================================================
CREATE TABLE IF NOT EXISTS llm_costs_log (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model           TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    tokens_input    INTEGER DEFAULT 0,
    tokens_output   INTEGER DEFAULT 0,
    cost_usd        NUMERIC(12, 8) DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_llm_costs_timestamp ON llm_costs_log (timestamp DESC);
ALTER TABLE llm_costs_log DISABLE ROW LEVEL SECURITY;

-- ==============================================================
-- Tabla: debates_log — registro completo de cada debate
-- Permite análisis de calidad, detección de sesgo y entrenamiento
-- de un evaluador propio. El prompt solo se guarda si el usuario
-- dio consentimiento explícito (campo save_prompt=true).
-- ==============================================================
CREATE TABLE IF NOT EXISTS debates_log (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lang                TEXT DEFAULT 'en',
    duration_ms         INTEGER DEFAULT 0,

    -- Prompt: solo con consentimiento del usuario
    prompt              TEXT DEFAULT NULL,

    -- Respuestas iniciales (antes de ver a los otros modelos)
    initial_gemini      TEXT DEFAULT '',
    initial_deepseek    TEXT DEFAULT '',
    initial_claude      TEXT DEFAULT '',

    -- Respuestas revisadas (después del debate)
    revised_gemini      TEXT DEFAULT '',
    revised_deepseek    TEXT DEFAULT '',
    revised_claude      TEXT DEFAULT '',

    -- Síntesis final
    synthesis           TEXT DEFAULT '',

    -- Evaluación GPT juez (referencia de calibración)
    gpt_score           SMALLINT DEFAULT NULL,
    gpt_observation     TEXT DEFAULT '',
    gpt_missed          TEXT DEFAULT '',

    -- Metadatos
    is_document         BOOLEAN DEFAULT FALSE,
    improve_mode        BOOLEAN DEFAULT FALSE,
    creative_mode       BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_debates_log_timestamp ON debates_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_debates_log_lang      ON debates_log (lang);
CREATE INDEX IF NOT EXISTS idx_debates_log_gpt_score ON debates_log (gpt_score);
ALTER TABLE debates_log DISABLE ROW LEVEL SECURITY;
