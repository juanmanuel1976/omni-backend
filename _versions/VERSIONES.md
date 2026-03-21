# Repositorio de Versiones — Crisalia

## Convención de nombres

`YYYY-MM-DD_NNN_archivo-original.ext`

- `YYYY-MM-DD` — fecha del cambio
- `NNN` — número de versión del día (001, 002, ...)
- `archivo-original.ext` — nombre del archivo modificado

---

## Registro

| Versión | Fecha | Archivo original | Cambio |
|---------|-------|-----------------|--------|
| 2026-03-17_001 | 2026-03-17 | omni_app.py | Eliminada restricción al mercado legal argentino del prompt del Tribunal |
| 2026-03-17_002 | 2026-03-17 | CLAUDE.md | Protocolo de validación simplificado: activación por marca `q.` |
| 2026-03-17_003 | 2026-03-17 | crisalia_bridge.py | Fase 1: argumento `-m` + inyección de CONTEXTO.md en prompt de jueces |
| 2026-03-17_004 | 2026-03-17 | crisalia_bridge.py | Fase 2: selección automática de tags según archivos tocados en el diff |
| 2026-03-17_005 | 2026-03-17 | omni_app.py | Monitor de costos: import costs_tracker, captura tokens por modelo, endpoint /api/costs/summary |
| 2026-03-18_001 | 2026-03-18 | costs_tracker.py | costs_tracker.py funcional con Supabase, streaming de los 3 modelos cubierto |
| 2026-03-18_002 | 2026-03-18 | omni_app.py | Estado dev estable: branding perspectivas, rate limit, feedback API, Modo Mejora |
| 2026-03-20_003 | 2026-03-20 | omni_app.py | **SNAPSHOT DEV SIN GPT** — estado justo antes de integrar GPT juez (commit 40ecc30). Punto de rollback limpio. |
| 2026-03-21_001 | 2026-03-21 | omni_app.py | **SNAPSHOT DEV CON GPT + debates_log** — GPT juez activo, logging completo de debates en Supabase |
| 2026-03-21_002 | 2026-03-21 | app.html | Snapshot frontend dev actual (Modo Mejora reubicado, Modo Creativo oculto) |
