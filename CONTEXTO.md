# Contexto del Proyecto Crisalia
# Actualizar este archivo cada vez que cambie el estado del proyecto.
# Última actualización: 2026-03-18

[PROYECTO]
Crisalia es una plataforma SaaS multi-IA dialéctica. Orquesta un debate entre Gemini, DeepSeek y Claude para producir síntesis validadas por consenso, con trazabilidad del razonamiento por fuente. Orientada a profesionales que necesitan análisis robustos en cualquier industria.

[FASE]
Fase 1 — Preparación comercial. El producto técnico está funcional y desplegado en producción. El foco actual es monetización: definir el paywall, integrar Stripe y construir el sistema de auth. El paywall NO puede implementarse solo en el frontend; requiere validación de sesión en el backend antes de cada llamada a los modelos.

[ARQUITECTURA]
- Backend: FastAPI (Python) — omni_app.py v6.2.1, desplegado en Render
  - URL dev: https://omni-backend-dev.onrender.com (live desde 2026-03-18)
- Frontend: HTML/JS/TailwindCSS — index.html (landing), app.html (herramienta)
- IAs: Gemini 2.0 Flash, DeepSeek Chat, Claude 3 Haiku
- Logging: Supabase (prod) o JSONL local (fallback)
- RAG: rag_manager.py — pipeline semántico para documentos >120KB
- OCR: ocr_processor.py — para PDFs escaneados
- QA: crisalia_bridge.py — valida cambios vía git diff + Tribunal Dialéctico

[DECISIONES_IRREVOCABLES]
- El debate dialéctico siempre tiene 3 rondas (inicial, crítica cruzada, veredicto)
- Gemini actúa como árbitro final en el veredicto
- El frontend no toma decisiones de autorización; todo pasa por el backend
- El Tribunal de QA solo se ejecuta cuando el prompt del usuario termina en "q."

[RESTRICCIONES]
- Sin sistema de auth implementado aún: no hacer cambios de acceso o permisos en el frontend
- No degradar el motor dialéctico para usuarios free
- Render free tier: timeouts agresivos en cold start, no hacer operaciones síncronas largas sin manejo de error

[NEGOCIO]
- Modelo: SaaS B2B por suscripción — Free / Pro ($49) / Firma ($199) / Enterprise (custom)
- Mercado: profesionales de análisis de alto impacto, cualquier industria
- KPIs clave: conversión Free→Pro, churn mensual, consultas por usuario activo
