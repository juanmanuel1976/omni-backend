# Contexto del Proyecto Crisalia
# Actualizar este archivo cada vez que cambie el estado del proyecto.
# Última actualización: 2026-03-19

[PROYECTO]
Crisalia es una plataforma SaaS multi-IA dialéctica. Orquesta un debate entre Gemini, DeepSeek y Claude para producir síntesis validadas por consenso, con trazabilidad del razonamiento por fuente. Orientada a profesionales que necesitan análisis robustos en cualquier industria.

[FASE]
Fase 1 — Preparación comercial. El producto técnico está funcional y desplegado en producción. El foco actual es monetización: definir el paywall, integrar Stripe y construir el sistema de auth. El paywall NO puede implementarse solo en el frontend; requiere validación de sesión en el backend antes de cada llamada a los modelos.

[ARQUITECTURA]
Hay DOS backends separados en Render:

- Backend PROD (main): sirve crisalia.io — rama main de git
  - Plan Render: $20/mes (web service)
  - Logging: Supabase tabla consultas_log (preguntas + respuestas de usuarios reales)
  - NO tocar con el monitor de costos ni con experimentos

- Backend DEV/API (dev): para desarrollo y uso interno del equipo — rama dev de git
  - URL: https://omni-backend-dev.onrender.com (live desde 2026-03-18)
  - Plan Render: $7/mes (API/worker)
  - Logging de costos LLM: Supabase tabla llm_costs_log → visualizado en costs.html
  - Este es el backend que usamos en este repo

- Frontend: HTML/JS/TailwindCSS — index.html (landing), app.html (herramienta)
- IAs: Gemini 2.0 Flash, DeepSeek Chat, Claude 3 Haiku
- RAG: rag_manager.py — pipeline semántico para documentos >120KB
- OCR: ocr_processor.py — para PDFs escaneados
- QA: crisalia_bridge.py — valida cambios vía git diff + Tribunal Dialéctico

[DECISIONES_IRREVOCABLES]
- El debate dialéctico siempre tiene 3 rondas (inicial, crítica cruzada, veredicto)
- Gemini actúa como árbitro final en el veredicto
- El frontend no toma decisiones de autorización; todo pasa por el backend
- El Tribunal de QA solo se ejecuta cuando el prompt del usuario termina en "q."
- El monitor de costos (costs.html + llm_costs_log) aplica SOLO al backend dev, nunca al prod

[RESTRICCIONES]
- Sin sistema de auth implementado aún: no hacer cambios de acceso o permisos en el frontend
- No degradar el motor dialéctico para usuarios free
- Backend dev en Render $7: timeouts agresivos en cold start, no hacer operaciones síncronas largas sin manejo de error

[NEGOCIO]
- Modelo: SaaS B2B por suscripción — Free / Pro ($49) / Firma ($199) / Enterprise (custom)
- Mercado: profesionales de análisis de alto impacto, cualquier industria
- KPIs clave: conversión Free→Pro, churn mensual, consultas por usuario activo
- Costos fijos actuales: Render $7/mes (dev) + $20/mes (prod)

[DECISIONES_2026-03-19]
- API pública ("Crisalia API"): abrir el motor de debate para que otras empresas lo integren en sus sistemas
  - Tier API Analítica (~$99/mes): acceso a consenso, disidencias, refinamiento, síntesis, export
  - Tier API Creativa (~$199/mes): todo lo anterior + Modo Creativo
  - Enterprise: precio a negociar
- Features de la web a integrar en la API: % consenso, puntos de disidencia, prompts sugeridos dinámicos, síntesis con trazabilidad, export .txt/.pdf
- Mejora pendiente: subida de múltiples documentos (hoy solo se puede subir 1)
- PDF descargable nunca funcionó bien — intentos previos rompieron el código. Debe hacerse desde el servidor, no desde el frontend.

[MODO_CREATIVO — DEFINICION VALIDADA POR TRIBUNAL 2026-03-19]
- NO asignar roles fijos (economista, filósofo, técnico) — eso asume el tema de antemano
- Cada LLM debe: pensar el problema desde MÍNIMO DOS perspectivas distintas, tenerlas en cuenta y comentarlas en su respuesta inicial
- En la ronda de crítica cruzada: cada LLM tendrá más perspectivas (las propias + las de los otros) para aceptar o desestimar
- Resultado: diversidad real sin forzar un ángulo que puede no encajar con el prompt del usuario
- Veredicto del Tribunal: idea sólida, pero riesgo de "diversidad fingida" — los modelos podrían cumplir la forma sin el fondo
- Riesgo adicional (DeepSeek): si se normaliza el patrón, los modelos aprenden a simular debate en vez de pensar diferente
- Estado: pendiente de experimento comparativo (normal vs. modo creativo con la misma pregunta)

[BENCHMARKS — VEREDICTO TRIBUNAL 2026-03-19]
- Posible medir calidad sin jueces humanos infalibles: combinar IA como juez + métricas automáticas
- Métricas útiles: exactitud en preguntas con respuesta única, consistencia factual (FactScore), consistencia lógica, robustez
- Benchmarks de referencia 2024: Arena-Hard, LLMBar, SelfCheckGPT
- Búsqueda web en tiempo real dentro del debate: recomendada, pero solo cuando el modelo no está seguro de un dato; los modelos deben citar fuentes
- Estado: pendiente de decisión sobre si incorporar a Crisalia

[BENCHMARK — primer resultado oficial 2026-03-19]
- Experimento: 10 preguntas reales de Compliance & Ética (categorías EQS)
- Resultado: Crisalia ganó 10/10 vs. Claude respondiendo solo
- Puntaje: Crisalia 188/200 vs. IA sola 140/200 (+34% de mejora)
- Tiempo promedio por consulta: 52 segundos
- Costo promedio por consulta: $0.007 USD
- Dato de venta válido: "+34% de mejora vs. IA sola en compliance" (con la aclaración de que es benchmark interno)
- Próximo paso: repetir con preguntas externas del PDF de EQS (requiere registro gratuito en eqs.com)

[PROXIMOS_PASOS — orden de prioridad]
1. Experimento comparativo del Modo Creativo (misma pregunta, flujo normal vs. flujo con múltiples perspectivas) — PENDIENTE de elegir pregunta
2. Calcular costo real por consulta en modo creativo (más llamadas a LLMs = más gasto)
3. Prompts sugeridos dinámicos en el backend (hoy son fijos en el frontend)
4. PDF descargable que funcione (desde el servidor)
5. Implementar Modo Creativo en el código
6. Documentar la API para terceros
7. (Futuro) Subida de múltiples documentos

[PERFIL_USUARIO]
- No es programador ni experto técnico ni económico
- Toda explicación debe ser en lenguaje cotidiano, sin jerga
- Los pasos técnicos deben traducirse a acciones de negocio comprensibles
