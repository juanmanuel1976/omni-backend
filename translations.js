const TRANSLATIONS = {
  es: {
    // Page title
    'page.title': 'Crisalia | Inteligencia Sintetizada para Decisiones Críticas',

    // Nav
    'nav.difference': 'La Diferencia',
    'nav.features': 'Características',
    'nav.use_cases': 'Casos de Uso',
    'nav.go_tool': 'Ir a la Herramienta',

    // Hero (index)
    'hero.subtitle': 'Crisalia no te da una respuesta, te entrega una estrategia. Orquestamos un panel de IAs expertas para generar insights robustos, transparentes y libres del sesgo de un modelo único.',
    'hero.cta_primary': 'Probar la Herramienta',
    'hero.cta_secondary': 'Descubrir Cómo',

    // Method section (index)
    'method.label': 'EL MÉTODO',
    'method.title': 'De una Voz a una Sinfonía',
    'method.trad_title': 'IA Tradicional: La Caja Negra',
    'method.trad_subtitle': 'Una única perspectiva, un riesgo inherente.',
    'method.your_query': 'Tu Query',
    'method.single_model': 'Modelo Único',
    'method.one_answer': 'Una Respuesta',
    'method.one_answer_desc': 'Opaca, potencialmente sesgada.',
    'method.crisalia_title': 'Crisalia: El Debate Dialéctico',
    'method.crisalia_subtitle': 'Múltiples expertos para un insight superior.',
    'method.synthetic_debate': '⟨ Debate Sintético ⟩',
    'method.strategic_insight': 'Insight Estratégico',
    'method.strategic_insight_desc': 'Validado, transparente y accionable.',

    // Features section (index & app)
    'features.label': 'PLATAFORMA',
    'features.title': 'Un Toolkit para la Inteligencia Aumentada',
    'features.consensus_title': 'Métrica de Consenso',
    'features.consensus_desc': 'Cuantificamos la certeza. Un bajo consenso indica temas complejos que requieren tu dirección. Un alto consenso valida conclusiones factuales y robustas.',
    'features.high_consensus': 'Alto Consenso (92%)',
    'features.low_consensus': 'Bajo Consenso (18%)',
    'features.chart_agreement': 'Acuerdo',
    'features.chart_disagreement': 'Desacuerdo',
    'features.chart_perspective_a': 'Perspectiva A',
    'features.chart_perspective_b': 'Perspectiva B',
    'features.docs_title': 'Análisis de Documentos',
    'features.docs_desc': 'Procesa informes extensos, papers académicos o contratos. Arroja tus `.pdf`, `.txt`, y `.md` y deja que Crisalia extraiga la inteligencia.',
    'features.trace_title': 'Trazabilidad del Razonamiento',
    'features.trace_desc': 'Auditabilidad total. La síntesis final utiliza un código de colores para trazar cada idea hasta su modelo de origen, eliminando la "caja negra" del análisis.',
    'features.source_gemini': 'Perspectiva Analítica',
    'features.source_deepseek': 'Perspectiva Crítica',
    'features.source_claude': 'Perspectiva Sintetizadora',

    // App-specific features section
    'features.smart_consensus_title': 'Métrica de Consenso Inteligente',
    'features.smart_consensus_desc': 'Sistema adaptativo que ajusta automáticamente el proceso según tu nivel de confianza deseado. Un bajo consenso activa prompts de refinamiento contextuales.',
    'features.dissidence_title': 'Análisis de Disidencias Automático',
    'features.dissidence_desc': 'Identifica automáticamente puntos de divergencia conceptual entre modelos y genera prompts específicos para resolver diferencias metodológicas.',
    'features.trace_desc_app': 'Auditabilidad total. La síntesis final utiliza un código de colores para trazar cada idea hasta su modelo de origen, eliminando la "caja negra" del análisis.',

    // Use cases section (index)
    'usecases.label': 'APLICACIONES',
    'usecases.title': 'Para Decisiones de Alto Impacto',
    'usecases.subtitle': 'Crisalia está diseñado para profesionales que no pueden permitirse un análisis superficial.',
    'usecases.strategic_title': 'Análisis Estratégico',
    'usecases.strategic_desc': 'Evalúa informes de mercado, analiza competidores y sintetiza tendencias para definir la dirección de tu negocio.',
    'usecases.rd_title': 'Investigación y Desarrollo (I+D)',
    'usecases.rd_desc': 'Procesa y resume literatura científica compleja, identifica brechas en el conocimiento y acelera la innovación.',
    'usecases.dd_title': 'Due Diligence Financiero y Legal',
    'usecases.dd_desc': 'Analiza contratos, detecta riesgos en documentos extensos y genera resúmenes ejecutivos en minutos, no días.',

    // Footer
    'footer.rights': '© 2025 Crisalia. Todos los derechos reservados.',
    'footer.rights_app': '© 2025 Crisalia - Mediacon SRL. Todos los derechos reservados.',
    'footer.acknowledgments': 'Agradecimientos',

    // App hero
    'app.hero.title': 'Inteligencia Sintetizada para Decisiones Críticas',
    'app.hero.subtitle': 'Crisalia no te da una respuesta, te entrega una estrategia. Orquestamos un panel de IAs expertas para generar insights robustos, transparentes y libres del sesgo de un modelo único.',

    // App step 1
    'step1.title': '1. Proporciona Contenido para Análisis',
    'step1.file_label': 'Seleccione los documentos para el análisis (.txt, .md, .pdf)',
    'step1.file_drag': 'o arrastre y suelte los archivos aquí',
    'step1.divider': 'O',
    'step1.textarea_placeholder': 'Pega el texto directamente aquí...',
    'step1.method_label': 'Metodología de Análisis:',
    'step1.mode_direct': 'Análisis Conciso',
    'step1.mode_perspectives': 'Análisis Exhaustivo',
    'step1.mode_debate': 'Análisis Dialéctico',
    'step1.generate_btn': 'Iniciar Análisis',

    // Dialectic config
    'dialectic.title': 'Configuración del Análisis Dialéctico',
    'dialectic.confidence_label': 'Nivel de Confianza Deseado:',
    'dialectic.exploratory': 'Exploratorio',
    'dialectic.exploratory_sub': '>60% consenso',
    'dialectic.balanced': 'Equilibrado',
    'dialectic.balanced_sub': '>70% consenso',
    'dialectic.conservative': 'Conservador',
    'dialectic.conservative_sub': '>80% consenso',
    'dialectic.note': '<strong class="text-violet-300">Nota:</strong> Si el consenso inicial no alcanza tu nivel deseado, el sistema sugerirá automáticamente prompts de refinamiento para mejorar la convergencia entre los modelos de IA.',

    // Timeline
    'timeline.title': 'Fases del Análisis',
    'timeline.step1': 'Consulta',
    'timeline.step2': 'Análisis Individual',
    'timeline.step3': 'Debate / Síntesis',
    'timeline.step4': 'Resultado Final',

    // Initial responses
    'initial.title': 'Respuestas Iniciales Individuales',
    'initial.subtitle': 'Revisa las perspectivas individuales antes del proceso dialéctico',

    // Consensus
    'consensus.title': 'Nivel de Consenso Entre Modelos',

    // Friction / dissidence
    'friction.title': 'Puntos de Disidencia Detectados',
    'friction.subtitle': 'Los modelos de IA tienen perspectivas diferentes en estos aspectos:',
    'friction.suggested_title': 'Prompts Sugeridos para Refinamiento',
    'friction.suggested_subtitle': 'Selecciona y edita un prompt para dirigir el refinamiento del debate:',
    'friction.custom_label': 'O escribe tu propio prompt de refinamiento:',
    'friction.custom_placeholder': "Ej: 'Enfócate en la viabilidad económica y reduce las diferencias metodológicas...'",
    'friction.refine_btn': 'Refinar Debate',

    // Step 2
    'step2.title': '2. Revisa las Perspectivas',
    'step2.subtitle': 'Evalúa las perspectivas generadas. Tu selección guiará la síntesis final.',

    // Step 3 buttons
    'btn.key_points': 'Generar Puntos Clave',
    'btn.strategic_synthesis': 'Desarrollar Síntesis Estratégica',

    // Final result
    'final.title': 'Síntesis Estratégica Final',
    'tab.synthesis': 'Síntesis',
    'tab.arguments': 'Argumentos Finales',
    'tab.history': 'Historial de Análisis',

    // Action buttons
    'btn.copy': 'Copiar',
    'btn.export_txt': 'Exportar .txt',
    'btn.deepen': 'Profundizar Análisis',
    'btn.download_txt': 'Descargar .txt',
    'btn.download_pdf': 'Descargar .pdf',

    // Card actions
    'action.include_synthesis': 'Incluir en Síntesis',
    'action.mark_counterpoint': 'Marcar como Contrapunto',

    // Errors
    'error.timeout_title': 'Análisis Excedió el Tiempo Límite',
    'error.connection_title': 'Error de Conexión',

    // Labels
    'label.reasoning_trace': 'Trazabilidad del Razonamiento:',
    'label.files_for_analysis': 'Archivos para análisis:',

    // Mode texts
    'mode.direct': '<strong>Análisis Conciso:</strong> Genera una respuesta directa y concisa, ideal para datos rápidos.',
    'mode.perspectives': '<strong>Análisis Exhaustivo:</strong> Desarrolla múltiples perspectivas y explora matices, ideal para investigación profunda.',
    'mode.debate': '<strong>Análisis Dialéctico:</strong> Somete las perspectivas a un examen cruzado para forjar una síntesis más robusta y validada.',

    // setLoading messages
    'loading.analyzing': 'Analizando...',
    'loading.starting': 'Iniciando análisis...',
    'loading.docs': 'Procesando documentos y preparando análisis RAG...',
    'loading.debate': 'Iniciando debate dialéctico...',
    'loading.perspectives': 'Generando perspectivas...',

    // Alert messages
    'alert.no_instruction': 'Por favor, proporciona una instrucción para guiar el debate.',
    'alert.no_question': 'Por favor, introduce una pregunta o consulta para analizar los documentos.',
    'alert.no_content': 'Por favor, proporciona contenido para análisis, ya sea subiendo archivos o pegando texto.',
    'error.server_prefix': 'Error del servidor:',
    'error.no_connection': 'No se pudo conectar al servidor.',
    'loading.retry': 'Reintentando',

    // Dynamic JS strings
    'error.timeout_body': "Esto suele ocurrir con documentos muy grandes o en modo 'Análisis Dialéctico', ya que los servidores gratuitos tienen un límite de tiempo. Intenta con un fragmento de texto más corto o un modo de análisis más simple.",
    'debate.in_progress_title': 'Análisis Dialéctico en Progreso...',
    'debate.in_progress_subtitle': 'Las perspectivas están siendo examinadas cruzadamente. Este proceso puede tomar más tiempo.',
    'debate.in_progress_subtitle_refine': 'Las perspectivas están siendo examinadas cruzadamente con tu contexto de refinamiento.',
    'debate.perspectives_completed': 'Perspectivas Individuales Completadas',
    'debate.perspectives_review': 'Revisa los argumentos refinados de cada modelo antes del resultado final.',
    'debate.synthesis_title': 'Síntesis del Análisis Dialéctico',
    'debate.synthesis_title_refined': 'Síntesis del Análisis Dialéctico Refinado',
    'creative.toggle': 'Modo Creativo',
    'creative.desc': 'Perspectivas en tensión real',
    'creative.badge': '✦ Modo Creativo',
    'creative.tooltip': 'Los modelos exploran múltiples perspectivas en tensión antes de responder, en lugar de responder desde un único ángulo. Aumenta la diversidad genuina del debate.',
    'refine.summary_title': 'Resumen de Puntos Clave',
    'refine.report_title': 'Síntesis Estratégica',
    'refine.report_refined_title': 'Síntesis Estratégica Refinada',
    'loading.debate_first': 'Ejecutando análisis multi-agente avanzado con validación cruzada... (3-4 minutos)',
    'loading.debate_refine': 'Refinando perspectivas con tu orientación específica... (1-2 minutos)',
    'alert.write_direction': 'Por favor, escribe una dirección para el análisis',
    'consensus.semantic_label': 'Análisis semántico:',
    'consensus.strong_label': 'Consenso fuerte:',
    'consensus.moderate_label': 'Consenso moderado:',
    'consensus.divergence_label': 'Divergencias:',
    'consensus.breakdown_btn': 'Ver desglose detallado',
    'consensus.basic_label': 'Análisis básico por palabras clave',
    'consensus.matches': 'coincidencias de',
    'consensus.concepts': 'conceptos',
    'prompt.resolve_diff': 'Resolver Diferencias Conceptuales',
    'prompt.balance_depth': 'Equilibrar Profundidad',
    'prompt.maximize_consensus': 'Maximizar Consenso',
    'exploration.options_subtitle': 'Puedes dirigir el análisis hacia aspectos específicos:',
    'exploration.btn.alternative': 'Explorar perspectivas contrarias',
    'exploration.btn.deeper': 'Profundizar en aspectos específicos',
    'exploration.btn.skeptical': 'Análisis escéptico',
    'exploration.btn.custom': 'Prompt personalizado',
    'exploration.custom_label': 'Dirige el análisis:',
    'exploration.custom_placeholder': "Ej: 'Enfócate en los riesgos financieros' o 'Analiza desde perspectiva usuario final'",
    'exploration.execute_btn': 'Ejecutar Análisis Dirigido',

    // Acknowledgments modal
    'ack.title': 'Agradecimientos',
    'ack.intro': 'CRISALIA fue posible gracias a la colaboración y apoyo de:',
    'ack.team_label': 'Equipo de Desarrollo',
    'ack.institutions_label': 'Instituciones',
    'ack.footer': 'Su contribución fue fundamental para el desarrollo de esta tecnología.',
    'footer.contact': 'Contacto',
    'export.title': 'Crisalia - Reporte de Análisis',
    'export.initial_query': 'CONSULTA INICIAL:',
    'export.trace': 'TRAZA DEL ANÁLISIS (Historial):',
    'export.iteration': 'Iteración',
    'export.query': 'Consulta:',
    'export.response': 'Respuesta:',
    'export.final_result': 'RESULTADO FINAL:',
  },

  en: {
    // Page title
    'page.title': 'Crisalia | Synthesized Intelligence for Critical Decisions',

    // Nav
    'nav.difference': 'The Difference',
    'nav.features': 'Features',
    'nav.use_cases': 'Use Cases',
    'nav.go_tool': 'Open Tool',

    // Hero (index)
    'hero.subtitle': "Crisalia doesn't give you an answer — it delivers a strategy. We orchestrate a panel of expert AIs to generate insights that are robust, transparent, and free from single-model bias.",
    'hero.cta_primary': 'Try the Tool',
    'hero.cta_secondary': 'See How It Works',

    // Method section (index)
    'method.label': 'THE METHOD',
    'method.title': 'From One Voice to a Symphony',
    'method.trad_title': 'Traditional AI: The Black Box',
    'method.trad_subtitle': 'A single perspective, an inherent risk.',
    'method.your_query': 'Your Query',
    'method.single_model': 'Single Model',
    'method.one_answer': 'One Answer',
    'method.one_answer_desc': 'Opaque, potentially biased.',
    'method.crisalia_title': 'Crisalia: The Dialectical Debate',
    'method.crisalia_subtitle': 'Multiple experts for superior insight.',
    'method.synthetic_debate': '⟨ Synthetic Debate ⟩',
    'method.strategic_insight': 'Strategic Insight',
    'method.strategic_insight_desc': 'Validated, transparent, and actionable.',

    // Features section (index & app)
    'features.label': 'PLATFORM',
    'features.title': 'A Toolkit for Augmented Intelligence',
    'features.consensus_title': 'Consensus Metric',
    'features.consensus_desc': 'We quantify certainty. Low consensus signals complex topics requiring your direction. High consensus validates factual, robust conclusions.',
    'features.high_consensus': 'High Consensus (92%)',
    'features.low_consensus': 'Low Consensus (18%)',
    'features.chart_agreement': 'Agreement',
    'features.chart_disagreement': 'Disagreement',
    'features.chart_perspective_a': 'Perspective A',
    'features.chart_perspective_b': 'Perspective B',
    'features.docs_title': 'Document Analysis',
    'features.docs_desc': 'Process extensive reports, academic papers, or contracts. Upload your `.pdf`, `.txt`, and `.md` files and let Crisalia extract the intelligence.',
    'features.trace_title': 'Reasoning Traceability',
    'features.trace_desc': 'Full auditability. The final synthesis uses color-coding to trace each idea back to its source model, eliminating the "black box" of analysis.',
    'features.source_gemini': 'Analytical Perspective',
    'features.source_deepseek': 'Critical Perspective',
    'features.source_claude': 'Synthesizing Perspective',

    // App-specific features section
    'features.smart_consensus_title': 'Intelligent Consensus Metric',
    'features.smart_consensus_desc': 'Adaptive system that automatically adjusts the process according to your desired confidence level. Low consensus triggers contextual refinement prompts.',
    'features.dissidence_title': 'Automatic Dissidence Analysis',
    'features.dissidence_desc': 'Automatically identifies points of conceptual divergence between models and generates specific prompts to resolve methodological differences.',
    'features.trace_desc_app': 'Full auditability. The final synthesis uses color-coding to trace each idea back to its source model, eliminating the "black box" of analysis.',

    // Use cases section (index)
    'usecases.label': 'APPLICATIONS',
    'usecases.title': 'For High-Impact Decisions',
    'usecases.subtitle': 'Crisalia is designed for professionals who cannot afford a superficial analysis.',
    'usecases.strategic_title': 'Strategic Analysis',
    'usecases.strategic_desc': 'Evaluate market reports, analyze competitors, and synthesize trends to define your business direction.',
    'usecases.rd_title': 'Research & Development (R&D)',
    'usecases.rd_desc': 'Process and summarize complex scientific literature, identify knowledge gaps, and accelerate innovation.',
    'usecases.dd_title': 'Financial & Legal Due Diligence',
    'usecases.dd_desc': 'Analyze contracts, detect risks in extensive documents, and generate executive summaries in minutes, not days.',

    // Footer
    'footer.rights': '© 2025 Crisalia. All rights reserved.',
    'footer.rights_app': '© 2025 Crisalia - Mediacon SRL. All rights reserved.',
    'footer.acknowledgments': 'Acknowledgments',

    // App hero
    'app.hero.title': 'Synthesized Intelligence for Critical Decisions',
    'app.hero.subtitle': "Crisalia doesn't give you an answer — it delivers a strategy. We orchestrate a panel of expert AIs to generate insights that are robust, transparent, and free from single-model bias.",

    // App step 1
    'step1.title': '1. Provide Content for Analysis',
    'step1.file_label': 'Select documents for analysis (.txt, .md, .pdf)',
    'step1.file_drag': 'or drag and drop files here',
    'step1.divider': 'OR',
    'step1.textarea_placeholder': 'Paste text directly here...',
    'step1.method_label': 'Analysis Methodology:',
    'step1.mode_direct': 'Concise Analysis',
    'step1.mode_perspectives': 'Comprehensive Analysis',
    'step1.mode_debate': 'Dialectical Analysis',
    'step1.generate_btn': 'Start Analysis',

    // Dialectic config
    'dialectic.title': 'Dialectical Analysis Settings',
    'dialectic.confidence_label': 'Desired Confidence Level:',
    'dialectic.exploratory': 'Exploratory',
    'dialectic.exploratory_sub': '>60% consensus',
    'dialectic.balanced': 'Balanced',
    'dialectic.balanced_sub': '>70% consensus',
    'dialectic.conservative': 'Conservative',
    'dialectic.conservative_sub': '>80% consensus',
    'dialectic.note': '<strong class="text-violet-300">Note:</strong> If the initial consensus does not reach your desired level, the system will automatically suggest refinement prompts to improve convergence between AI models.',

    // Timeline
    'timeline.title': 'Analysis Phases',
    'timeline.step1': 'Query',
    'timeline.step2': 'Individual Analysis',
    'timeline.step3': 'Debate / Synthesis',
    'timeline.step4': 'Final Result',

    // Initial responses
    'initial.title': 'Individual Initial Responses',
    'initial.subtitle': 'Review individual perspectives before the dialectical process',

    // Consensus
    'consensus.title': 'Consensus Level Between Models',

    // Friction / dissidence
    'friction.title': 'Detected Dissidence Points',
    'friction.subtitle': 'The AI models have different perspectives on these aspects:',
    'friction.suggested_title': 'Suggested Refinement Prompts',
    'friction.suggested_subtitle': 'Select and edit a prompt to guide the refinement of the debate:',
    'friction.custom_label': 'Or write your own refinement prompt:',
    'friction.custom_placeholder': "E.g.: 'Focus on economic viability and reduce methodological differences...'",
    'friction.refine_btn': 'Refine Debate',

    // Step 2
    'step2.title': '2. Review Perspectives',
    'step2.subtitle': 'Evaluate the generated perspectives. Your selection will guide the final synthesis.',

    // Step 3 buttons
    'btn.key_points': 'Generate Key Points',
    'btn.strategic_synthesis': 'Develop Strategic Synthesis',

    // Final result
    'final.title': 'Final Strategic Synthesis',
    'tab.synthesis': 'Synthesis',
    'tab.arguments': 'Final Arguments',
    'tab.history': 'Analysis History',

    // Action buttons
    'btn.copy': 'Copy',
    'btn.export_txt': 'Export .txt',
    'btn.deepen': 'Deepen Analysis',
    'btn.download_txt': 'Download .txt',
    'btn.download_pdf': 'Download .pdf',

    // Card actions
    'action.include_synthesis': 'Include in Synthesis',
    'action.mark_counterpoint': 'Mark as Counterpoint',

    // Errors
    'error.timeout_title': 'Analysis Timed Out',
    'error.connection_title': 'Connection Error',

    // Labels
    'label.reasoning_trace': 'Reasoning Traceability:',
    'label.files_for_analysis': 'Files for analysis:',

    // Mode texts
    'mode.direct': '<strong>Concise Analysis:</strong> Generates a direct and concise response, ideal for quick data.',
    'mode.perspectives': '<strong>Comprehensive Analysis:</strong> Develops multiple perspectives and explores nuances, ideal for deep research.',
    'mode.debate': '<strong>Dialectical Analysis:</strong> Subjects perspectives to cross-examination to forge a more robust and validated synthesis.',

    // setLoading messages
    'loading.analyzing': 'Analyzing...',
    'loading.starting': 'Starting analysis...',
    'loading.docs': 'Processing documents and preparing RAG analysis...',
    'loading.debate': 'Starting dialectical debate...',
    'loading.perspectives': 'Generating perspectives...',

    // Alert messages
    'alert.no_instruction': 'Please provide an instruction to guide the debate.',
    'alert.no_question': 'Please enter a question or query to analyze the documents.',
    'alert.no_content': 'Please provide content for analysis, either by uploading files or pasting text.',
    'error.server_prefix': 'Server error:',
    'error.no_connection': 'Could not connect to the server.',
    'loading.retry': 'Retrying',

    // Dynamic JS strings
    'error.timeout_body': "This usually occurs with very large documents or in 'Dialectical Analysis' mode, as free-tier servers have a time limit. Try a shorter text fragment or a simpler analysis mode.",
    'debate.in_progress_title': 'Dialectical Analysis in Progress...',
    'debate.in_progress_subtitle': 'Perspectives are being cross-examined. This process may take longer.',
    'debate.in_progress_subtitle_refine': 'Perspectives are being cross-examined with your refinement context.',
    'debate.perspectives_completed': 'Individual Perspectives Completed',
    'debate.perspectives_review': 'Review the refined arguments from each model before the final result.',
    'debate.synthesis_title': 'Dialectical Analysis Synthesis',
    'debate.synthesis_title_refined': 'Refined Dialectical Analysis Synthesis',
    'creative.toggle': 'Creative Mode',
    'creative.desc': 'Perspectives in real tension',
    'creative.badge': '✦ Creative Mode',
    'creative.tooltip': 'Models explore multiple perspectives in genuine tension before responding, instead of answering from a single angle. Increases real diversity in the debate.',
    'refine.summary_title': 'Key Points Summary',
    'refine.report_title': 'Strategic Synthesis',
    'refine.report_refined_title': 'Refined Strategic Synthesis',
    'loading.debate_first': 'Running advanced multi-agent analysis with cross-validation... (3-4 minutes)',
    'loading.debate_refine': 'Refining perspectives with your specific guidance... (1-2 minutes)',
    'alert.write_direction': 'Please write a direction for the analysis',
    'consensus.semantic_label': 'Semantic analysis:',
    'consensus.strong_label': 'Strong consensus:',
    'consensus.moderate_label': 'Moderate consensus:',
    'consensus.divergence_label': 'Divergences:',
    'consensus.breakdown_btn': 'View detailed breakdown',
    'consensus.basic_label': 'Basic keyword analysis',
    'consensus.matches': 'matches out of',
    'consensus.concepts': 'concepts',
    'prompt.resolve_diff': 'Resolve Conceptual Differences',
    'prompt.balance_depth': 'Balance Depth',
    'prompt.maximize_consensus': 'Maximize Consensus',
    'exploration.options_subtitle': 'You can direct the analysis toward specific aspects:',
    'exploration.btn.alternative': 'Explore contrarian perspectives',
    'exploration.btn.deeper': 'Explore specific aspects in depth',
    'exploration.btn.skeptical': 'Skeptical analysis',
    'exploration.btn.custom': 'Custom prompt',
    'exploration.custom_label': 'Direct the analysis:',
    'exploration.custom_placeholder': "E.g.: 'Focus on financial risks' or 'Analyze from end-user perspective'",
    'exploration.execute_btn': 'Execute Directed Analysis',

    // Acknowledgments modal
    'ack.title': 'Acknowledgments',
    'ack.intro': 'CRISALIA was made possible thanks to the collaboration and support of:',
    'ack.team_label': 'Development Team',
    'ack.institutions_label': 'Institutions',
    'ack.footer': 'Their contribution was fundamental to the development of this technology.',
    'footer.contact': 'Contact',
    'export.title': 'Crisalia - Analysis Report',
    'export.initial_query': 'INITIAL QUERY:',
    'export.trace': 'ANALYSIS TRACE (History):',
    'export.iteration': 'Iteration',
    'export.query': 'Query:',
    'export.response': 'Response:',
    'export.final_result': 'FINAL RESULT:',
  }
};

function t(key) {
  const lang = localStorage.getItem('crisalia_lang') || 'en';
  return (TRANSLATIONS[lang] && TRANSLATIONS[lang][key]) || (TRANSLATIONS['en'][key]) || key;
}

function applyTranslations() {
  const lang = localStorage.getItem('crisalia_lang') || 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const translation = t(key);
    if (translation) {
      if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        el.placeholder = translation;
      } else {
        el.innerHTML = translation;
      }
    }
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const key = el.getAttribute('data-i18n-title');
    const translation = t(key);
    if (translation) el.title = translation;
  });
  // Update html lang attribute
  document.documentElement.lang = lang;
  // Update language switcher buttons
  document.querySelectorAll('.lang-switcher-btn').forEach(btn => {
    const btnLang = btn.getAttribute('data-lang');
    btn.classList.toggle('active-lang', btnLang === lang);
  });
}

function setLanguage(lang) {
  localStorage.setItem('crisalia_lang', lang);
  applyTranslations();
}

document.addEventListener('DOMContentLoaded', applyTranslations);
