// ==============================================================================
// DIALECTIC ENHANCEMENTS - Funcionalidades Dialécticas Completas (v6.3 CORREGIDO)
// ==============================================================================

// Estado global extendido para funcionalidades dialécticas
let dialecticState = {
    confidenceLevel: 'balanced', // exploratory, balanced, conservative
    consensusThresholds: {
        exploratory: 60,
        balanced: 70,
        conservative: 80
    },
    realDissidences: [],
    suggestedPrompts: [],
    initialResponsesShown: false,
    consensusHistory: [],
    lastConsensusScore: 0
};

// FUNCIÓN AUXILIAR: getKeywords (definida primero para evitar errores)
function getKeywords(text) {
    if (!text) return new Set();
    // Definir stopWords local si no existe globalmente
    const stopWords = window.stopWords || new Set(['the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'on', 'at', 'from', 'by', 'in', 'of', 'to', 'is', 'are', 'was', 'were', 'it', 'this', 'that', 'they', 'he', 'she', 'we', 'i', 'you']);
    
    return new Set(
        text.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopWords.has(word))
    );
}

// FUNCIÓN PRINCIPAL: Análisis semántico con Claude como mediador (CORREGIDA)
async function calculateAndDisplayConsensusEnhanced() {
    console.log('=== INICIANDO ANÁLISIS SEMÁNTICO ===');

    // Plan free: skip consensus display
    if (window.PLAN === 'free') {
        const consensusContainer = document.getElementById('consensus-container');
        if (consensusContainer) consensusContainer.classList.add('hidden');
        return 0;
    }

    // Verificar que state existe
    if (!window.state || !window.state.initial_responses) {
        console.error('State no disponible');
        return calculateTraditionalConsensus();
    }
    
    console.log('API_BASE_URL:', typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'NO DEFINIDO');
    console.log('State inicial_responses:', window.state.initial_responses);
    
    const responses = {
        gemini: window.state.initial_responses.gemini.content || '',
        deepseek: window.state.initial_responses.deepseek.content || '',
        claude: window.state.initial_responses.claude.content || ''
    };
    
    console.log('Responses para análisis:', responses);
    console.log('Longitudes de respuestas:', {
        gemini: responses.gemini.length,
        deepseek: responses.deepseek.length,
        claude: responses.claude.length
    });
    
    // Verificar que tenemos contenido
    const validResponses = Object.values(responses).filter(r => r && r.trim().length > 0);
    console.log('Respuestas válidas:', validResponses.length);
    
    if (validResponses.length < 2) {
        console.log('No hay suficientes respuestas válidas');
        const consensusContainer = document.getElementById('consensus-container');
        if (consensusContainer) consensusContainer.classList.add('hidden');
        return 0;
    }
    
    try {
        console.log('Llamando a análisis semántico...');
        
        // Verificar que API_BASE_URL esté definido
        if (typeof API_BASE_URL === 'undefined') {
            throw new Error('API_BASE_URL no está definido');
        }
        
        const apiUrl = `${API_BASE_URL}/api/semantic-consensus`;
        console.log('URL completa:', apiUrl);
        
        // Llamar al análisis semántico del backend usando Claude
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ responses: responses })
        });
        
        console.log('Response status:', response.status);
        console.log('Response OK:', response.ok);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const analysis = await response.json();
        console.log('Análisis semántico recibido:', analysis);
        
        const score = analysis.consensus_score || 0;
        console.log('Score semántico final:', score);
        
        // Actualizar UI con análisis detallado
        updateSemanticConsensusDisplay(score, analysis);
        
        // Detectar consenso estancado
        dialecticState.consensusHistory.push(score);
        console.log('Historial de consenso:', dialecticState.consensusHistory);
        
        if (dialecticState.consensusHistory.length >= 3) {
            const last3 = dialecticState.consensusHistory.slice(-3);
            const isStagnant = Math.max(...last3) - Math.min(...last3) <= 5;
            console.log('Últimos 3 consensos:', last3, 'Estancado:', isStagnant);
            
            const selectedLevel = document.querySelector('input[name="confidenceLevel"]:checked')?.value || 'balanced';
            const threshold = dialecticState.consensusThresholds[selectedLevel];
            console.log('Nivel seleccionado:', selectedLevel, 'Threshold:', threshold);
            
            if (isStagnant && score < threshold) {
                console.log('Mostrando advertencia de estancamiento');
                showConsensusStagnationWarning(score);
            }
        }
        
        // Usar análisis avanzado si estamos en modo debate
        const selectedMode = document.querySelector('input[name="analysisMode"]:checked')?.value;
        console.log('Modo seleccionado:', selectedMode);
        
        if (selectedMode === 'debate') {
            console.log('Ejecutando análisis avanzado de bajo consenso');
            handleAdvancedLowConsensus(window.state.initial_responses, score);
        } else if (score < 40) {
            console.log('Ejecutando manejo de bajo consenso básico');
            handleLowConsensus();
        }
        
        console.log('Análisis semántico completado exitosamente');
        return score;
        
    } catch (error) {
        console.error('Error en análisis semántico:', error);
        console.error('Error stack:', error.stack);
        console.log('Fallback a método tradicional');
        
        // Fallback al método tradicional si falla el semántico
        const fallbackScore = calculateTraditionalConsensus();
        console.log('Score fallback:', fallbackScore);
        return fallbackScore;
    }
}

// FUNCIÓN AUXILIAR: Mostrar análisis semántico detallado
function updateSemanticConsensusDisplay(score, analysis) {
    const consensusContainer = document.getElementById('consensus-container');
    const consensusScore = document.getElementById('consensus-score');
    const consensusBar = document.getElementById('consensus-bar');
    const consensusStatus = document.getElementById('consensus-status');
    
    if (consensusContainer) consensusContainer.classList.remove('hidden');
    if (consensusScore) consensusScore.textContent = `${score}%`;
    if (consensusBar) {
        setTimeout(() => { consensusBar.style.width = `${score}%`; }, 100);
    }
    
    // Mostrar análisis detallado
    if (consensusStatus && analysis) {
        const strongCount = analysis.strong_consensus ? analysis.strong_consensus.length : 0;
        const moderateCount = analysis.moderate_consensus ? analysis.moderate_consensus.length : 0;
        
        consensusStatus.innerHTML = `
            <div class="text-sm text-gray-300 mt-2">
                <div class="mb-1"><strong>${t('consensus.semantic_label')}</strong> ${strongCount} ${t('consensus.strong_label').replace(':','')} , ${moderateCount} ${t('consensus.moderate_label').replace(':','')}</div>
                ${analysis.explanation ? `<div class="text-xs text-gray-400 mb-2">${analysis.explanation}</div>` : ''}
                <details class="cursor-pointer">
                    <summary class="text-violet-400 hover:text-violet-300">${t('consensus.breakdown_btn')}</summary>
                    <div class="mt-2 text-xs space-y-1 bg-gray-900/50 p-2 rounded">
                        ${strongCount > 0 ? `<div><strong class="text-green-400">${t('consensus.strong_label')}</strong> ${analysis.strong_consensus.join(', ')}</div>` : ''}
                        ${moderateCount > 0 ? `<div><strong class="text-yellow-400">${t('consensus.moderate_label')}</strong> ${analysis.moderate_consensus.join(', ')}</div>` : ''}
                        ${analysis.divergence_areas && analysis.divergence_areas.length > 0 ? `<div><strong class="text-red-400">${t('consensus.divergence_label')}</strong> ${analysis.divergence_areas.join(', ')}</div>` : ''}
                    </div>
                </details>
            </div>
        `;
    }
}

// FUNCIÓN FALLBACK: Cálculo tradicional básico
function calculateTraditionalConsensus() {
    console.log('Usando método tradicional de consenso por keywords');
    
    if (!window.state || !window.state.initial_responses) {
        console.error('State no disponible para consenso tradicional');
        return 0;
    }
    
    const keywordsGemini = getKeywords(window.state.initial_responses.gemini.content);
    const keywordsDeepseek = getKeywords(window.state.initial_responses.deepseek.content);
    const keywordsClaude = getKeywords(window.state.initial_responses.claude.content);
    
    const allSets = [keywordsGemini, keywordsDeepseek, keywordsClaude].filter(s => s.size > 0);
    if (allSets.length < 2) {
        const consensusContainer = document.getElementById('consensus-container');
        if (consensusContainer) consensusContainer.classList.add('hidden');
        return 0;
    }
    
    const intersection = new Set([...allSets[0]].filter(word => allSets.every(set => set.has(word))));
    const union = new Set([...keywordsGemini, ...keywordsDeepseek, ...keywordsClaude]);
    const score = union.size === 0 ? 0 : Math.round((intersection.size / union.size) * 100);
    
    // Actualizar UI básica
    const consensusContainer = document.getElementById('consensus-container');
    const consensusScore = document.getElementById('consensus-score');
    const consensusBar = document.getElementById('consensus-bar');
    const consensusStatus = document.getElementById('consensus-status');
    
    if (consensusContainer) consensusContainer.classList.remove('hidden');
    if (consensusScore) consensusScore.textContent = `${score}%`;
    if (consensusBar) {
        setTimeout(() => { consensusBar.style.width = `${score}%`; }, 100);
    }
    if (consensusStatus) {
        consensusStatus.innerHTML = `<div class="text-sm text-gray-400 mt-2">${t('consensus.basic_label')} (${intersection.size} ${t('consensus.matches')} ${union.size} ${t('consensus.concepts')})</div>`;
    }
    
    return score;
}

// Algoritmo mejorado de análisis de disidencias
function analyzeRealDissidences(responses) {
    const dissidences = [];
    const models = Object.keys(responses);
    
    // Análisis semántico básico
    const concepts = {
        scope: ['alcance', 'ámbito', 'cobertura', 'extensión', 'rango'],
        methodology: ['metodología', 'enfoque', 'aproximación', 'estrategia', 'método'],
        priority: ['prioridad', 'importancia', 'relevancia', 'peso', 'énfasis'],
        timeline: ['tiempo', 'plazo', 'cronograma', 'período', 'duración'],
        resources: ['recursos', 'presupuesto', 'costo', 'inversión', 'capital'],
        risk: ['riesgo', 'incertidumbre', 'peligro', 'amenaza', 'vulnerabilidad']
    };
    
    // Detectar diferencias conceptuales
    Object.entries(concepts).forEach(([category, keywords]) => {
        const modelMentions = {};
        models.forEach(model => {
            const content = responses[model].content.toLowerCase();
            modelMentions[model] = keywords.some(keyword => content.includes(keyword));
        });
        
        const mentioning = models.filter(model => modelMentions[model]);
        const notMentioning = models.filter(model => !modelMentions[model]);
        
        if (mentioning.length > 0 && notMentioning.length > 0) {
            dissidences.push({
                category: category,
                description: `Algunos modelos abordan aspectos de ${category}, mientras otros no los consideran`,
                mentioning: mentioning,
                notMentioning: notMentioning,
                severity: mentioning.length === 1 ? 'high' : 'medium'
            });
        }
    });
    
    // Análisis de longitud y profundidad
    const lengths = models.map(model => responses[model].content.length);
    const avgLength = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const lengthVariation = Math.max(...lengths) / Math.min(...lengths);
    
    if (lengthVariation > 2) {
        dissidences.push({
            category: 'depth',
            description: 'Diferencias significativas en la profundidad del análisis entre modelos',
            details: `Variación de ${lengthVariation.toFixed(1)}x en extensión de respuestas`,
            severity: 'medium'
        });
    }
    
    return dissidences;
}

// Generador de prompts contextuales
function generateContextualPrompts(originalPrompt, responses, dissidences) {
    const prompts = [];
    
    // Prompt basado en disidencias principales
    if (dissidences.length > 0) {
        const mainCategories = dissidences.map(d => d.category).slice(0, 2);
        prompts.push({
            title: t('prompt.resolve_diff'),
            prompt: `Enfócate en reconciliar las diferencias sobre ${mainCategories.join(' y ')} mencionadas en las respuestas anteriores. Proporciona una perspectiva unificada que integre los mejores aspectos de cada enfoque.`,
            type: "integration"
        });
    }
    
    // Prompt de profundización
    const responseEntries = Object.entries(responses);
    if (responseEntries.length >= 2) {
        const shortestResponse = responseEntries.reduce((a, b) => 
            a[1].content.length < b[1].content.length ? a : b);
        const longestResponse = responseEntries.reduce((a, b) => 
            a[1].content.length > b[1].content.length ? a : b);
        
        if (longestResponse[1].content.length > shortestResponse[1].content.length * 1.5) {
            prompts.push({
                title: t('prompt.balance_depth'),
                prompt: `Toma el nivel de detalle de ${longestResponse[0]} pero mantén la concisión de ${shortestResponse[0]}. Proporciona un análisis equilibrado que sea tanto completo como accesible.`,
                type: "balance"
            });
        }
    }
    
    // Prompt de consenso forzado
    prompts.push({
        title: t('prompt.maximize_consensus'),
        prompt: `Revisa las respuestas anteriores e identifica los puntos donde TODOS los modelos coinciden. Construye una nueva respuesta basada exclusivamente en estos consensos, agregando solo información que complemente sin contradecir.`,
        type: "consensus"
    });
    
    return prompts;
}

// Función para mostrar respuestas iniciales en modo dialéctico
function displayInitialResponses(responses) {
    if (dialecticState.initialResponsesShown) return;
    
    const container = document.getElementById('initial-responses-container');
    const grid = document.getElementById('initial-responses-grid');
    
    if (!container || !grid) return;
    
    let html = '';
    Object.entries(responses).forEach(([model, data]) => {
        const content = data.content || data;
        html += `
            <div class="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h4 class="text-sm font-semibold text-white mb-2 capitalize flex items-center">
                    <span class="w-3 h-3 rounded-full mr-2 ${model === 'gemini' ? 'bg-[#4285F4]' : model === 'deepseek' ? 'bg-[#34A853]' : 'bg-[#D48105]'}"></span>
                    ${model}
                </h4>
                <div class="text-xs text-gray-400 max-h-32 overflow-y-auto">
                    ${content.substring(0, 300) + (content.length > 300 ? '...' : '')}
                </div>
            </div>
        `;
    });
    
    grid.innerHTML = html;
    container.classList.remove('hidden');
    dialecticState.initialResponsesShown = true;
}

// Función para mostrar disidencias reales - CORREGIDA
function displayRealDissidences(dissidences, consensusScore = 0) {
    const container = document.getElementById('dissidence-analysis');
    if (!container) return;
    
    let html = '';
    
    // MOSTRAR CONTROLES SIEMPRE, independiente del consenso
    if (dissidences.length === 0 || consensusScore > 99) {
        html = `
            <div class="dissidence-item p-4 bg-blue-900/20 border border-blue-700/50 rounded-lg">
                <div class="flex items-center mb-2">
                    <svg class="w-5 h-5 text-blue-400 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span class="text-blue-300 font-medium">Consenso: ${consensusScore}% - Opciones disponibles</span>
                </div>
                <p class="text-sm text-blue-200 mb-3">${t('exploration.options_subtitle')}</p>

                <div class="space-y-2">
                    <button class="exploration-btn w-full text-left px-3 py-2 text-sm bg-blue-800/30 hover:bg-blue-700/50 rounded border border-blue-600/50 transition-colors"
                            data-exploration="alternative">
                        ${t('exploration.btn.alternative')}
                    </button>
                    <button class="exploration-btn w-full text-left px-3 py-2 text-sm bg-purple-800/30 hover:bg-purple-700/50 rounded border border-purple-600/50 transition-colors"
                            data-exploration="deeper">
                        ${t('exploration.btn.deeper')}
                    </button>
                    <button class="exploration-btn w-full text-left px-3 py-2 text-sm bg-amber-800/30 hover:bg-amber-700/50 rounded border border-amber-600/50 transition-colors"
                            data-exploration="skeptical">
                        ${t('exploration.btn.skeptical')}
                    </button>
                    <button class="exploration-btn w-full text-left px-3 py-2 text-sm bg-green-800/30 hover:bg-green-700/50 rounded border border-green-600/50 transition-colors"
                            data-exploration="custom">
                        ${t('exploration.btn.custom')}
                    </button>
                </div>

                <!-- Campo para prompt personalizado (inicialmente oculto) -->
                <div id="custom-exploration-area" class="hidden mt-4">
                    <label class="text-sm font-medium text-gray-300 mb-2 block">${t('exploration.custom_label')}</label>
                    <textarea id="custom-exploration-prompt" placeholder="${t('exploration.custom_placeholder')}"
                              class="w-full h-20 p-3 bg-gray-900 border border-gray-700 text-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 text-sm"></textarea>
                    <button id="execute-custom-exploration" class="mt-2 px-4 py-2 bg-violet-600 text-white rounded hover:bg-violet-700 transition-colors text-sm">
                        ${t('exploration.execute_btn')}
                    </button>
                </div>
            </div>
        `;
    } else {
        // Mostrar disidencias reales cuando existen
        dissidences.forEach((dissidence, index) => {
            const severityColors = {
                high: 'border-red-500/50 bg-red-900/20',
                medium: 'border-amber-500/50 bg-amber-900/20', 
                low: 'border-blue-500/50 bg-blue-900/20'
            };
            
            html += `
                <div class="dissidence-item p-4 ${severityColors[dissidence.severity]} rounded-lg" data-dissidence="${index}">
                    <div class="flex items-start justify-between mb-2">
                        <h4 class="text-sm font-medium text-white capitalize">${dissidence.category.replace('_', ' ')}</h4>
                        <div class="flex space-x-2">
                            <button class="dissidence-action text-xs px-2 py-1 rounded bg-green-700/30 text-green-300 hover:bg-green-600/40" data-action="include" data-index="${index}">Incluir</button>
                            <button class="dissidence-action text-xs px-2 py-1 rounded bg-red-700/30 text-red-300 hover:bg-red-600/40" data-action="exclude" data-index="${index}">Excluir</button>
                            <button class="dissidence-action text-xs px-2 py-1 rounded bg-gray-700/30 text-gray-300 hover:bg-gray-600/40" data-action="ignore" data-index="${index}">Ignorar</button>
                        </div>
                    </div>
                    <p class="text-sm text-gray-300">${dissidence.description}</p>
                    ${dissidence.details ? `<p class="text-xs text-gray-400 mt-1">${dissidence.details}</p>` : ''}
                </div>
            `;
        });
    }
    
    container.innerHTML = html;
    
    // Event listeners
    container.querySelectorAll('.exploration-btn').forEach(btn => {
        btn.addEventListener('click', handleExplorationClick);
    });
    
    container.querySelectorAll('.dissidence-action').forEach(btn => {
        btn.addEventListener('click', handleDissidenceAction);
    });
    
    // Listener para custom exploration
    const customBtn = document.getElementById('execute-custom-exploration');
    if (customBtn) {
        customBtn.addEventListener('click', handleCustomExplorationClick);
    }
    
    console.log('displayRealDissidences completado con event listeners');
}

// Función para mostrar prompts sugeridos
function displaySuggestedPrompts(prompts) {
    const container = document.getElementById('suggested-prompts');
    if (!container) return;
    
    let html = '';
    prompts.forEach((prompt, index) => {
        html += `
            <div class="suggested-prompt p-4 rounded-lg cursor-pointer" data-index="${index}">
                <div class="flex items-center justify-between mb-2">
                    <h4 class="text-sm font-medium text-white">${prompt.title}</h4>
                    <span class="text-xs px-2 py-1 rounded-full bg-green-700/30 text-green-300">${prompt.type}</span>
                </div>
                <p class="text-sm text-gray-300">${prompt.prompt}</p>
                <button class="use-prompt-btn mt-2 text-xs px-3 py-1 rounded bg-violet-600/30 text-violet-300 hover:bg-violet-500/40" data-index="${index}">Usar este prompt</button>
            </div>
        `;
    });
    
    container.innerHTML = html;
    
    // Event listeners para prompts sugeridos
    container.querySelectorAll('.use-prompt-btn').forEach(btn => {
        btn.addEventListener('click', handleSuggestedPromptSelection);
    });
}

// Manejador de acciones de disidencia
function handleDissidenceAction(event) {
    const action = event.target.dataset.action;
    const index = parseInt(event.target.dataset.index);
    const dissidenceItem = event.target.closest('.dissidence-item');
    
    // Remover clases previas
    dissidenceItem.classList.remove('opacity-60', 'ring-2', 'ring-green-500', 'ring-red-500');
    
    // Aplicar nuevo estado visual
    switch(action) {
        case 'include':
            dissidenceItem.classList.add('ring-2', 'ring-green-500');
            dialecticState.realDissidences[index].userAction = 'include';
            break;
        case 'exclude':
            dissidenceItem.classList.add('ring-2', 'ring-red-500');
            dialecticState.realDissidences[index].userAction = 'exclude';
            break;
        case 'ignore':
            dissidenceItem.classList.add('opacity-60');
            dialecticState.realDissidences[index].userAction = 'ignore';
            break;
    }
    
    // Actualizar botones
    dissidenceItem.querySelectorAll('.dissidence-action').forEach(btn => {
        btn.classList.remove('bg-violet-600', 'text-white');
        btn.classList.add('bg-gray-700/30', 'text-gray-300');
    });
    
    event.target.classList.remove('bg-gray-700/30', 'text-gray-300');
    event.target.classList.add('bg-violet-600', 'text-white');
}

// Manejador de selección de prompts sugeridos
function handleSuggestedPromptSelection(event) {
    const index = parseInt(event.target.dataset.index);
    const selectedPrompt = dialecticState.suggestedPrompts[index];
    
    // Poner el prompt en el textarea
    const customPromptArea = document.getElementById('custom-refinement-prompt');
    if (customPromptArea) {
        customPromptArea.value = selectedPrompt.prompt;
        customPromptArea.focus();
    }
    
    // Highlight visual del prompt seleccionado
    document.querySelectorAll('.suggested-prompt').forEach(p => {
        p.classList.remove('ring-2', 'ring-violet-500');
    });
    event.target.closest('.suggested-prompt').classList.add('ring-2', 'ring-violet-500');
}

// Función para mostrar advertencia de consenso estancado
function showConsensusStagnationWarning(currentScore) {
    const statusElement = document.getElementById('consensus-status');
    const consensusContainer = document.getElementById('consensus-container');
    
    if (statusElement) {
        statusElement.innerHTML = `
            <div class="flex items-center justify-center mt-3 p-3 bg-amber-900/20 border border-amber-500/30 rounded-lg">
                <svg class="w-5 h-5 text-amber-400 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L3.35 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span class="text-amber-300 text-sm font-medium">Consenso estancado (${currentScore}%) - Considera finalizar o cambiar enfoque</span>
            </div>
        `;
    }
    
    // Agregar botón de síntesis forzada si no existe
    if (consensusContainer && !document.getElementById('force-synthesis-btn')) {
        const forceButton = document.createElement('button');
        forceButton.id = 'force-synthesis-btn';
        forceButton.className = 'mt-3 w-full px-4 py-2 bg-amber-600 text-white font-medium rounded-lg hover:bg-amber-700 transition-colors text-sm';
        forceButton.textContent = 'Sintetizar con Consenso Actual';
        forceButton.addEventListener('click', handleForcedSynthesis);
        
        consensusContainer.appendChild(forceButton);
    }
}

// Función para síntesis forzada
async function handleForcedSynthesis() {
    // Ocultar advertencias y botones de refinamiento
    const frictionContainer = document.getElementById('friction-container');
    const statusElement = document.getElementById('consensus-status');
    
    if (frictionContainer) frictionContainer.classList.add('hidden');
    if (statusElement) statusElement.innerHTML = '<span class="text-blue-300 text-sm">Sintetizando con consenso actual...</span>';
    
    // Agregar contexto especial para síntesis forzada
    const forcedContext = {
        userRefinementPrompt: "Crea una síntesis final basada en los consensos existentes, aceptando las diferencias restantes como perspectivas complementarias.",
        confidenceLevel: dialecticState.confidenceLevel,
        targetConsensus: 'forced',
        forcedSynthesis: true
    };
    
    // Agregar al historial y ejecutar síntesis
    if (window.state) {
        window.state.history.push({ 
            prompt: "Síntesis forzada por usuario",
            dissidenceContext: forcedContext,
            synthesis: "" 
        });
    }
    
    // Llamar al debate con contexto de síntesis forzada
    handleDebateWithContext(forcedContext);
}

// Función principal para manejar bajo consenso con nuevas funcionalidades - CORREGIDA
function handleAdvancedLowConsensus(responses, consensusScore, forceShow = false) {
    if (window.PLAN === 'free') return; // Free plan: no dissidence controls shown
    const selectedLevelInput = document.querySelector('input[name="confidenceLevel"]:checked');
    const selectedLevel = selectedLevelInput ? selectedLevelInput.value : 'balanced';
    const threshold = dialecticState.consensusThresholds[selectedLevel];
    
    console.log(`handleAdvancedLowConsensus: score=${consensusScore}, threshold=${threshold}, forceShow=${forceShow}`);
    
    // MOSTRAR SIEMPRE en modo debate, independientemente del consenso
    if (!forceShow && consensusScore >= threshold) {
        console.log('Consenso suficiente, pero mostrando opciones en modo debate');
        // NO hacer return aquí en modo debate
    }
    
    // Mostrar respuestas iniciales primero
    displayInitialResponses(responses);
    
    // Analizar disidencias reales
    dialecticState.realDissidences = analyzeRealDissidences(responses);
    
    // Generar prompts contextuales
    dialecticState.suggestedPrompts = generateContextualPrompts(
        window.state ? window.state.prompt : '', 
        responses, 
        dialecticState.realDissidences
    );
    
    // Mostrar interfaz mejorada
    displayRealDissidences(dialecticState.realDissidences, consensusScore);
    displaySuggestedPrompts(dialecticState.suggestedPrompts);
    
    // Actualizar mensaje de consenso
    const statusElement = document.getElementById('consensus-status');
    
    if (statusElement && consensusScore < threshold) {
    // Prevenir duplicación temporal
    if (!statusElement.querySelector('.text-amber-300')) {
        const currentHTML = statusElement.innerHTML;
        statusElement.innerHTML = currentHTML + `<div class="mt-2 text-amber-300 text-sm">Consenso insuficiente para nivel <strong>${selectedLevel}</strong> (objetivo: ${threshold}%)</div>`;
    }
}
    
    const frictionContainer = document.getElementById('friction-container');
    if (frictionContainer) {
        frictionContainer.classList.remove('hidden');
        console.log('friction-container mostrado');
    }
}

// Función mejorada para el refinamiento del debate
async function handleAdvancedRefineDebate() {
    const customPrompt = document.getElementById('custom-refinement-prompt').value.trim();
    
    if (!customPrompt) {
        alert('Por favor, selecciona un prompt sugerido o escribe uno personalizado.');
        return;
    }
    
    // Preparar contexto de disidencias para el backend
    const dissidenceContext = {
        includedDissidences: dialecticState.realDissidences.filter(d => d.userAction === 'include'),
        excludedDissidences: dialecticState.realDissidences.filter(d => d.userAction === 'exclude'),
        userRefinementPrompt: customPrompt,
        confidenceLevel: dialecticState.confidenceLevel,
        targetConsensus: dialecticState.consensusThresholds[dialecticState.confidenceLevel]
    };
    
    // Agregar al historial
    if (window.state) {
        window.state.history.push({ 
            prompt: "Refinamiento dirigido: " + customPrompt,
            dissidenceContext: dissidenceContext,
            synthesis: "" 
        });
    }
    
    // Ocultar interfaz de disidencias
    const frictionContainer = document.getElementById('friction-container');
    if (frictionContainer) frictionContainer.classList.add('hidden');
    
    const customPromptInput = document.getElementById('custom-refinement-prompt');
    if (customPromptInput) customPromptInput.value = '';
    
    // Reiniciar estado
    dialecticState.initialResponsesShown = false;
    
    // Llamar al debate refinado
    handleDebateWithContext(dissidenceContext);
}

// Versión extendida del debate que incluye contexto de disidencias
// ==============================================================================
// FUNCIÓN CORREGIDA Y FINAL (para reemplazar la original)
// ==============================================================================
async function handleDebateWithContext(dissidenceContext = null) {
    if (!window.updateTimeline || !window.state) {
        console.error('Funciones globales no disponibles');
        return;
    }
    
    window.updateTimeline(2);
    window.state.isStreaming = false;
    
    const dom = {
        step2: document.getElementById('step2'),
        step3: document.getElementById('step3'),
        finalResultContainer: document.getElementById('final-result-container'),
        responsesGrid: document.getElementById('responses-grid'),
        step2Header: document.getElementById('step2-header')
    };
    
    if (dom.step2) dom.step2.classList.remove('hidden'); 
    if (dom.step3) dom.step3.classList.add('hidden'); 
    if (dom.finalResultContainer) dom.finalResultContainer.classList.add('hidden');
    
    const loadingSpinner = `<div class="animate-pulse space-y-2"><div class="h-4 bg-gray-700 rounded"></div><div class="h-4 bg-gray-700 rounded w-5/6"></div></div>`;
    if (dom.responsesGrid && window.templates) {
        dom.responsesGrid.innerHTML = Object.keys(window.state.initial_responses).map(model => 
            window.templates.responseCard(model, loadingSpinner)).join('');
    }
    
    if (dom.step2Header) {
       // Detectar si es primera vez o refinamiento
const isRefinement = dissidenceContext && dissidenceContext.userRefinementPrompt;
const headerText = isRefinement ?
    t('debate.in_progress_subtitle_refine') :
    t('debate.in_progress_subtitle');

dom.step2Header.innerHTML = `<h2 class="text-2xl font-semibold mb-2 text-white">${t('debate.in_progress_title')}</h2><p class="text-gray-400">${headerText}</p>`;
    }
    
    if (window.setLoading) window.setLoading(true, "Generando respuestas iniciales de cada modelo...");
    
    try {
        if (window.analysisTimeout) clearTimeout(window.analysisTimeout);
        if (window.handleTimeoutError) {
            window.analysisTimeout = setTimeout(window.handleTimeoutError, 360000);
        }
        
       const requestBody = {
    prompt: window.state.prompt,
    history: window.state.history,
    // FIX v6.2.2: Incluir initial_responses para que el backend no regenere
    // respuestas desde cero al ejecutar Maximizar Consenso o refinamientos.
    initial_responses: window.state.initial_responses
        ? Object.fromEntries(
            Object.entries(window.state.initial_responses).map(
                ([k, v]) => [k, typeof v === 'object' && v !== null ? (v.content || '') : String(v)]
            )
          )
        : null
};

if (dissidenceContext) {
    requestBody.dissidenceContext = dissidenceContext;
}

if (window.CREATIVE_MODE) {
    requestBody.creative_mode = true;
}

requestBody.lang = localStorage.getItem('crisalia_lang') || 'en';

const isFirstIteration = !dissidenceContext || !dissidenceContext.userRefinementPrompt;
const progressMessage = isFirstIteration ?
    t('loading.debate_first') :
    t('loading.debate_refine');

console.log('Iniciando análisis dialéctico completo...');
if (window.setLoading) window.setLoading(true, progressMessage);

const response = await window.fetchWithRetries(`${window.API_BASE_URL}/api/debate`, {
    method: 'POST', 
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
});
        
        window.updateTimeline(3);
        if (window.analysisTimeout) clearTimeout(window.analysisTimeout);
        const result = await response.json();
        
        Object.keys(result.revised).forEach(model => {
            window.state.initial_responses[model].content = result.revised[model];
            const contentDiv = document.getElementById(`content-${model}`);
            if (contentDiv && window.marked) {
                contentDiv.innerHTML = window.marked.parse(result.revised[model]);
                const actionsDiv = document.getElementById(`actions-${model}`);
                if(actionsDiv) actionsDiv.classList.add('hidden');
            }
        });
        // Plan free: ocultar las tarjetas individuales después de llenarlas
        if (window.PLAN === 'free' && dom.responsesGrid && window.lockedResponsesHTML) {
            dom.responsesGrid.innerHTML = window.lockedResponsesHTML();
        }
        if (dom.step2Header) {
            dom.step2Header.innerHTML = `<h2 class="text-2xl font-semibold mb-2 text-white">${t('debate.perspectives_completed')}</h2><p class="text-gray-400">${t('debate.perspectives_review')}</p>`;
        }
        window.state.synthesis = result.synthesis;
        window.state.history.push({ prompt: window.state.prompt, synthesis: result.synthesis });
        
        if (dom.finalResultContainer && window.templates && window.highlightSynthesisSources) {
            dom.finalResultContainer.classList.remove('hidden');
            const highlightedHTML = await window.highlightSynthesisSources(window.state.synthesis);
            dom.finalResultContainer.innerHTML = window.templates.finalResultCard(t('debate.synthesis_title_refined'), highlightedHTML);
            dom.finalResultContainer.scrollIntoView({behavior: 'smooth', block: 'start' });
        }
        
        if (window.attachFinalCardListeners) window.attachFinalCardListeners();
        window.updateTimeline(4);
        
        // ==========================================================
        // ESTE ES EL BLOQUE AGREGADO QUE ACTIVA EL BUCLE DE REFINAMIENTO
        // ==========================================================
        const newConsensusScore = await calculateAndDisplayConsensusEnhanced();
        handleAdvancedLowConsensus(window.state.initial_responses, newConsensusScore, true); // forceShow=true para mostrar siempre
        
    } catch (error) {
        if (window.analysisTimeout) clearTimeout(window.analysisTimeout);
        if (dom.responsesGrid && window.templates) {
            dom.responsesGrid.innerHTML = window.templates.errorCard(error);
        }
    } finally {
        if (window.setLoading) window.setLoading(false);
    }
}

// Manejadores de exploración - NUEVAS FUNCIONES AGREGADAS
function handleExplorationClick(event) {
    const explorationType = event.target.dataset.exploration;
    console.log('Click en exploración:', explorationType);
    
    if (explorationType === 'custom') {
        const customArea = document.getElementById('custom-exploration-area');
        if (customArea) {
            customArea.classList.remove('hidden');
            document.getElementById('custom-exploration-prompt').focus();
        }
        return;
    }
    
    const originalPrompt = window.state ? window.state.prompt : '';
    const explorationPrompt = generateExplorationPrompt(explorationType, originalPrompt);
    
    if (explorationPrompt) {
        executeRefinement(explorationPrompt);
    }
}

function handleCustomExplorationClick() {
    const customPrompt = document.getElementById('custom-exploration-prompt').value.trim();
    if (!customPrompt) {
        alert(t('alert.write_direction'));
        return;
    }
    
    executeRefinement(customPrompt);
}

function generateExplorationPrompt(type, originalPrompt) {
    const prompts = {
        alternative: `Desafía el consenso actual sobre: "${originalPrompt}". Explora perspectivas contrarias, asunciones cuestionables, o enfoques completamente diferentes.`,
        deeper: `Profundiza en aspectos específicos de: "${originalPrompt}". Analiza implicaciones, consecuencias a largo plazo, y factores subyacentes.`,
        skeptical: `Adopta perspectiva escéptica sobre: "${originalPrompt}". ¿Qué podría estar mal? ¿Qué riesgos no se consideraron? ¿Qué evidencia contradictoria existe?`
    };
    
    return prompts[type] || null;
}

function executeRefinement(promptText) {
    const customRefinementPrompt = document.getElementById('custom-refinement-prompt');
    if (customRefinementPrompt) {
        customRefinementPrompt.value = promptText;
        
        // Llamar directamente a handleAdvancedRefineDebate
        setTimeout(() => {
            handleAdvancedRefineDebate();
        }, 100);
    }
}

// Inicialización de funcionalidades dialécticas mejoradas
function initDialecticEnhancements() {
    // Mostrar/ocultar configuración dialéctica
    const modeRadios = document.querySelectorAll('input[name="analysisMode"]');
    const dialecticConfig = document.getElementById('dialectic-config');
    
    function toggleDialecticConfig() {
        const selectedModeInput = document.querySelector('input[name="analysisMode"]:checked');
        const selectedMode = selectedModeInput ? selectedModeInput.value : 'direct';
        if (selectedMode === 'debate' && dialecticConfig) {
            dialecticConfig.classList.remove('hidden');
        } else if (dialecticConfig) {
            dialecticConfig.classList.add('hidden');
        }
    }
    
    modeRadios.forEach(radio => {
        radio.addEventListener('change', toggleDialecticConfig);
    });
    
    // Listener para nivel de confianza
    const confidenceRadios = document.querySelectorAll('input[name="confidenceLevel"]');
    confidenceRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            dialecticState.confidenceLevel = e.target.value;
        });
    });
    
    // Reemplazar manejador de refinamiento
    const refineDebateBtn = document.getElementById('refine-debate-btn');
    if (refineDebateBtn) {
        const oldHandler = window.handleRefineDebate;
        if (oldHandler) {
            refineDebateBtn.removeEventListener('click', oldHandler);
        }
        refineDebateBtn.addEventListener('click', handleAdvancedRefineDebate);
    }
    
    // Inicialización inicial
    toggleDialecticConfig();
    
    // Inicializar historial de consenso
    dialecticState.consensusHistory = [];
    dialecticState.lastConsensusScore = 0;
}

// Función fallback para bajo consenso (modo no dialéctico)
function handleLowConsensus() {
    const frictionContainer = document.getElementById('friction-container');
    if (frictionContainer && window.fetchAndDisplayFrictionAnalysis) {
        frictionContainer.classList.remove('hidden');
        window.fetchAndDisplayFrictionAnalysis();
    }
}

// Exportar funciones para uso global
window.dialecticEnhancements = {
    initDialecticEnhancements,
    handleAdvancedLowConsensus,
    handleAdvancedRefineDebate,
    handleDebateWithContext,
    calculateAndDisplayConsensusEnhanced,
    showConsensusStagnationWarning,
    dialecticState
};

// Compatibilidad con llamadas del HTML (CORREGIDA)
window.calculateAndDisplayConsensus = function() {
    console.log('Función de compatibilidad ejecutándose...');
    return calculateAndDisplayConsensusEnhanced();
};