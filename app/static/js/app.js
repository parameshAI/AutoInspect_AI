document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const makeSelect = document.getElementById('make');
    const modelSelect = document.getElementById('model');
    const yearSelect = document.getElementById('year');
    const transSelect = document.getElementById('transmission');
    const fuelSelect = document.getElementById('fuel_type');
    const bodySelect = document.getElementById('body_type');
    
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    
    const form = document.getElementById('inspection-form');
    const overlay = document.getElementById('analysis-overlay');
    const processingText = document.getElementById('processing-text');
    
    const emptyState = document.getElementById('empty-state');
    const resultsContainer = document.getElementById('results-container');
    const samplesContainer = document.getElementById('samples-container');
    
    let optionsData = {};
    let selectedImageFile = null;
    let synthesisVoice = null;

    // --- Initialization ---
    initOptions();
    initSamples();
    initVoices();

    // --- Data Loading ---
    async function initOptions() {
        try {
            const res = await fetch('/api/options');
            optionsData = await res.json();
            
            populateSelect(makeSelect, optionsData.makes);
            populateSelect(yearSelect, optionsData.years);
            populateSelect(transSelect, optionsData.transmissions);
            populateSelect(fuelSelect, optionsData.fuel_types);
            populateSelect(bodySelect, optionsData.body_types);
            
            // Trigger model update on make change
            makeSelect.addEventListener('change', () => {
                const make = makeSelect.value;
                const models = optionsData.models_by_make[make] || [];
                populateSelect(modelSelect, models);
            });
            
            // Initialize models for first make
            makeSelect.dispatchEvent(new Event('change'));
            
        } catch (e) {
            console.error("Failed to load options", e);
        }
    }

    async function initSamples() {
        try {
            const res = await fetch('/api/sample-cars');
            const data = await res.json();
            
            samplesContainer.innerHTML = '';
            data.samples.forEach(sample => {
                const card = document.createElement('div');
                card.className = 'sample-card';
                card.innerHTML = `
                    <img src="${sample.image_url}" alt="${sample.title}">
                    <p style="font-weight:700; margin-bottom:0.2rem">${sample.make} ${sample.model}</p>
                    <p>${sample.expected_condition.toUpperCase()}</p>
                `;
                card.addEventListener('click', () => runSampleInspection(sample.id));
                samplesContainer.appendChild(card);
            });
        } catch (e) {
            console.error("Failed to load samples", e);
        }
    }

    function populateSelect(selectElem, items) {
        selectElem.innerHTML = '';
        items.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item;
            opt.textContent = item;
            selectElem.appendChild(opt);
        });
    }

    // --- Drag & Drop ---
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
    
    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;
        selectedImageFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // --- Submission ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        if (selectedImageFile) {
            formData.set('image', selectedImageFile);
        }
        
        showLoader();
        
        try {
            const res = await fetch('/api/inspect', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                renderResults(data);
            } else {
                alert('Inspection Error: ' + data.detail);
                hideLoader();
            }
        } catch (error) {
            console.error(error);
            alert('Failed to connect to the server.');
            hideLoader();
        }
    });

    async function runSampleInspection(sampleId) {
        showLoader();
        
        try {
            const res = await fetch('/api/inspect/sample', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sample_id: sampleId })
            });
            const data = await res.json();
            
            if (res.ok) {
                // Auto-fill form to match sample
                makeSelect.value = data.specs.make;
                makeSelect.dispatchEvent(new Event('change'));
                setTimeout(() => { modelSelect.value = data.specs.model; }, 100);
                yearSelect.value = data.specs.year;
                document.getElementById('mileage').value = data.specs.mileage;
                document.getElementById('engine_size').value = data.specs.engine_size;
                transSelect.value = data.specs.transmission;
                fuelSelect.value = data.specs.fuel_type;
                bodySelect.value = data.specs.body_type;
                
                // Show image
                if (data.image_url) {
                    imagePreview.src = data.image_url;
                    imagePreview.style.display = 'block';
                }
                
                renderResults(data);
            } else {
                alert('Sample Inspection Error: ' + data.detail);
                hideLoader();
            }
        } catch (error) {
            console.error(error);
            alert('Failed to connect to the server.');
            hideLoader();
        }
    }

    function showLoader() {
        overlay.style.display = 'flex';
        
        const texts = [
            "EXTRACTING TABULAR FEATURES...",
            "ANALYZING MARKET DEPRECIATION...",
            "PYTORCH CNN FORWARD PASS...",
            "CALCULATING ADAPTIVE_AVG_POOL...",
            "SYNTHESIZING LLM REPORT..."
        ];
        
        let i = 0;
        window.loaderInterval = setInterval(() => {
            processingText.innerText = texts[i % texts.length];
            i++;
        }, 1200);
    }
    
    function hideLoader() {
        overlay.style.display = 'none';
        clearInterval(window.loaderInterval);
    }

    // --- Results Rendering ---
    function renderResults(data) {
        hideLoader();
        
        emptyState.style.display = 'none';
        resultsContainer.classList.remove('active');
        // Force reflow
        void resultsContainer.offsetWidth;
        resultsContainer.classList.add('active');
        
        // Header
        document.getElementById('res-title').innerText = `${data.specs.year} ${data.specs.make} ${data.specs.model}`;
        document.getElementById('res-id').innerText = data.inspection_id;
        
        // Badge
        const badge = document.getElementById('res-condition-badge');
        badge.innerText = data.condition_label.toUpperCase();
        if (data.condition.toLowerCase() === 'damaged') {
            badge.className = 'condition-badge-large condition-damaged';
        } else {
            badge.className = 'condition-badge-large condition-whole';
        }
        
        // Prices
        document.getElementById('res-price').innerText = formatCurrency(data.adjusted_final_price);
        document.getElementById('res-range').innerText = `Estimated Market Range: ${formatCurrency(data.price_range_low)} - ${formatCurrency(data.price_range_high)}`;
        document.getElementById('res-base-price').innerText = formatCurrency(data.base_predicted_price);
        document.getElementById('res-penalty').innerText = `-${data.damage_penalty_percent}%`;
        document.getElementById('res-penalty').style.color = data.damage_penalty_percent > 0 ? 'var(--accent-danger)' : 'var(--text-secondary)';
        
        // Metrics
        const m = data.metrics;
        setMetric('body', m.body_integrity_score);
        setMetric('struct', m.structural_score);
        setMetric('paint', m.paint_condition_score);
        setMetric('market', m.market_desirability_score);
        
        // Report
        document.getElementById('res-generator').innerText = `By: ${data.generated_by}`;
        
        const contentDiv = document.getElementById('res-report-content');
        contentDiv.innerHTML = '';
        
        data.report_paragraphs.forEach((p, index) => {
            const pElem = document.createElement('div');
            pElem.className = 'report-paragraph';
            pElem.innerText = p;
            contentDiv.appendChild(pElem);
            
            // Staggered animation
            setTimeout(() => {
                pElem.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
                pElem.style.opacity = '1';
                pElem.style.transform = 'translateY(0)';
            }, index * 400 + 300);
            
            pElem.style.opacity = '0';
            pElem.style.transform = 'translateY(15px)';
        });

        // Store for audio
        window.currentReportText = data.report_paragraphs.join('. ');
    }

    function setMetric(id, value) {
        document.getElementById(`met-${id}`).innerText = `${value}/100`;
        setTimeout(() => {
            document.getElementById(`bar-${id}`).style.width = `${value}%`;
        }, 500);
    }

    function formatCurrency(num) {
        return '$' + num.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
    }

    // --- Utilities (Print & Audio) ---
    document.getElementById('btn-print').addEventListener('click', () => {
        window.print();
    });

    function initVoices() {
        if ('speechSynthesis' in window) {
            speechSynthesis.onvoiceschanged = () => {
                const voices = speechSynthesis.getVoices();
                // Try to find a good authoritative/smooth voice
                synthesisVoice = voices.find(v => v.name.includes('Google US English') || v.name.includes('Daniel') || v.name.includes('Samantha')) || voices[0];
            };
        }
    }

    document.getElementById('btn-audio').addEventListener('click', () => {
        if (!window.currentReportText) return;
        if (!('speechSynthesis' in window)) {
            alert("Text-to-speech is not supported in this browser.");
            return;
        }
        
        speechSynthesis.cancel(); // Stop current playing
        
        const utterance = new SpeechSynthesisUtterance(window.currentReportText);
        if (synthesisVoice) utterance.voice = synthesisVoice;
        utterance.rate = 1.05;
        utterance.pitch = 0.95;
        
        speechSynthesis.speak(utterance);
    });
});
