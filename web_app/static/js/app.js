/**
 * QuAC Visualizer JavaScript Application
 * Handles image toggling, mask overlays, filtering, and API interactions
 */

class ReportInfoPane {
    constructor() {
        this.reportInfo = null;
        this.mainChart = null;
    }

    async loadReportInfo() {
        try {
            const response = await fetch('/api/report/info');
            this.reportInfo = await response.json();
            this.updateReportInfoDisplay();
            return this.reportInfo;
        } catch (error) {
            console.error('Error loading report info:', error);
            return null;
        }
    }

    updateReportInfoDisplay() {
        const reportDetails = document.getElementById('report-details');
        if (this.reportInfo.error) {
            reportDetails.innerHTML = `<div class="text-danger">${this.reportInfo.error}</div>`;
            return;
        }

        let displayHtml = `
            <div><strong>Name:</strong> ${this.reportInfo.name}</div>
            <div><strong>Total Explanations:</strong> ${this.reportInfo.num_explanations}</div>
            <div><strong>Data Score Range:</strong> ${this.reportInfo.score_range.data_min.toFixed(3)} - ${this.reportInfo.score_range.data_max.toFixed(3)}</div>
        `;

        if (this.reportInfo.blinding_enabled) {
            displayHtml += `<div class="text-warning"><strong>⚠️ Blinding Mode Active</strong> - Class information hidden</div>`;
        }

        reportDetails.innerHTML = displayHtml;
    }

    async loadMainChart() {
        try {
            const response = await fetch('/api/curve');
            const data = await response.json();
            
            if (data.error) {
                console.error('Error loading main chart:', data.error);
                return;
            }

            this.renderMainChart(data);
        } catch (error) {
            console.error('Error loading main chart:', error);
        }
    }

    renderMainChart(data) {
        const ctx = document.getElementById('quac-curve').getContext('2d');
        
        if (this.mainChart) {
            this.mainChart.destroy();
        }

        this.mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.x_values,
                datasets: [
                    {
                        label: 'Median',
                        data: data.median,
                        borderColor: 'rgb(108, 92, 231)',
                        backgroundColor: 'rgba(108, 92, 231, 0.1)',
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: '25th Percentile',
                        data: data.p25,
                        borderColor: 'rgba(108, 92, 231, 0.3)',
                        backgroundColor: 'rgba(108, 92, 231, 0.05)',
                        borderWidth: 1,
                        fill: '+1'
                    },
                    {
                        label: '75th Percentile',
                        data: data.p75,
                        borderColor: 'rgba(108, 92, 231, 0.3)',
                        backgroundColor: 'rgba(108, 92, 231, 0.05)',
                        borderWidth: 1,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Mask Size',
                            font: { size: 10 }
                        },
                        ticks: { font: { size: 8 } }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Score Change',
                            font: { size: 10 }
                        },
                        ticks: { font: { size: 8 } }
                    }
                },
                plugins: {
                    legend: {
                        labels: { font: { size: 10 } }
                    }
                }
            }
        });
    }
}

class FilterPane {
    constructor() {
        this.onFilterChangeCallback = null;
    }

    populateFilterOptions(reportInfo) {
        const sourceSelect = document.getElementById('source-class');
        const targetSelect = document.getElementById('target-class');

        if (reportInfo.blinding_enabled) {
            // In blinding mode, disable class filter dropdowns
            sourceSelect.disabled = true;
            targetSelect.disabled = true;
            
            // Add a placeholder option explaining blinding mode
            const sourceOption = document.createElement('option');
            sourceOption.value = '';
            sourceOption.textContent = 'Hidden (Blinding Mode)';
            sourceSelect.appendChild(sourceOption);
            
            const targetOption = document.createElement('option');
            targetOption.value = '';
            targetOption.textContent = 'Hidden (Blinding Mode)';
            targetSelect.appendChild(targetOption);
        } else {
            // Normal mode - populate with actual classes
            // Populate source classes
            reportInfo.source_classes.forEach(cls => {
                const option = document.createElement('option');
                option.value = cls;
                option.textContent = cls;
                sourceSelect.appendChild(option);
            });

            // Populate target classes
            reportInfo.target_classes.forEach(cls => {
                const option = document.createElement('option');
                option.value = cls;
                option.textContent = cls;
                targetSelect.appendChild(option);
            });
        }
    }

    setupDualRangeSlider() {
        const container = document.querySelector('.dual-range-container');
        const minThumb = document.getElementById('min-thumb');
        const maxThumb = document.getElementById('max-thumb');
        const range = document.getElementById('dual-range');
        const minInput = document.getElementById('min-score');
        const maxInput = document.getElementById('max-score');
        const minValue = document.getElementById('min-score-value');
        const maxValue = document.getElementById('max-score-value');

        let isDragging = false;
        let activeThumb = null;

        const updateDisplay = () => {
            const minVal = parseFloat(minInput.value);
            const maxVal = parseFloat(maxInput.value);
            
            // Ensure min <= max
            if (minVal > maxVal) {
                if (activeThumb === minThumb) {
                    maxInput.value = minVal;
                } else {
                    minInput.value = maxVal;
                }
            }
            
            const finalMin = parseFloat(minInput.value);
            const finalMax = parseFloat(maxInput.value);
            
            // Update visual display
            minValue.textContent = finalMin.toFixed(3);
            maxValue.textContent = finalMax.toFixed(3);
            
            // Update thumb positions (0-100%)
            const minPercent = finalMin * 100;
            const maxPercent = finalMax * 100;
            
            minThumb.style.left = `${minPercent}%`;
            maxThumb.style.left = `${maxPercent}%`;
            
            // Update range bar
            range.style.left = `${minPercent}%`;
            range.style.width = `${maxPercent - minPercent}%`;
        };

        const startDrag = (thumb, event) => {
            isDragging = true;
            activeThumb = thumb;
            document.body.style.userSelect = 'none';
            event.preventDefault();
        };

        const handleDrag = (event) => {
            if (!isDragging || !activeThumb) return;
            
            const rect = container.getBoundingClientRect();
            const clientX = event.touches ? event.touches[0].clientX : event.clientX;
            const x = clientX - rect.left;
            const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
            const value = percent / 100;
            
            if (activeThumb === minThumb) {
                minInput.value = value.toFixed(3);
            } else {
                maxInput.value = value.toFixed(3);
            }
            
            updateDisplay();
        };

        const endDrag = () => {
            if (isDragging) {
                isDragging = false;
                activeThumb = null;
                document.body.style.userSelect = '';
            }
        };

        // Mouse events
        minThumb.addEventListener('mousedown', (e) => startDrag(minThumb, e));
        maxThumb.addEventListener('mousedown', (e) => startDrag(maxThumb, e));

        document.addEventListener('mousemove', handleDrag);
        document.addEventListener('mouseup', endDrag);

        // Touch events
        minThumb.addEventListener('touchstart', (e) => startDrag(minThumb, e.touches[0]));
        maxThumb.addEventListener('touchstart', (e) => startDrag(maxThumb, e.touches[0]));

        document.addEventListener('touchmove', (e) => {
            e.preventDefault();
            handleDrag(e.touches[0]);
        });
        document.addEventListener('touchend', endDrag);

        // Initial display update
        updateDisplay();
    }

    setupFilterEventListeners(applyFiltersCallback, resetFiltersCallback) {
        // Filter form
        document.getElementById('filter-form').addEventListener('submit', (e) => {
            e.preventDefault();
            applyFiltersCallback();
        });

        // Reset filters button
        document.getElementById('reset-filters').addEventListener('click', () => {
            resetFiltersCallback();
        });
    }

    resetFilters() {
        document.getElementById('source-class').value = '';
        document.getElementById('target-class').value = '';
        
        // Reset dual range slider to full range
        const minSlider = document.getElementById('min-score');
        const maxSlider = document.getElementById('max-score');
        minSlider.value = 0;
        maxSlider.value = 1;
        document.getElementById('min-score-value').textContent = '0.000';
        document.getElementById('max-score-value').textContent = '1.000';
        
        // Update slider visual
        const minThumb = document.getElementById('min-thumb');
        const maxThumb = document.getElementById('max-thumb');
        const range = document.getElementById('dual-range');
        
        minThumb.style.left = '0%';
        maxThumb.style.left = '100%';
        range.style.left = '0%';
        range.style.width = '100%';
    }

    getFilterValues() {
        const formData = new FormData(document.getElementById('filter-form'));
        const filters = {};
        for (let [key, value] of formData.entries()) {
            if (value) filters[key] = value;
        }
        return filters;
    }

    onFilterChange(callback) {
        this.onFilterChangeCallback = callback;
    }
}

class ExplanationListPane {
    constructor() {
        this.currentExplanations = [];
        this.totalExplanations = 0;
        this.offset = 0;
        this.limit = 20;
        this.onExplanationSelectCallback = null;
    }

    async loadExplanations(reportInfo, resetOffset = false) {
        try {
            if (resetOffset) {
                this.offset = 0;
            }

            const formData = new FormData(document.getElementById('filter-form'));
            const params = new URLSearchParams();
            
            // Add form parameters with explicit handling
            const sourceClass = formData.get('source_class');
            const targetClass = formData.get('target_class');
            const minScore = formData.get('min_score');
            const maxScore = formData.get('max_score');
            
            // Only add non-empty parameters
            if (sourceClass && sourceClass !== '') {
                params.append('source_class', sourceClass);
            }
            if (targetClass && targetClass !== '') {
                params.append('target_class', targetClass);
            }
            if (minScore && minScore !== '') {
                params.append('min_score', minScore);
            }
            if (maxScore && maxScore !== '') {
                params.append('max_score', maxScore);
            }
            
            params.append('offset', this.offset);
            params.append('limit', this.limit);

            console.log('Loading explanations with params:', params.toString()); // Debug log

            const response = await fetch(`/api/explanations?${params}`);
            const data = await response.json();
            
            if (data.error) {
                console.error('Error loading explanations:', data.error);
                return;
            }

            if (this.offset === 0) {
                this.currentExplanations = data.explanations;
            } else {
                this.currentExplanations.push(...data.explanations);
            }
            
            this.totalExplanations = data.total;
            this.updateExplanationsList(reportInfo);
            this.updateLoadMoreButton();
        } catch (error) {
            console.error('Error loading explanations:', error);
        }
    }

    async loadMoreExplanations(reportInfo) {
        this.offset += this.limit;
        await this.loadExplanations(reportInfo, false);
    }

    updateExplanationsList(reportInfo) {
        const listContainer = document.getElementById('explanation-list');
        
        if (this.offset === 0) {
            listContainer.innerHTML = '';
        }
        
        this.currentExplanations.slice(this.offset === 0 ? 0 : -this.limit).forEach(exp => {
            const item = this.createExplanationListItem(exp, reportInfo);
            listContainer.appendChild(item);
        });

        document.getElementById('explanation-count').textContent = this.totalExplanations;
    }

    createExplanationListItem(explanation, reportInfo) {
        const item = document.createElement('div');
        item.className = 'list-group-item explanation-item';
        item.dataset.explanationId = explanation.id;

        const scoreClass = this.getScoreClass(explanation.score);
        
        // Hide class information if blinding is enabled
        const classDisplay = reportInfo.blinding_enabled 
            ? 'Hidden → Hidden' 
            : `${explanation.source_class} → ${explanation.target_class}`;
        
        item.innerHTML = `
            <div class="explanation-header">
                ${classDisplay}
            </div>
            <div class="explanation-details">
                Method: ${explanation.method || 'N/A'}
            </div>
            <div class="explanation-score">
                <span class="score-badge ${scoreClass}">${explanation.score.toFixed(4)}</span>
                <div class="score-bar mt-1">
                    <div class="score-fill" style="width: ${(explanation.score * 100)}%"></div>
                </div>
            </div>
        `;

        item.addEventListener('click', () => {
            this.selectExplanation(explanation);
        });

        return item;
    }

    getScoreClass(score) {
        if (score >= 0.6) return '';
        if (score >= 0.3) return 'medium-score';
        return 'low-score';
    }

    updateLoadMoreButton() {
        const container = document.getElementById('load-more-container');
        const button = document.getElementById('load-more');
        
        if (this.currentExplanations.length >= this.totalExplanations) {
            container.style.display = 'none';
        } else {
            container.style.display = 'block';
            button.textContent = `Load More (${this.currentExplanations.length}/${this.totalExplanations})`;
        }
    }

    selectExplanation(explanation) {
        // Update visual selection efficiently
        const listContainer = document.getElementById('explanation-list');
        const currentActive = listContainer.querySelector('.explanation-item.active');
        if (currentActive) {
            currentActive.classList.remove('active');
        }
        
        const newActive = listContainer.querySelector(`[data-explanation-id="${explanation.id}"]`);
        if (newActive) {
            newActive.classList.add('active');
        }

        // Call the callback to notify the main class
        if (this.onExplanationSelectCallback) {
            this.onExplanationSelectCallback(explanation);
        }
    }

    setupEventListeners(onLoadMoreCallback) {
        // Load more button
        document.getElementById('load-more').addEventListener('click', () => {
            onLoadMoreCallback();
        });
    }

    setExplanationSelectCallback(callback) {
        this.onExplanationSelectCallback = callback;
    }
}

class ExplanationViewerPane {
    constructor() {
        this.currentExplanation = null;
        this.currentMask = null;
    }

    async loadMaskData() {
        if (!this.currentExplanation) return;
        
        try {
            const response = await fetch(`/api/mask/${this.currentExplanation.id}`);
            const responseText = await response.text();
            
            let data;
            try {
                data = JSON.parse(responseText);
            } catch (jsonError) {
                console.error('JSON parse error:', jsonError);
                return;
            }
            
            if (data.error) {
                console.error('Error loading mask:', data.error);
                return;
            }
            
            this.currentMask = data;
        } catch (error) {
            console.error('Error loading mask data:', error);
        }
    }

    async showImage(imageType) {
        if (!this.currentExplanation) return;

        const mainImage = document.getElementById('main-image');
        const maskOverlay = document.getElementById('mask-overlay');
        
        mainImage.classList.add('loading');

        try {
            // Use new ID-based endpoint
            mainImage.src = `/api/image/${this.currentExplanation.id}/${imageType}`;
            
            await new Promise((resolve, reject) => {
                mainImage.onload = () => {
                    mainImage.classList.remove('loading');
                    
                    // Update canvas size to match image
                    maskOverlay.width = mainImage.naturalWidth;
                    maskOverlay.height = mainImage.naturalHeight;
                    maskOverlay.style.width = mainImage.clientWidth + 'px';
                    maskOverlay.style.height = mainImage.clientHeight + 'px';
                    
                    // Update mask overlay with current opacity
                    const opacity = parseInt(document.getElementById('mask-opacity').value);
                    this.updateMaskOpacity(opacity);
                    
                    resolve();
                };
                mainImage.onerror = reject;
            });
            
        } catch (error) {
            console.error('Error loading image:', error);
            mainImage.classList.remove('loading');
        }
    }

    updateMaskOpacity(opacity) {
        const maskOverlay = document.getElementById('mask-overlay');
        
        if (!this.currentMask) {
            maskOverlay.style.display = 'none';
        } else {
            maskOverlay.style.display = 'block';
            // Always draw the spotlight effect first
            this.drawMaskOverlay(opacity);
            // Then draw the contour on top (independent of opacity)
            this.drawMaskContour();
        }
    }

    drawMaskOverlay(opacity = 30) {
        if (!this.currentMask) {
            console.log('No current mask data');
            return;
        }

        const canvas = document.getElementById('mask-overlay');
        const ctx = canvas.getContext('2d');
        const maskData = this.currentMask.mask;
        
        if (!maskData || !Array.isArray(maskData) || !maskData.length) {
            console.error('Invalid mask data format');
            return;
        }
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const height = maskData.length;
        const width = maskData[0]?.length;
        
        if (!width) {
            console.error('Invalid mask dimensions');
            return;
        }

        // Create ImageData for the mask (server returns RGB)
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;
        
        // Convert opacity percentage to alpha value (0-255)
        const alpha = Math.round((opacity / 100) * 255);
        
        // First pass: create base overlay
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const index = (y * width + x) * 4;
                const pixel = maskData[y][x];
                
                // Check if any channel has non-zero value (works for any color mask)
                const hasMask = pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0;
                
                data[index] = 0;             // Black overlay
                data[index + 1] = 0;         // Black overlay  
                data[index + 2] = 0;         // Black overlay
                data[index + 3] = hasMask ? 0 : alpha;  // Alpha only where mask is zero
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        // Draw contour separately if enabled
        this.drawMaskContour();
    }

    drawMaskContour() {
        if (!this.currentMask) {
            return;
        }

        const showContour = document.getElementById('mask-contour').checked;
        if (!showContour) {
            return;
        }

        const canvas = document.getElementById('mask-overlay');
        const ctx = canvas.getContext('2d');
        const maskData = this.currentMask.mask;
        
        const height = maskData.length;
        const width = maskData[0]?.length;
        
        // Get current canvas data to modify it
        const currentImageData = ctx.getImageData(0, 0, width, height);
        const data = currentImageData.data;
        
        for (let y = 2; y < height - 2; y++) {
            for (let x = 2; x < width - 2; x++) {
                const pixel = maskData[y][x];
                const hasMask = pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0;
                
                if (!hasMask) {
                    // Check larger neighborhood for thicker contour (2-pixel radius)
                    let hasNeighborMask = false;
                    for (let dy = -2; dy <= 2; dy++) {
                        for (let dx = -2; dx <= 2; dx++) {
                            const neighbor = maskData[y + dy][x + dx];
                            if (neighbor[0] > 0 || neighbor[1] > 0 || neighbor[2] > 0) {
                                hasNeighborMask = true;
                                break;
                            }
                        }
                        if (hasNeighborMask) break;
                    }
                    
                    if (hasNeighborMask) {
                        // Directly set contour pixels in the existing canvas data
                        const index = (y * width + x) * 4;
                        data[index] = 255;     // Red
                        data[index + 1] = 0;   // Green  
                        data[index + 2] = 255; // Blue (magenta)
                        data[index + 3] = 200; // Alpha - fixed high opacity
                    }
                }
            }
        }
        
        // Put the modified data back
        ctx.putImageData(currentImageData, 0, 0);
    }

    showViewer() {
        const imageViewer = document.getElementById('image-viewer');
        const noExplanation = document.getElementById('no-explanation');
        
        // Completely hide the no-explanation div
        noExplanation.style.display = 'none';
        noExplanation.style.position = 'absolute';
        noExplanation.style.top = '-9999px';
        
        // Show the image viewer
        imageViewer.style.display = 'flex';
        imageViewer.style.visibility = 'visible';
        imageViewer.style.position = 'static';
    }

    closeViewer() {
        const imageViewer = document.getElementById('image-viewer');
        const noExplanation = document.getElementById('no-explanation');
        
        // Hide the image viewer
        imageViewer.style.display = 'none';
        imageViewer.style.visibility = 'hidden';
        imageViewer.style.position = 'absolute';
        imageViewer.style.top = '-9999px';
        
        // Show the no-explanation div
        noExplanation.style.display = 'flex';
        noExplanation.style.visibility = 'visible';
        noExplanation.style.position = 'static';
        
        // Clear current explanation and annotation
        this.currentExplanation = null;
        this.currentMask = null;
        
        // Clear annotation textarea
        const annotationTextarea = document.getElementById('explanation-annotation');
        if (annotationTextarea) {
            annotationTextarea.value = '';
        }
    }

    updateExplanationMetadata(explanation) {
        document.getElementById('exp-score').textContent = explanation.score.toFixed(4);
        
        // Hide class information if blinding is enabled
        const reportInfo = window.app?.reportInfo; // Access through global app instance
        if (reportInfo?.blinding_enabled) {
            document.getElementById('exp-source-class').textContent = 'Hidden';
            document.getElementById('exp-target-class').textContent = 'Hidden';
        } else {
            document.getElementById('exp-source-class').textContent = explanation.source_class;
            document.getElementById('exp-target-class').textContent = explanation.target_class;
        }
        
        document.getElementById('exp-method').textContent = explanation.method || 'N/A';
    }

    setCurrentExplanation(explanation) {
        this.currentExplanation = explanation;
        this.currentMask = null;
        this.loadAnnotation();
    }

    loadAnnotation() {
        const annotationTextarea = document.getElementById('explanation-annotation');
        if (this.currentExplanation && annotationTextarea) {
            annotationTextarea.value = this.currentExplanation.annotation || '';
        }
    }

    async saveAnnotation() {
        if (!this.currentExplanation) return;
        
        const annotationTextarea = document.getElementById('explanation-annotation');
        const annotation = annotationTextarea.value;
        
        console.log('Saving annotation for explanation:', this.currentExplanation);
        console.log('Explanation ID:', this.currentExplanation.id);
        
        try {
            // Save to backend
            const response = await fetch(`/api/explanation/${this.currentExplanation.id}/annotation`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ annotation: annotation })
            });
            
            if (response.ok) {
                // Update the in-memory explanation object only if backend save succeeded
                this.currentExplanation.annotation = annotation;
                
                // Show save status
                this.showAnnotationSaved();
                
                console.log(`Annotation saved for explanation ${this.currentExplanation.id}:`, annotation);
            } else {
                console.error('Failed to save annotation:', await response.text());
            }
        } catch (error) {
            console.error('Error saving annotation:', error);
        }
    }

    showAnnotationSaved() {
        const status = document.getElementById('annotation-status');
        if (status) {
            status.style.display = 'inline';
            setTimeout(() => {
                status.style.display = 'none';
            }, 2000);
        }
    }

    setupAnnotationAutoSave() {
        const annotationTextarea = document.getElementById('explanation-annotation');
        if (!annotationTextarea) return;

        let autoSaveTimeout;

        annotationTextarea.addEventListener('input', () => {
            // Clear previous timeout
            if (autoSaveTimeout) {
                clearTimeout(autoSaveTimeout);
            }

            // Set new timeout for auto-save (2 seconds after typing stops)
            autoSaveTimeout = setTimeout(() => {
                this.saveAnnotation();
            }, 2000);
        });

        // Also save when user leaves the textarea
        annotationTextarea.addEventListener('blur', () => {
            if (autoSaveTimeout) {
                clearTimeout(autoSaveTimeout);
            }
            this.saveAnnotation();
        });
    }

    setupEventListeners() {
        // Image toggle buttons
        document.getElementById('show-query').addEventListener('change', () => {
            if (this.currentExplanation) {
                this.showImage('query');
            }
        });

        document.getElementById('show-counterfactual').addEventListener('change', () => {
            if (this.currentExplanation) {
                this.showImage('counterfactual');
            }
        });

        // Mask opacity slider
        document.getElementById('mask-opacity').addEventListener('input', (e) => {
            const opacity = parseInt(e.target.value);
            this.updateMaskOpacity(opacity);
        });

        // Mask contour checkbox
        document.getElementById('mask-contour').addEventListener('change', (e) => {
            const opacity = parseInt(document.getElementById('mask-opacity').value);
            this.updateMaskOpacity(opacity);
        });

        // Close viewer button
        document.getElementById('close-viewer').addEventListener('click', () => {
            this.closeViewer();
        });

        // Setup annotation auto-save
        this.setupAnnotationAutoSave();
        
        // Add keyboard shortcut for annotation (A key)
        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'a' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                // Only if not typing in another input
                if (document.activeElement.tagName !== 'INPUT' && 
                    document.activeElement.tagName !== 'TEXTAREA') {
                    const annotationTextarea = document.getElementById('explanation-annotation');
                    if (annotationTextarea && this.currentExplanation) {
                        annotationTextarea.focus();
                        e.preventDefault();
                    }
                }
            }
        });
    }
}

class QuACVisualizer {
    constructor() {
        this.reportInfoPane = new ReportInfoPane();
        this.filterPane = new FilterPane();
        this.explanationListPane = new ExplanationListPane();
        this.explanationViewerPane = new ExplanationViewerPane();
        this.reportInfo = null; // Keep for compatibility with other methods
        this.charts = {};
        this.sidebarWidth = 300; // Track current sidebar width
        this.sidebarCollapsed = false;
        
        // Make app globally accessible for panes
        window.app = this;
        
        this.init();
    }

    setNoExplanationState() {
        const imageViewer = document.getElementById('image-viewer');
        const noExplanation = document.getElementById('no-explanation');
        
        // Completely hide image viewer
        imageViewer.style.display = 'none';
        imageViewer.style.visibility = 'hidden';
        imageViewer.style.position = 'absolute';
        imageViewer.style.top = '-9999px';
        
        // Show centered no-explanation message
        noExplanation.style.display = 'flex';
        noExplanation.style.visibility = 'visible';
        noExplanation.style.position = 'static';
        noExplanation.style.top = 'auto';
    }

    async init() {
        // Ensure proper initial state - no explanation selected
        this.setNoExplanationState();
        
        const reportInfo = await this.reportInfoPane.loadReportInfo();
        if (reportInfo) {
            this.reportInfo = reportInfo; // Store for other methods to access
            this.filterPane.populateFilterOptions(reportInfo);
        }
        
        // Setup ExplanationListPane callbacks
        this.explanationListPane.setExplanationSelectCallback((explanation) => {
            this.handleExplanationSelection(explanation);
        });
        
        this.setupEventListeners();
        await this.explanationListPane.loadExplanations(this.reportInfo, true);
        await this.reportInfoPane.loadMainChart();
    }
    setupEventListeners() {
        // Setup filter event listeners through FilterPane
        this.filterPane.setupFilterEventListeners(
            () => this.applyFilters(),
            () => this.resetFilters()
        );

        // Setup dual range slider
        this.filterPane.setupDualRangeSlider();

        // Setup explanation list event listeners
        this.explanationListPane.setupEventListeners(() => {
            this.explanationListPane.loadMoreExplanations(this.reportInfo);
        });

        // Setup explanation viewer event listeners
        this.explanationViewerPane.setupEventListeners();

        // Collapsed sidebar indicator toggle
        document.getElementById('sidebar-collapsed').addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Resizable dividers
        this.setupResizableDivider();
        this.setupSidebarResizer();

        // Download buttons
        document.getElementById('download-json').addEventListener('click', () => {
            this.downloadFilteredData('json');
        });

        document.getElementById('download-images').addEventListener('click', () => {
            this.downloadFilteredData('images');
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Don't trigger shortcuts when typing in input fields or textareas
            if (document.activeElement.tagName === 'INPUT' || 
                document.activeElement.tagName === 'TEXTAREA') {
                // Escape key should unfocus from textarea/input instead of closing viewer
                if (e.key === 'Escape') {
                    document.activeElement.blur();
                    e.preventDefault();
                }
                return;
            }

            if (this.explanationViewerPane.currentExplanation) {
                switch(e.key) {
                    case 'q':
                    case 'Q':
                        document.getElementById('show-query').checked = true;
                        this.explanationViewerPane.showImage('query');
                        break;
                    case 'c':
                    case 'C':
                        document.getElementById('show-counterfactual').checked = true;
                        this.explanationViewerPane.showImage('counterfactual');
                        break;
                    case 'm':
                    case 'M':
                        const maskSlider = document.getElementById('mask-opacity');
                        const currentOpacity = parseInt(maskSlider.value);
                        const newOpacity = currentOpacity === 0 ? 30 : 0;
                        maskSlider.value = newOpacity;
                        document.getElementById('mask-opacity-value').textContent = `${newOpacity}%`;
                        this.explanationViewerPane.updateMaskOpacity(newOpacity);
                        break;
                    case 'Escape':
                        this.explanationViewerPane.closeViewer();
                        break;
                }
            }
        });
    }

    async applyFilters() {
        this.showLoading(true);
        await this.explanationListPane.loadExplanations(this.reportInfo, true);
        await this.updateMainChart();
        this.showLoading(false);
    }

    async updateMainChart() {
        try {
            const formData = new FormData(document.getElementById('filter-form'));
            const params = new URLSearchParams();
            
            for (let [key, value] of formData.entries()) {
                if (value) params.append(key, value);
            }

            const response = await fetch(`/api/curve?${params}`);
            const data = await response.json();
            
            if (data.error) {
                console.error('Error loading filtered chart:', data.error);
                return;
            }

            this.reportInfoPane.renderMainChart(data);
            
        } catch (error) {
            console.error('Error updating main chart:', error);
        }
    }



    resetFilters() {
        this.filterPane.resetFilters();
        this.applyFilters();
    }



    handleExplanationSelection(explanation) {
        // Set the explanation in the viewer pane
        this.explanationViewerPane.setCurrentExplanation(explanation);
        
        // Show the viewer
        this.explanationViewerPane.showViewer();
        
        // Update metadata
        this.explanationViewerPane.updateExplanationMetadata(explanation);
        
        // Show initial image (query by default)
        document.getElementById('show-query').checked = true;
        
        // Load mask data and show image asynchronously (non-blocking)
        this.explanationViewerPane.loadMaskData().then(() => {
            this.explanationViewerPane.showImage('query');
        });
        
        // Update individual curve
        this.updateIndividualCurve(explanation);
    }

    updateIndividualCurve(explanation) {
        if (!explanation.normalized_mask_sizes || !explanation.score_changes) {
            return;
        }

        const ctx = document.getElementById('individual-curve').getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts.individual) {
            this.charts.individual.destroy();
        }

        // Reverse the arrays so x-axis goes from 0 to 1 instead of 1 to 0
        const xValues = [...explanation.normalized_mask_sizes].reverse();
        const yValues = [...explanation.score_changes].reverse();

        this.charts.individual = new Chart(ctx, {
            type: 'line',
            data: {
                labels: xValues,
                datasets: [{
                    label: 'Score Change',
                    data: yValues,
                    borderColor: 'rgb(108, 92, 231)',
                    backgroundColor: 'rgba(108, 92, 231, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        display: true,
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.1
                        },
                        title: {
                            display: true,
                            text: 'Normalized Mask Size'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Score Change'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: `QuAC Score: ${explanation.score.toFixed(4)}`
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        right: 20,
                        bottom: 10,
                        left: 20
                    }
                }
            }
        });
    }

    downloadFilteredData(type) {
        console.log('Download button clicked, type:', type);
        
        const formData = new FormData(document.getElementById('filter-form'));
        const params = new URLSearchParams();
        
        for (let [key, value] of formData.entries()) {
            if (value) params.append(key, value);
        }

        const endpoint = type === 'json' ? 'explanations' : 'images';
        const url = `/api/download/${endpoint}?${params}`;
        
        console.log('Download URL:', url);
        
        // Try window.open first (works better with SSH tunnels)
        try {
            window.open(url, '_blank');
            console.log('Used window.open for download');
        } catch (error) {
            console.log('window.open failed, trying direct navigation:', error);
            // Fallback: direct navigation
            window.location.href = url;
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'block' : 'none';
    }

    updateSidebarLayout(width, autoCollapsed = false) {
        const sidebar = document.getElementById('sidebar');
        const sidebarCollapsed = document.getElementById('sidebar-collapsed');
        const sidebarDivider = document.getElementById('sidebar-divider');
        
        // Auto-collapse if width is too small for content (less than 200px)
        if (width > 0 && width < 200 && !autoCollapsed) {
            this.sidebarCollapsed = true;
            width = 0;
            autoCollapsed = true;
        }
        
        if (width === 0) {
            // Collapsed state
            sidebar.style.width = '0px';
            sidebar.style.padding = '0';
            sidebar.style.overflow = 'hidden';
            sidebarCollapsed.style.display = 'flex';
            sidebarDivider.style.display = 'none';
        } else {
            // Expanded state  
            sidebar.style.width = width + 'px';
            sidebar.style.padding = '1rem';
            sidebar.style.overflow = 'auto';
            sidebarCollapsed.style.display = 'none';
            sidebarDivider.style.display = 'block';
        }
        
        return autoCollapsed;
    }

    toggleSidebar() {
        this.sidebarCollapsed = !this.sidebarCollapsed;
        
        if (this.sidebarCollapsed) {
            this.updateSidebarLayout(0);
        } else {
            this.updateSidebarLayout(this.sidebarWidth);
        }
    }

    setupResizableDivider() {
        const divider = document.getElementById('divider');
        const explanationsPane = document.getElementById('explanations-pane');
        const imageViewerPane = document.getElementById('image-viewer-pane');
        let isResizing = false;

        divider.addEventListener('mousedown', (e) => {
            isResizing = true;
            document.body.classList.add('resizing');
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;

            const container = divider.parentElement;
            const containerRect = container.getBoundingClientRect();
            const containerWidth = containerRect.width;
            const mouseX = e.clientX - containerRect.left;
            
            // Calculate new width as percentage
            const newWidth = (mouseX / containerWidth) * 100;
            
            // Set minimum and maximum widths (20% to 80%)
            const minWidth = 20;
            const maxWidth = 80;
            
            if (newWidth >= minWidth && newWidth <= maxWidth) {
                explanationsPane.style.width = newWidth + '%';
                // The image viewer pane will flex-grow to fill remaining space
            }
        });

        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.classList.remove('resizing');
            }
        });
    }

    setupSidebarResizer() {
        const sidebarDivider = document.getElementById('sidebar-divider');
        const sidebar = document.getElementById('sidebar');
        let isResizing = false;

        sidebarDivider.addEventListener('mousedown', (e) => {
            if (this.sidebarCollapsed) return;
            isResizing = true;
            document.body.classList.add('resizing');
            // Disable transitions during resize
            sidebar.style.transition = 'none';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;

            const mouseX = e.clientX;
            const minWidth = 50; // Allow dragging smaller to trigger auto-collapse
            const maxWidth = 600;
            
            if (mouseX >= minWidth && mouseX <= maxWidth) {
                this.sidebarWidth = Math.max(200, mouseX); // Store minimum 200px for when we expand again
                const autoCollapsed = this.updateSidebarLayout(mouseX);
                if (autoCollapsed) {
                    this.sidebarCollapsed = true;
                }
            }
        });

        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.classList.remove('resizing');
                // Re-enable transitions after resize
                sidebar.style.transition = 'width 0.3s ease-in-out, padding 0.3s ease-in-out';
            }
        });
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new QuACVisualizer();
});