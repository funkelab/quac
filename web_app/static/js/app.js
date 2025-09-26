/**
 * QuAC Visualizer JavaScript Application
 * Handles image toggling, mask overlays, filtering, and API interactions
 */

class QuACVisualizer {
    constructor() {
        this.currentExplanations = [];
        this.currentExplanation = null;
        this.currentMask = null;
        this.offset = 0;
        this.limit = 20;
        this.totalExplanations = 0;
        this.charts = {};
        this.reportInfo = null;
        
        this.init();
    }

    async init() {
        await this.loadReportInfo();
        this.setupEventListeners();
        await this.loadExplanations();
        this.setupQuACCurve();
    }

    async loadReportInfo() {
        try {
            const response = await fetch('/api/report/info');
            this.reportInfo = await response.json();
            this.updateReportInfoDisplay();
            this.populateFilterOptions();
        } catch (error) {
            console.error('Error loading report info:', error);
        }
    }

    updateReportInfoDisplay() {
        const reportDetails = document.getElementById('report-details');
        if (this.reportInfo.error) {
            reportDetails.innerHTML = `<div class="text-danger">${this.reportInfo.error}</div>`;
            return;
        }

        reportDetails.innerHTML = `
            <div><strong>Name:</strong> ${this.reportInfo.name}</div>
            <div><strong>Total Explanations:</strong> ${this.reportInfo.num_explanations}</div>
            <div><strong>Data Score Range:</strong> ${this.reportInfo.score_range.data_min.toFixed(3)} - ${this.reportInfo.score_range.data_max.toFixed(3)}</div>
        `;

        // Keep filter sliders at theoretical range (0-1), but set initial values to data range
        const minSlider = document.getElementById('min-score');
        const maxSlider = document.getElementById('max-score');
        
        // Sliders always allow full 0-1 range
        minSlider.min = 0;
        minSlider.max = 1;
        maxSlider.min = 0;
        maxSlider.max = 1;
        
        // Set initial values to data range for convenience
        minSlider.value = Math.max(0, this.reportInfo.score_range.data_min);
        maxSlider.value = Math.min(1, this.reportInfo.score_range.data_max);
        
        // Update display values
        document.getElementById('min-score-value').textContent = minSlider.value;
        document.getElementById('max-score-value').textContent = maxSlider.value;
    }

    populateFilterOptions() {
        const sourceSelect = document.getElementById('source-class');
        const targetSelect = document.getElementById('target-class');

        // Populate source classes
        this.reportInfo.source_classes.forEach(cls => {
            const option = document.createElement('option');
            option.value = cls;
            option.textContent = cls;
            sourceSelect.appendChild(option);
        });

        // Populate target classes
        this.reportInfo.target_classes.forEach(cls => {
            const option = document.createElement('option');
            option.value = cls;
            option.textContent = cls;
            targetSelect.appendChild(option);
        });
    }

    setupEventListeners() {
        // Filter form
        document.getElementById('filter-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.applyFilters();
        });

        // Reset filters button
        document.getElementById('reset-filters').addEventListener('click', () => {
            this.resetFilters();
        });

        // Range slider updates
        document.getElementById('min-score').addEventListener('input', (e) => {
            document.getElementById('min-score-value').textContent = parseFloat(e.target.value).toFixed(2);
        });

        document.getElementById('max-score').addEventListener('input', (e) => {
            document.getElementById('max-score-value').textContent = parseFloat(e.target.value).toFixed(2);
        });

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

        // Mask toggle
        document.getElementById('show-mask').addEventListener('change', (e) => {
            this.toggleMask(e.target.checked);
        });

        // Close viewer
        document.getElementById('close-viewer').addEventListener('click', () => {
            this.closeViewer();
        });

        // Load more button
        document.getElementById('load-more').addEventListener('click', () => {
            this.loadMoreExplanations();
        });

        // Download buttons
        document.getElementById('download-json').addEventListener('click', () => {
            this.downloadFilteredData('json');
        });

        document.getElementById('download-images').addEventListener('click', () => {
            this.downloadFilteredData('images');
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (this.currentExplanation) {
                switch(e.key) {
                    case 'q':
                    case 'Q':
                        document.getElementById('show-query').checked = true;
                        this.showImage('query');
                        break;
                    case 'c':
                    case 'C':
                        document.getElementById('show-counterfactual').checked = true;
                        this.showImage('counterfactual');
                        break;
                    case 'm':
                    case 'M':
                        const maskCheckbox = document.getElementById('show-mask');
                        maskCheckbox.checked = !maskCheckbox.checked;
                        this.toggleMask(maskCheckbox.checked);
                        break;
                    case 'Escape':
                        this.closeViewer();
                        break;
                }
            }
        });
    }

    async applyFilters() {
        this.showLoading(true);
        this.offset = 0;
        await this.loadExplanations();
        await this.updateQuACCurve();
        this.showLoading(false);
    }

    resetFilters() {
        // Reset form fields to default values
        document.getElementById('source-class').value = '';
        document.getElementById('target-class').value = '';
        document.getElementById('min-score').value = 0;
        document.getElementById('max-score').value = 1;
        
        // Update display values
        document.getElementById('min-score-value').textContent = '0.00';
        document.getElementById('max-score-value').textContent = '1.00';
        
        // Apply the reset filters
        this.applyFilters();
    }

    async loadExplanations() {
        try {
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
            this.updateExplanationsList();
            this.updateLoadMoreButton();
        } catch (error) {
            console.error('Error loading explanations:', error);
        }
    }

    async loadMoreExplanations() {
        this.offset += this.limit;
        await this.loadExplanations();
    }

    updateExplanationsList() {
        const listContainer = document.getElementById('explanation-list');
        
        if (this.offset === 0) {
            listContainer.innerHTML = '';
        }
        
        this.currentExplanations.slice(this.offset === 0 ? 0 : -this.limit).forEach(exp => {
            const item = this.createExplanationListItem(exp);
            listContainer.appendChild(item);
        });

        document.getElementById('explanation-count').textContent = this.totalExplanations;
    }

    createExplanationListItem(explanation) {
        const item = document.createElement('div');
        item.className = 'list-group-item explanation-item';
        item.dataset.explanationId = explanation.id;

        const scoreClass = this.getScoreClass(explanation.score);
        
        item.innerHTML = `
            <div class="explanation-header">
                ${explanation.source_class} → ${explanation.target_class}
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

    async selectExplanation(explanation) {
        // Update visual selection
        document.querySelectorAll('.explanation-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-explanation-id="${explanation.id}"]`).classList.add('active');

        this.currentExplanation = explanation;
        this.currentMask = null;

        // Show viewer and update metadata
        document.getElementById('no-explanation').style.display = 'none';
        document.getElementById('image-viewer').style.display = 'block';
        
        this.updateExplanationMetadata(explanation);
        
        // Load mask data
        if (explanation.mask_path) {
            await this.loadMaskData(explanation.mask_path);
        }
        
        // Show initial image (query by default)
        document.getElementById('show-query').checked = true;
        await this.showImage('query');
        
        // Update individual curve
        this.updateIndividualCurve(explanation);
    }

    updateExplanationMetadata(explanation) {
        document.getElementById('viewer-title').textContent = 
            `${explanation.source_class} → ${explanation.target_class}`;
        document.getElementById('exp-score').textContent = explanation.score.toFixed(4);
        document.getElementById('exp-source-class').textContent = explanation.source_class;
        document.getElementById('exp-target-class').textContent = explanation.target_class;
        document.getElementById('exp-method').textContent = explanation.method || 'N/A';
    }

    async loadMaskData(maskPath) {
        try {
            const response = await fetch(`/api/mask/${encodeURIComponent(maskPath)}`);
            const data = await response.json();
            
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
        const imageInfo = document.getElementById('image-info');
        
        mainImage.classList.add('loading');

        const imagePath = imageType === 'query' ? 
            this.currentExplanation.query_path : 
            this.currentExplanation.counterfactual_path;

        try {
            mainImage.src = `/api/image/${encodeURIComponent(imagePath)}`;
            
            await new Promise((resolve, reject) => {
                mainImage.onload = () => {
                    mainImage.classList.remove('loading');
                    
                    // Update canvas size to match image
                    maskOverlay.width = mainImage.naturalWidth;
                    maskOverlay.height = mainImage.naturalHeight;
                    maskOverlay.style.width = mainImage.clientWidth + 'px';
                    maskOverlay.style.height = mainImage.clientHeight + 'px';
                    
                    // Update mask overlay if it's enabled
                    if (document.getElementById('show-mask').checked) {
                        this.drawMaskOverlay();
                    }
                    
                    resolve();
                };
                mainImage.onerror = reject;
            });

            imageInfo.textContent = `${imageType.charAt(0).toUpperCase() + imageType.slice(1)} Image - ${imagePath}`;
            
        } catch (error) {
            console.error('Error loading image:', error);
            mainImage.classList.remove('loading');
        }
    }

    toggleMask(show) {
        const maskOverlay = document.getElementById('mask-overlay');
        
        if (show && this.currentMask) {
            maskOverlay.style.display = 'block';
            this.drawMaskOverlay();
        } else {
            maskOverlay.style.display = 'none';
        }
    }

    drawMaskOverlay() {
        if (!this.currentMask) return;

        const canvas = document.getElementById('mask-overlay');
        const ctx = canvas.getContext('2d');
        const maskData = this.currentMask.mask;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const height = maskData.length;
        const width = maskData[0].length;
        
        // Create ImageData for efficient pixel manipulation
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const maskValue = maskData[y][x];
                const index = (y * width + x) * 4;
                
                // Higher mask values (closer to 1) = lower opacity overlay (more visible)
                // Lower mask values (closer to 0) = higher opacity overlay (more hidden)
                const opacity = (1 - maskValue) * 255;
                
                data[index] = 0;     // Red
                data[index + 1] = 0; // Green
                data[index + 2] = 0; // Blue
                data[index + 3] = opacity; // Alpha
            }
        }
        
        // Draw the mask
        ctx.putImageData(imageData, 0, 0);
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

        const xValues = explanation.normalized_mask_sizes;
        const yValues = explanation.score_changes;

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
                maintainAspectRatio: true,
                aspectRatio: 2,
                scales: {
                    x: {
                        display: true,
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
                }
            }
        });
    }

    async setupQuACCurve() {
        await this.updateQuACCurve();
    }

    async updateQuACCurve() {
        try {
            const formData = new FormData(document.getElementById('filter-form'));
            const params = new URLSearchParams();
            
            for (let [key, value] of formData.entries()) {
                if (value) params.append(key, value);
            }

            const response = await fetch(`/api/curve?${params}`);
            const data = await response.json();
            
            if (data.error) {
                console.error('Error loading curve:', data.error);
                return;
            }

            const ctx = document.getElementById('quac-curve').getContext('2d');
            
            // Destroy existing chart if it exists
            if (this.charts.main) {
                this.charts.main.destroy();
            }

            this.charts.main = new Chart(ctx, {
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
                    maintainAspectRatio: true,
                    aspectRatio: 1.5,
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
                                text: 'Score',
                                font: { size: 10 }
                            },
                            ticks: { font: { size: 8 } }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            labels: { font: { size: 8 } }
                        },
                        title: {
                            display: true,
                            text: `QuAC Curve (n=${data.num_samples})`,
                            font: { size: 10 }
                        }
                    }
                }
            });
            
        } catch (error) {
            console.error('Error updating QuAC curve:', error);
        }
    }

    closeViewer() {
        document.getElementById('image-viewer').style.display = 'none';
        document.getElementById('no-explanation').style.display = 'block';
        
        // Clear selection
        document.querySelectorAll('.explanation-item').forEach(item => {
            item.classList.remove('active');
        });
        
        this.currentExplanation = null;
        this.currentMask = null;
    }

    downloadFilteredData(type) {
        const formData = new FormData(document.getElementById('filter-form'));
        const params = new URLSearchParams();
        
        for (let [key, value] of formData.entries()) {
            if (value) params.append(key, value);
        }

        const endpoint = type === 'json' ? 'explanations' : 'images';
        const url = `/api/download/${endpoint}?${params}`;
        
        // Create temporary link to trigger download
        const link = document.createElement('a');
        link.href = url;
        link.download = '';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'block' : 'none';
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new QuACVisualizer();
});