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
        this.sidebarWidth = 300; // Track current sidebar width
        this.sidebarCollapsed = false;
        
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

        let displayHtml = `
            <div><strong>Name:</strong> ${this.reportInfo.name}</div>
            <div><strong>Total Explanations:</strong> ${this.reportInfo.num_explanations}</div>
            <div><strong>Data Score Range:</strong> ${this.reportInfo.score_range.data_min.toFixed(3)} - ${this.reportInfo.score_range.data_max.toFixed(3)}</div>
        `;

        if (this.reportInfo.blinding_enabled) {
            displayHtml += `<div class="text-warning"><strong>⚠️ Blinding Mode Active</strong> - Class information hidden</div>`;
        }

        reportDetails.innerHTML = displayHtml;

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

        if (this.reportInfo.blinding_enabled) {
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

        // Collapsed sidebar indicator toggle
        document.getElementById('sidebar-collapsed').addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Resizable dividers
        this.setupResizableDivider();
        this.setupSidebarResizer();

        // Dual range slider setup
        this.setupDualRangeSlider();

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
        const maskOpacitySlider = document.getElementById('mask-opacity');
        const maskOpacityValue = document.getElementById('mask-opacity-value');
        
        maskOpacitySlider.addEventListener('input', (e) => {
            const opacity = parseInt(e.target.value);
            maskOpacityValue.textContent = `${opacity}%`;
            this.updateMaskOpacity(opacity);
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
                        const maskSlider = document.getElementById('mask-opacity');
                        const currentOpacity = parseInt(maskSlider.value);
                        const newOpacity = currentOpacity === 0 ? 30 : 0;
                        maskSlider.value = newOpacity;
                        document.getElementById('mask-opacity-value').textContent = `${newOpacity}%`;
                        this.updateMaskOpacity(newOpacity);
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

    setupDualRangeSlider() {
        const container = document.querySelector('.dual-range-container');
        const minThumb = document.getElementById('min-thumb');
        const maxThumb = document.getElementById('max-thumb');
        const range = document.getElementById('dual-range');
        const minInput = document.getElementById('min-score');
        const maxInput = document.getElementById('max-score');
        const minValue = document.getElementById('min-score-value');
        const maxValue = document.getElementById('max-score-value');
        
        let minVal = 0;
        let maxVal = 1;
        let isDragging = false;
        let activeThumb = null;
        
        const updateDisplay = () => {
            const containerWidth = container.offsetWidth;
            const minPercent = minVal * 100;
            const maxPercent = maxVal * 100;
            
            // Update thumb positions
            minThumb.style.left = `${minPercent}%`;
            maxThumb.style.left = `${maxPercent}%`;
            
            // Update range bar
            range.style.left = `${minPercent}%`;
            range.style.width = `${maxPercent - minPercent}%`;
            
            // Update hidden inputs
            minInput.value = minVal;
            maxInput.value = maxVal;
            
            // Update display values
            minValue.textContent = minVal.toFixed(2);
            maxValue.textContent = maxVal.toFixed(2);
        };
        
        const getValueFromPosition = (clientX) => {
            const rect = container.getBoundingClientRect();
            const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            return Math.round(percent * 100) / 100; // Round to 2 decimal places
        };
        
        const startDrag = (thumb, e) => {
            isDragging = true;
            activeThumb = thumb;
            thumb.classList.add('dragging');
            
            // Bring active thumb to front
            if (thumb === minThumb) {
                minThumb.style.zIndex = '4';
                maxThumb.style.zIndex = '3';
            } else {
                maxThumb.style.zIndex = '4';
                minThumb.style.zIndex = '3';
            }
            
            e.preventDefault();
        };
        
        const handleDrag = (e) => {
            if (!isDragging || !activeThumb) return;
            
            const newValue = getValueFromPosition(e.clientX);
            
            if (activeThumb === minThumb) {
                minVal = Math.min(newValue, maxVal);
            } else {
                maxVal = Math.max(newValue, minVal);
            }
            
            updateDisplay();
        };
        
        const endDrag = () => {
            if (activeThumb) {
                activeThumb.classList.remove('dragging');
            }
            isDragging = false;
            activeThumb = null;
        };
        
        // Mouse events
        minThumb.addEventListener('mousedown', (e) => startDrag(minThumb, e));
        maxThumb.addEventListener('mousedown', (e) => startDrag(maxThumb, e));
        
        document.addEventListener('mousemove', handleDrag);
        document.addEventListener('mouseup', endDrag);
        
        // Touch events for mobile
        minThumb.addEventListener('touchstart', (e) => startDrag(minThumb, e.touches[0]));
        maxThumb.addEventListener('touchstart', (e) => startDrag(maxThumb, e.touches[0]));
        
        document.addEventListener('touchmove', (e) => {
            if (e.touches[0]) handleDrag(e.touches[0]);
        });
        document.addEventListener('touchend', endDrag);
        
        // Click on track to move nearest thumb
        container.addEventListener('click', (e) => {
            if (isDragging || e.target.classList.contains('dual-thumb')) return;
            
            const clickValue = getValueFromPosition(e.clientX);
            const distToMin = Math.abs(clickValue - minVal);
            const distToMax = Math.abs(clickValue - maxVal);
            
            if (distToMin < distToMax) {
                minVal = Math.min(clickValue, maxVal);
            } else {
                maxVal = Math.max(clickValue, minVal);
            }
            
            updateDisplay();
        });
        
        // Initialize
        updateDisplay();
    }

    resetFilters() {
        // Reset form fields to default values
        document.getElementById('source-class').value = '';
        document.getElementById('target-class').value = '';
        document.getElementById('min-score').value = 0;
        document.getElementById('max-score').value = 1;
        
        // Update display values and visual range
        document.getElementById('min-score-value').textContent = '0.00';
        document.getElementById('max-score-value').textContent = '1.00';
        
        // Reset visual range
        const sliderRange = document.getElementById('slider-range');
        sliderRange.style.left = '0%';
        sliderRange.style.width = '100%';
        
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
        
        // Hide class information if blinding is enabled
        const classDisplay = this.reportInfo.blinding_enabled 
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

    async selectExplanation(explanation) {
        // Update visual selection
        document.querySelectorAll('.explanation-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-explanation-id="${explanation.id}"]`).classList.add('active');

        this.currentExplanation = explanation;
        this.currentMask = null;

        // Show viewer and update metadata
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
        document.getElementById('exp-score').textContent = explanation.score.toFixed(4);
        
        // Hide class information if blinding is enabled
        if (this.reportInfo.blinding_enabled) {
            document.getElementById('exp-source-class').textContent = 'Hidden';
            document.getElementById('exp-target-class').textContent = 'Hidden';
        } else {
            document.getElementById('exp-source-class').textContent = explanation.source_class;
            document.getElementById('exp-target-class').textContent = explanation.target_class;
        }
        
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
        
        if (opacity === 0 || !this.currentMask) {
            maskOverlay.style.display = 'none';
        } else {
            maskOverlay.style.display = 'block';
            this.drawMaskOverlay(opacity);
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
        
        console.log('Mask data shape:', this.currentMask.shape);
        console.log('Mask opacity:', opacity + '%');
        
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
        
        console.log(`Mask dimensions: ${width}x${height}`);
        
        // Create ImageData for the mask (server always returns RGB)
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;
        
        // Convert opacity percentage to alpha value (0-255)
        const alpha = Math.round((opacity / 100) * 255);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const index = (y * width + x) * 4;
                const pixel = maskData[y][x];
                
                data[index] = pixel[0];      // Red
                data[index + 1] = pixel[1];  // Green
                data[index + 2] = pixel[2];  // Blue
                data[index + 3] = alpha;     // Dynamic opacity
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
        // Return to no explanation selected state
        this.setNoExplanationState();
        
        // Clear selection
        document.querySelectorAll('.explanation-item').forEach(item => {
            item.classList.remove('active');
        });
        
        this.currentExplanation = null;
        this.currentMask = null;
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