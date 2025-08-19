/**
 * components/data.js
 * 데이터 뷰어 컴포넌트
 * 데이터셋 정보 조회 및 시각화
 */

class DataViewer {
    constructor() {
        this.currentDataset = null;
        this.init();
    }
    
    init() {
        // 데이터셋 선택 이벤트
        const datasetSelect = document.getElementById('dataset-select');
        if (datasetSelect) {
            datasetSelect.addEventListener('change', (e) => {
                this.currentDataset = e.target.value;
            });
        }
        
        // 데이터 타입 선택 이벤트
        const dataTypeSelect = document.getElementById('data-type-select');
        if (dataTypeSelect) {
            dataTypeSelect.addEventListener('change', (e) => {
                const showSampleCount = e.target.value === 'sample' || e.target.value === 'images';
                const sampleCount = document.getElementById('sample-count');
                if (sampleCount) {
                    sampleCount.style.display = showSampleCount ? 'block' : 'none';
                }
            });
        }
    }
    
    /**
     * 데이터 정보 로드
     */
    async loadDataInfo() {
        const datasetSelect = document.getElementById('dataset-select');
        const dataTypeSelect = document.getElementById('data-type-select');
        const sampleCount = document.getElementById('sample-count');
        const dataInfo = document.getElementById('data-info');
        
        if (!datasetSelect?.value) {
            alert('데이터셋을 선택해주세요.');
            return;
        }
        
        const dataset = datasetSelect.value;
        const dataType = dataTypeSelect.value;
        const n = sampleCount.value;
        
        // 로딩 표시
        dataInfo.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
        
        try {
            const data = await API.getDatasetInfo(dataset, dataType, n);
            this.displayDataInfo(data, dataType);
        } catch (error) {
            console.error('Failed to load data info:', error);
            dataInfo.innerHTML = `
                <div class="empty-state">
                    <p class="text-muted">데이터를 불러올 수 없습니다.</p>
                    <small>${error.message}</small>
                </div>
            `;
        }
    }
    
    /**
     * 데이터 정보 표시
     */
    displayDataInfo(data, type) {
        const dataInfo = document.getElementById('data-info');
        
        switch (type) {
            case 'shape':
                this.displayShape(data, dataInfo);
                break;
            case 'structure':
                this.displayStructure(data, dataInfo);
                break;
            case 'sample':
                this.displaySample(data, dataInfo);
                break;
            case 'images':
                this.displayImages(data, dataInfo);
                break;
        }
    }
    
    /**
     * 데이터 크기 표시
     */
    displayShape(data, container) {
        container.innerHTML = `
            <div class="data-shape">
                <h3>데이터셋 크기</h3>
                <table class="data-table">
                    <tr>
                        <th>행 (샘플 수)</th>
                        <td>${data.rows.toLocaleString()}</td>
                    </tr>
                    <tr>
                        <th>열 (특징 수)</th>
                        <td>${data.cols.toLocaleString()}</td>
                    </tr>
                </table>
            </div>
        `;
    }
    
    /**
     * 데이터 구조 표시
     */
    displayStructure(data, container) {
        const columns = data.columns.map(col => `
            <tr>
                <td>${col.name}</td>
                <td><code>${col.dtype}</code></td>
            </tr>
        `).join('');
        
        container.innerHTML = `
            <div class="data-structure">
                <h3>데이터 구조</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>컬럼명</th>
                            <th>데이터 타입</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${columns}
                    </tbody>
                </table>
            </div>
        `;
    }
    
    /**
     * 샘플 데이터 표시
     */
    displaySample(data, container) {
        const headers = data.columns.map(col => `<th>${col}</th>`).join('');
        const rows = data.sample.map(row => {
            const cells = row.map(cell => `<td>${this.truncateValue(cell)}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        
        container.innerHTML = `
            <div class="data-sample">
                <h3>샘플 데이터</h3>
                <div style="overflow-x: auto;">
                    <table class="data-table">
                        <thead>
                            <tr>${headers}</tr>
                        </thead>
                        <tbody>
                            ${rows}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    /**
     * 이미지 표시
     */
    displayImages(data, container) {
        const images = data.images.map((img, idx) => `
            <div class="image-item">
                <img src="data:image/png;base64,${img}" alt="Sample ${idx}">
                <div class="image-label">샘플 ${idx + 1}</div>
            </div>
        `).join('');
        
        container.innerHTML = `
            <div class="data-images">
                <h3>이미지 미리보기</h3>
                <div class="image-grid">
                    ${images}
                </div>
            </div>
        `;
    }
    
    /**
     * 값 자르기 (긴 값 표시용)
     */
    truncateValue(value) {
        const str = String(value);
        if (str.length > 20) {
            return str.substring(0, 17) + '...';
        }
        return str;
    }
}

// 전역 DataViewer 인스턴스
window.DataViewer = new DataViewer();