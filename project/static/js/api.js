/**
 * API 통신 모듈
 * 백엔드와의 모든 통신을 담당
 */

class APIClient {
    constructor() {
        this.baseURL = '';
        this.headers = {
            'Content-Type': 'application/json'
        };
    }
    
    /**
     * 코드 변환 요청
     */
    async convertCode(stage = 'all', formData = null) {
        const data = formData || StateManager.getFormData();
        
        // FormData로 변환 (multipart/form-data)
        const fd = new FormData();
        fd.append('stage', stage);
        
        Object.entries(data).forEach(([key, value]) => {
            if (Array.isArray(value)) {
                value.forEach(v => fd.append(key, v));
            } else if (value !== null && value !== undefined) {
                fd.append(key, value);
            }
        });
        
        try {
            const response = await fetch('/convert', {
                method: 'POST',
                body: fd
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.text();
        } catch (error) {
            console.error('Convert error:', error);
            throw error;
        }
    }
    
    /**
     * 코드 실행 요청
     */
    async runCode(stage) {
        try {
            const response = await fetch(`/run/${stage}`, {
                method: 'POST',
                headers: this.headers
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Execution failed');
            }
            
            return data;
        } catch (error) {
            console.error('Run error:', error);
            throw error;
        }
    }
    
    /**
     * 데이터셋 정보 조회
     */
    async getDatasetInfo(file, type = 'shape', n = 5) {
        try {
            const params = new URLSearchParams({
                file: file,
                type: type,
                n: n
            });
            
            const response = await fetch(`/data-info?${params}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Data info error:', error);
            throw error;
        }
    }
    
    /**
     * 로그 스트리밍 시작
     */
    startLogStream(stage, onMessage, onError) {
        // 기존 스트림 종료
        this.stopLogStream();
        
        // 새 스트림 시작
        this.eventSource = new EventSource(`/logs/stream?stage=${stage}`);
        
        this.eventSource.onmessage = (event) => {
            if (onMessage) onMessage(event.data);
        };
        
        this.eventSource.onerror = (error) => {
            if (onError) onError(error);
            this.stopLogStream();
        };
        
        return this.eventSource;
    }
    
    /**
     * 로그 스트리밍 중지
     */
    stopLogStream() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
    
    /**
     * 코드 다운로드
     */
    downloadCode(stage) {
        const link = document.createElement('a');
        link.href = `/download/${stage}`;
        link.download = true;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// 전역 API 클라이언트 생성
window.API = new APIClient();