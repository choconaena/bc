/**
 * 상태 관리 모듈
 * 애플리케이션의 전체 상태를 중앙에서 관리
 */

class StateManager {
    constructor() {
        this.state = {
            // 사용자 정보
            uid: this.getCookie('uid') || 'guest',
            
            // 현재 활성 탭/스테이지
            activeMainTab: 'code',
            activeStage: 'pre',
            activeCodeStage: 'pre',
            
            // 폼 데이터
            formData: {},
            
            // 블록 상태
            blockStates: {},
            
            // 실행 상태
            isRunning: false,
            currentProcess: null
        };
        
        this.listeners = new Map();
        this.storageKey = `aiBlocks_${this.state.uid}`;
        
        this.loadFromStorage();
    }
    
    // 쿠키 읽기
    getCookie(name) {
        const match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
        return match ? decodeURIComponent(match[1]) : null;
    }
    
    // 상태 가져오기
    get(path) {
        const keys = path.split('.');
        let value = this.state;
        
        for (const key of keys) {
            value = value[key];
            if (value === undefined) return null;
        }
        
        return value;
    }
    
    // 상태 설정
    set(path, value) {
        const keys = path.split('.');
        const lastKey = keys.pop();
        let target = this.state;
        
        for (const key of keys) {
            if (!target[key]) target[key] = {};
            target = target[key];
        }
        
        const oldValue = target[lastKey];
        target[lastKey] = value;
        
        // 변경 알림
        this.notify(path, value, oldValue);
        
        // 스토리지 저장
        this.saveToStorage();
    }
    
    // 리스너 등록
    on(path, callback) {
        if (!this.listeners.has(path)) {
            this.listeners.set(path, new Set());
        }
        this.listeners.get(path).add(callback);
        
        // 등록 해제 함수 반환
        return () => {
            const listeners = this.listeners.get(path);
            if (listeners) {
                listeners.delete(callback);
            }
        };
    }
    
    // 변경 알림
    notify(path, newValue, oldValue) {
        // 정확한 경로 리스너
        const exactListeners = this.listeners.get(path);
        if (exactListeners) {
            exactListeners.forEach(callback => {
                callback(newValue, oldValue, path);
            });
        }
        
        // 와일드카드 리스너 (예: 'formData.*')
        const wildcardPath = path.split('.').slice(0, -1).join('.') + '.*';
        const wildcardListeners = this.listeners.get(wildcardPath);
        if (wildcardListeners) {
            wildcardListeners.forEach(callback => {
                callback(newValue, oldValue, path);
            });
        }
    }
    
    // 로컬 스토리지에 저장
    saveToStorage() {
        try {
            const dataToSave = {
                activeMainTab: this.state.activeMainTab,
                activeStage: this.state.activeStage,
                activeCodeStage: this.state.activeCodeStage,
                formData: this.state.formData,
                blockStates: this.state.blockStates
            };
            
            localStorage.setItem(this.storageKey, JSON.stringify(dataToSave));
        } catch (e) {
            console.warn('Failed to save state to localStorage:', e);
        }
    }
    
    // 로컬 스토리지에서 로드
    loadFromStorage() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (saved) {
                const data = JSON.parse(saved);
                Object.assign(this.state, data);
            }
        } catch (e) {
            console.warn('Failed to load state from localStorage:', e);
        }
    }
    
    // 서버에서 받은 초기 상태 로드
    loadServerState() {
        const stateElement = document.getElementById('app-state');
        if (stateElement) {
            try {
                const serverState = JSON.parse(stateElement.textContent);
                this.state.formData = { ...this.state.formData, ...serverState };
                this.saveToStorage();
            } catch (e) {
                console.warn('Failed to parse server state:', e);
            }
        }
    }
    
    // 폼 데이터 업데이트
    updateFormData(name, value) {
        this.set(`formData.${name}`, value);
    }
    
    // 폼 데이터 가져오기
    getFormData() {
        return { ...this.state.formData };
    }
    
    // 블록 상태 업데이트
    updateBlockState(blockId, active) {
        this.set(`blockStates.${blockId}`, active);
    }
    
    // 블록 상태 가져오기
    getBlockState(blockId) {
        return this.state.blockStates[blockId] !== false;
    }
    
    // 전체 상태 리셋
    reset() {
        this.state = {
            uid: this.state.uid,
            activeMainTab: 'code',
            activeStage: 'pre',
            activeCodeStage: 'pre',
            formData: {},
            blockStates: {},
            isRunning: false,
            currentProcess: null
        };
        
        this.saveToStorage();
        this.notify('*', this.state, null);
    }
}

// 전역 상태 관리자 생성
window.StateManager = new StateManager();