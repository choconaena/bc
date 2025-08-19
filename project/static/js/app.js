/**
 * 메인 애플리케이션
 * 모든 컴포넌트를 초기화하고 조정
 */

class App {
    constructor() {
        this.components = {};
        this.isReady = false;
    }
    
    /**
     * 애플리케이션 초기화
     */
    async init() {
        try {
            // DOM이 준비될 때까지 대기
            if (document.readyState === 'loading') {
                await new Promise(resolve => {
                    document.addEventListener('DOMContentLoaded', resolve);
                });
            }
            
            // 서버 상태 로드
            StateManager.loadServerState();
            
            // 컴포넌트 초기화
            this.initComponents();
            
            // 이벤트 리스너 설정
            this.setupEventListeners();
            
            // 폼 데이터 복원
            this.restoreFormData();
            
            // 초기화 완료
            this.isReady = true;
            console.log('✅ AI Block Coding App initialized');
            
        } catch (error) {
            console.error('❌ App initialization failed:', error);
        }
    }
    
    /**
     * 컴포넌트 초기화
     */
    initComponents() {
        // 탭 매니저
        this.tabManager = new TabManager();
        this.components.tabs = this.tabManager;
        
        // 블록 매니저
        this.blockManager = new BlockManager();
        this.components.blocks = this.blockManager;
        
        // 로그 매니저
        this.logManager = new LogManager();
        this.components.logs = this.logManager;
        
        // 데이터 뷰어 (필요시 추가)
        // this.dataViewer = new DataViewer();
        // this.components.data = this.dataViewer;
    }
    
    /**
     * 이벤트 리스너 설정
     */
    setupEventListeners() {
        // 폼 입력 변경 감지
        const form = document.getElementById('pipeline-form');
        if (form) {
            form.addEventListener('input', this.handleFormInput.bind(this));
            form.addEventListener('change', this.handleFormChange.bind(this));
        }
        
        // 윈도우 이벤트
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
        
        // 키보드 단축키
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }
    
    /**
     * 폼 입력 처리
     */
    handleFormInput(event) {
        const { name, value } = event.target;
        if (name) {
            StateManager.updateFormData(name, value);
        }
    }
    
    /**
     * 폼 변경 처리
     */
    handleFormChange(event) {
        const { name, type, checked, value } = event.target;
        
        if (type === 'checkbox') {
            // 체크박스는 특별 처리
            const currentValue = StateManager.get(`formData.${name}`);
            
            if (event.target.closest('.checkbox-group')) {
                // 체크박스 그룹인 경우
                let values = Array.isArray(currentValue) ? [...currentValue] : [];
                
                if (checked) {
                    if (!values.includes(value)) {
                        values.push(value);
                    }
                } else {
                    values = values.filter(v => v !== value);
                }
                
                StateManager.updateFormData(name, values);
            } else {
                // 단일 체크박스
                StateManager.updateFormData(name, checked ? 'on' : '');
            }
        } else {
            StateManager.updateFormData(name, value);
        }
    }
    
    /**
     * 페이지 이탈 전 처리
     */
    handleBeforeUnload(event) {
        // 실행 중인 프로세스가 있으면 경고
        if (StateManager.get('isRunning')) {
            const message = '실행 중인 프로세스가 있습니다. 페이지를 벗어나시겠습니까?';
            event.preventDefault();
            event.returnValue = message;
            return message;
        }
    }
    
    /**
     * 키보드 단축키 처리
     */
    handleKeyDown(event) {
        // Ctrl/Cmd + S: 저장 (폼 제출)
        if ((event.ctrlKey || event.metaKey) && event.key === 's') {
            event.preventDefault();
            this.saveAll();
        }
        
        // Ctrl/Cmd + Enter: 현재 스테이지 실행
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            this.runCurrentStage();
        }
        
        // ESC: 실행 중지
        if (event.key === 'Escape' && StateManager.get('isRunning')) {
            this.stopExecution();
        }
    }
    
    /**
     * 폼 데이터 복원
     */
    restoreFormData() {
        if (this.blockManager) {
            this.blockManager.restoreFormData();
        }
    }
    
    /**
     * 전체 저장
     */
    async saveAll() {
        try {
            const form = document.getElementById('pipeline-form');
            if (form) {
                // stage=all로 설정하고 제출
                const stageInput = form.querySelector('input[name="stage"]') || 
                                 document.createElement('input');
                stageInput.type = 'hidden';
                stageInput.name = 'stage';
                stageInput.value = 'all';
                form.appendChild(stageInput);
                
                form.submit();
            }
        } catch (error) {
            console.error('Save failed:', error);
        }
    }
    
    /**
     * 현재 스테이지 실행
     */
    async runCurrentStage() {
        const currentStage = StateManager.get('activeStage');
        if (currentStage && this.logManager) {
            await this.logManager.runStage(currentStage);
        }
    }
    
    /**
     * 실행 중지
     */
    stopExecution() {
        if (this.logManager) {
            this.logManager.stopStream();
        }
    }
    
    /**
     * 컴포넌트 접근자
     */
    getComponent(name) {
        return this.components[name];
    }
    
    /**
     * 상태 리셋
     */
    reset() {
        if (confirm('모든 설정을 초기화하시겠습니까?')) {
            StateManager.reset();
            location.reload();
        }
    }
}

// 전역 앱 인스턴스 생성 및 초기화
window.app = new App();
window.app.init();