/**
 * 탭 관리 컴포넌트
 * 메인 탭과 스테이지 탭 관리
 */

class TabManager {
    constructor() {
        this.init();
    }
    
    init() {
        this.initMainTabs();
        this.initStageTabs();
        this.initCodeStageTabs();
        this.restoreTabStates();
    }
    
    /**
     * 메인 탭 초기화 (코드/데이터/로그)
     */
    initMainTabs() {
        const tabButtons = document.querySelectorAll('.tab-header .tab-btn');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;
                this.switchMainTab(tabName);
            });
        });
    }
    
    switchMainTab(tabName) {
        // 탭 버튼 상태 변경
        document.querySelectorAll('.tab-header .tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        
        // 탭 패널 상태 변경
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.panel === tabName);
        });
        
        // 상태 저장
        StateManager.set('activeMainTab', tabName);
    }
    
    /**
     * 스테이지 탭 초기화 (전처리/모델/학습/평가)
     */
    initStageTabs() {
        const stageButtons = document.querySelectorAll('.stage-tabs .stage-tab');
        
        stageButtons.forEach(button => {
            button.addEventListener('click', () => {
                const stage = button.dataset.stage;
                this.switchStageTab(stage);
            });
        });
    }
    
    switchStageTab(stage) {
        // 탭 버튼 상태 변경
        document.querySelectorAll('.stage-tabs .stage-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.stage === stage);
        });
        
        // 스테이지 패널 상태 변경
        document.querySelectorAll('.stage-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.stagePanel === stage);
        });
        
        // 상태 저장
        StateManager.set('activeStage', stage);
    }
    
    /**
     * 코드 뷰 내부 스테이지 탭 초기화
     */
    initCodeStageTabs() {
        const codeStageButtons = document.querySelectorAll('.code-stage-tabs .code-stage-tab');
        
        codeStageButtons.forEach(button => {
            button.addEventListener('click', () => {
                const stage = button.dataset.codeStage;
                this.switchCodeStageTab(stage);
            });
        });
    }
    
    switchCodeStageTab(stage) {
        // 탭 버튼 상태 변경
        document.querySelectorAll('.code-stage-tabs .code-stage-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.codeStage === stage);
        });
        
        // 코드 패널 상태 변경
        document.querySelectorAll('.code-stage-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.codePanel === stage);
        });
        
        // 상태 저장
        StateManager.set('activeCodeStage', stage);
    }
    
    /**
     * 저장된 탭 상태 복원
     */
    restoreTabStates() {
        // 메인 탭 복원
        const activeMainTab = StateManager.get('activeMainTab');
        if (activeMainTab) {
            this.switchMainTab(activeMainTab);
        }
        
        // 스테이지 탭 복원
        const activeStage = StateManager.get('activeStage');
        if (activeStage) {
            this.switchStageTab(activeStage);
        }
        
        // 코드 스테이지 탭 복원
        const activeCodeStage = StateManager.get('activeCodeStage');
        if (activeCodeStage) {
            this.switchCodeStageTab(activeCodeStage);
        }
    }
    
    /**
     * 특정 탭으로 전환
     */
    goToTab(mainTab, stage = null) {
        this.switchMainTab(mainTab);
        
        if (stage) {
            if (mainTab === 'code') {
                this.switchCodeStageTab(stage);
            } else {
                this.switchStageTab(stage);
            }
        }
    }
}

// 컴포넌트 초기화는 app.js에서 수행