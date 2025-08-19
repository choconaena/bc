/**
 * 로그 관리 컴포넌트
 * 실행 로그 스트리밍 및 표시
 */

class LogManager {
    constructor() {
        this.logView = null;
        this.autoScroll = true;
        this.currentStage = null;
        this.init();
    }
    
    init() {
        this.logView = document.getElementById('log-view');
        this.initControls();
        this.initRunButtons();
    }
    
    /**
     * 로그 컨트롤 초기화
     */
    initControls() {
        // 로그 지우기 버튼
        const clearBtn = document.getElementById('log-clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearLog();
            });
        }
        
        // 자동 스크롤 체크박스
        const autoScrollCheck = document.getElementById('log-autoscroll');
        if (autoScrollCheck) {
            autoScrollCheck.addEventListener('change', (e) => {
                this.autoScroll = e.target.checked;
            });
            this.autoScroll = autoScrollCheck.checked;
        }
    }
    
    /**
     * 실행 버튼 초기화
     */
    initRunButtons() {
        // 모든 실행 버튼에 이벤트 바인딩
        document.querySelectorAll('[data-run]').forEach(button => {
            button.addEventListener('click', async (e) => {
                e.preventDefault();
                const stage = button.dataset.run;
                await this.runStage(stage);
            });
        });
    }
    
    /**
     * 스테이지 실행
     */
    async runStage(stage) {
        // 이미 실행 중인지 확인
        if (StateManager.get('isRunning')) {
            if (!confirm('이미 실행 중인 프로세스가 있습니다. 중지하고 새로 실행하시겠습니까?')) {
                return;
            }
            this.stopStream();
        }
        
        try {
            // 실행 상태 설정
            StateManager.set('isRunning', true);
            StateManager.set('currentProcess', stage);
            
            // 로그 탭으로 전환
            const tabManager = window.app?.tabManager;
            if (tabManager) {
                tabManager.goToTab('log');
            }
            
            // 로그 초기화
            this.clearLog();
            this.appendLog(`[시작] ${stage} 실행 시작...`);
            
            // API 호출
            const result = await API.runCode(stage);
            
            if (result.ok) {
                // 로그 스트리밍 시작
                this.startStream(stage);
            } else {
                throw new Error(result.error || '실행 실패');
            }
            
        } catch (error) {
            this.appendLog(`[오류] ${error.message}`, 'error');
            StateManager.set('isRunning', false);
            StateManager.set('currentProcess', null);
        }
    }
    
    /**
     * 로그 스트리밍 시작
     */
    startStream(stage) {
        this.currentStage = stage;
        
        API.startLogStream(
            stage,
            (message) => {
                this.appendLog(message);
            },
            (error) => {
                this.appendLog('[스트림 종료]', 'info');
                StateManager.set('isRunning', false);
                StateManager.set('currentProcess', null);
            }
        );
    }
    
    /**
     * 로그 스트리밍 중지
     */
    stopStream() {
        API.stopLogStream();
        this.currentStage = null;
        StateManager.set('isRunning', false);
        StateManager.set('currentProcess', null);
    }
    
    /**
     * 로그 추가
     */
    appendLog(message, type = 'normal') {
        if (!this.logView) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const logLine = document.createElement('div');
        logLine.className = `log-line log-${type}`;
        
        // ANSI 컬러 코드 처리 (간단한 버전)
        message = this.parseAnsiColors(message);
        
        logLine.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;
        this.logView.appendChild(logLine);
        
        // 자동 스크롤
        if (this.autoScroll) {
            this.logView.scrollTop = this.logView.scrollHeight;
        }
    }
    
    /**
     * ANSI 컬러 코드 파싱
     */
    parseAnsiColors(text) {
        // 간단한 ANSI 코드 변환
        const colorMap = {
            '30': 'black',
            '31': 'red',
            '32': 'green',
            '33': 'yellow',
            '34': 'blue',
            '35': 'magenta',
            '36': 'cyan',
            '37': 'white',
            '90': 'gray'
        };
        
        return text.replace(/\x1b\[(\d+)m/g, (match, code) => {
            const color = colorMap[code];
            if (color) {
                return `<span style="color: ${color}">`;
            } else if (code === '0') {
                return '</span>';
            }
            return '';
        });
    }
    
    /**
     * 로그 지우기
     */
    clearLog() {
        if (this.logView) {
            this.logView.innerHTML = '';
        }
    }
    
    /**
     * 로그 다운로드
     */
    downloadLog() {
        if (!this.logView) return;
        
        const logText = this.logView.innerText;
        const blob = new Blob([logText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `log_${this.currentStage || 'output'}_${Date.now()}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
    }
}

// 컴포넌트 초기화는 app.js에서 수행