/**
 * 블록 관리 컴포넌트
 * 블록의 활성화/비활성화 및 상태 관리
 */

class BlockManager {
    constructor() {
        this.blocks = new Map();
        this.init();
    }
    
    init() {
        // 모든 블록 초기화
        document.querySelectorAll('.block').forEach(block => {
            this.initBlock(block);
        });
        
        // 이벤트 위임으로 블록 클릭 처리
        document.addEventListener('click', this.handleBlockClick.bind(this));
        
        // 폼 제출 시 처리
        const form = document.getElementById('pipeline-form');
        if (form) {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        }
        
        // 조건부 필드 초기화
        this.initConditionalFields();
    }
    
    initBlock(block) {
        const blockId = block.id;
        const isRequired = block.classList.contains('block-required');
        const savedState = StateManager.getBlockState(blockId);
        
        // 초기 상태 설정
        const isActive = isRequired || savedState !== false;
        this.setBlockActive(block, isActive);
        
        // 상태 저장
        this.blocks.set(blockId, {
            element: block,
            required: isRequired,
            active: isActive
        });
    }
    
    handleBlockClick(event) {
        // 블록 클릭 확인
        const block = event.target.closest('.block');
        if (!block) return;
        
        // 내부 컨트롤 클릭은 무시
        const isControl = event.target.matches('input, select, textarea, label, button');
        if (isControl) return;
        
        // 필수 블록은 토글 불가
        if (block.classList.contains('block-required')) return;
        
        // 상태 토글
        this.toggleBlock(block);
    }
    
    toggleBlock(block) {
        const blockData = this.blocks.get(block.id);
        if (!blockData || blockData.required) return;
        
        const newState = !blockData.active;
        this.setBlockActive(block, newState);
        
        // 상태 업데이트
        blockData.active = newState;
        StateManager.updateBlockState(block.id, newState);
    }
    
    setBlockActive(block, active) {
        block.dataset.active = active;
        
        // 필수 블록이 아닌 경우 내부 컨트롤 활성화/비활성화
        if (!block.classList.contains('block-required')) {
            const controls = block.querySelectorAll('input, select, textarea');
            controls.forEach(control => {
                control.disabled = !active;
                
                // 비활성화 시 validation 속성 제거
                if (!active) {
                    control.removeAttribute('required');
                    control.dataset.originalName = control.name;
                    control.removeAttribute('name');
                } else if (control.dataset.originalName) {
                    control.name = control.dataset.originalName;
                }
            });
        }
    }
    
    handleFormSubmit(event) {
        // 모든 활성 블록의 컨트롤 활성화
        this.blocks.forEach((blockData, blockId) => {
            if (blockData.active) {
                const controls = blockData.element.querySelectorAll('input, select, textarea');
                controls.forEach(control => {
                    control.disabled = false;
                    if (control.dataset.originalName) {
                        control.name = control.dataset.originalName;
                    }
                });
            }
        });
        
        // 폼 데이터 저장
        const formData = new FormData(event.target);
        const data = {};
        
        for (const [key, value] of formData.entries()) {
            if (data[key]) {
                if (Array.isArray(data[key])) {
                    data[key].push(value);
                } else {
                    data[key] = [data[key], value];
                }
            } else {
                data[key] = value;
            }
        }
        
        StateManager.state.formData = data;
        StateManager.saveToStorage();
    }
    
    initConditionalFields() {
        // 테스트 데이터 사용 여부
        const isTestSelect = document.getElementById('is_test');
        if (isTestSelect) {
            const updateTestFields = () => {
                const useTest = isTestSelect.value === 'true';
                const testDiv = document.getElementById('test-div');
                const ratioDiv = document.getElementById('ratio-div');
                
                if (testDiv) testDiv.classList.toggle('hidden', !useTest);
                if (ratioDiv) ratioDiv.classList.toggle('hidden', useTest);
            };
            
            isTestSelect.addEventListener('change', updateTestFields);
            updateTestFields();
        }
        
        // 잘못된 라벨 필터링
        const dropBadCheck = document.getElementById('drop_bad');
        if (dropBadCheck) {
            const updateBadParams = () => {
                const params = document.getElementById('drop_bad_params');
                if (params) {
                    params.classList.toggle('hidden', !dropBadCheck.checked);
                }
            };
            
            dropBadCheck.addEventListener('change', updateBadParams);
            updateBadParams();
        }
        
        // Conv2 사용 여부
        const useConv2Check = document.querySelector('input[name="use_conv2"]');
        if (useConv2Check) {
            const updateConv2Fields = () => {
                const conv2Fields = useConv2Check.closest('.block').querySelector('.conditional-field');
                if (conv2Fields) {
                    conv2Fields.classList.toggle('hidden', !useConv2Check.checked);
                }
            };
            
            useConv2Check.addEventListener('change', updateConv2Fields);
            updateConv2Fields();
        }
        
        // 드롭아웃 사용 여부
        const useDropoutCheck = document.querySelector('input[name="use_dropout"]');
        if (useDropoutCheck) {
            const updateDropoutFields = () => {
                const dropoutFields = useDropoutCheck.closest('.block').querySelector('.conditional-field');
                if (dropoutFields) {
                    dropoutFields.classList.toggle('hidden', !useDropoutCheck.checked);
                }
            };
            
            useDropoutCheck.addEventListener('change', updateDropoutFields);
            updateDropoutFields();
        }
    }
    
    // 폼 데이터 복원
    restoreFormData() {
        const formData = StateManager.getFormData();
        
        Object.entries(formData).forEach(([name, value]) => {
            const elements = document.querySelectorAll(`[name="${name}"]`);
            
            elements.forEach(element => {
                const type = element.type || element.tagName.toLowerCase();
                
                if (type === 'checkbox') {
                    if (Array.isArray(value)) {
                        element.checked = value.includes(element.value);
                    } else {
                        element.checked = value === 'on' || value === element.value;
                    }
                } else if (type === 'radio') {
                    element.checked = element.value === value;
                } else if (element.tagName.toLowerCase() === 'select' && element.multiple) {
                    const values = Array.isArray(value) ? value : [value];
                    Array.from(element.options).forEach(option => {
                        option.selected = values.includes(option.value);
                    });
                } else {
                    element.value = Array.isArray(value) ? value[0] : value;
                }
                
                // change 이벤트 발생
                element.dispatchEvent(new Event('change', { bubbles: true }));
            });
        });
    }
}

// 컴포넌트 초기화는 app.js에서 수행