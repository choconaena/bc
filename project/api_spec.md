# AI 블록코딩 API 명세서

## 1. 개요

### 1.1 시스템 아키텍처
- **Frontend**: HTML/CSS/JS (블록 UI, 코드 미리보기, 데이터 시각화)
- **Backend**: Flask (코드 생성, 실행, 로그 스트리밍)
- **Storage**: 사용자별 워크스페이스 (UUID 기반 격리)

### 1.2 핵심 개념
- **블록(Block)**: AI 파이프라인의 각 단계를 나타내는 UI 컴포넌트
- **스니펫(Snippet)**: 블록 설정에 따라 생성되는 Python 코드 조각
- **워크스페이스(Workspace)**: 사용자별 작업 공간 (코드, 데이터, 로그)
- **스테이지(Stage)**: 파이프라인 단계 (pre/model/train/eval)

---

## 2. API 엔드포인트

### 2.1 페이지 렌더링

#### GET `/`
**설명**: 루트 접근 시 `/app`으로 리다이렉트  
**응답**: 302 Redirect to `/app`  
**쿠키**: `uid` 설정 (없을 경우 새로 생성)

#### GET `/app`
**설명**: 메인 애플리케이션 페이지  
**응답**: HTML (index.html)  
**템플릿 변수**:
```python
{
    "options": ["mnist_train.csv", ...],  # dataset/ 폴더의 CSV 목록
    "form_state": {...},                  # 저장된 폼 상태 (JSON)
    "snippet_pre": "...",                 # 전처리 코드
    "snippet_model": "...",               # 모델 코드
    "snippet_train": "...",               # 학습 코드
    "snippet_eval": "..."                 # 평가 코드
}
```

---

### 2.2 코드 생성 및 변환

#### POST `/convert`
**설명**: 블록 설정을 Python 코드로 변환  
**요청 본문** (form-data):

##### 공통 파라미터
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| stage | string | ✓ | 변환 대상: "pre", "model", "train", "eval", "all" |

##### 전처리 블록 파라미터 (stage=pre)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| dataset | string | - | 훈련 데이터 CSV 파일명 |
| is_test | string | "false" | 테스트 데이터 사용 여부 ("true"/"false") |
| testdataset | string | - | 테스트 데이터 CSV 파일명 (is_test=true일 때) |
| a | number | 80 | 훈련 데이터 비율 (%) (is_test=false일 때) |
| drop_na | checkbox | - | 결측치 제거 여부 |
| drop_bad | checkbox | - | 잘못된 라벨 제거 여부 |
| min_label | number | 0 | 최소 라벨값 (drop_bad=true일 때) |
| max_label | number | 9 | 최대 라벨값 (drop_bad=true일 때) |
| split_xy | checkbox | - | X/y 분리 여부 |
| resize_n | number | 28 | 이미지 리사이즈 크기 (n×n) |
| augment_method | string | - | 증강 방법: "rotate", "hflip", "vflip", "translate" |
| augment_param | number | - | 증강 파라미터 (각도/픽셀) |
| normalize | string | - | 정규화 방법: "0-1", "-1-1" |

##### 모델 설계 블록 파라미터 (stage=model)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| input_w | number | 28 | 입력 이미지 너비 |
| input_h | number | 28 | 입력 이미지 높이 |
| input_c | number | 1 | 입력 채널 수 (1:흑백, 3:컬러) |
| conv1_filters | number | 32 | Conv1 필터 수 |
| conv1_kernel | number | 3 | Conv1 커널 크기 |
| conv1_padding | string | "valid" | Conv1 패딩: "same", "valid" |
| conv1_activation | string | "relu" | Conv1 활성함수 |
| pool1_type | string | - | Pool1 종류: "max", "avg" |
| pool1_size | number | 2 | Pool1 크기 |
| pool1_stride | number | 2 | Pool1 스트라이드 |
| use_conv2 | checkbox | - | Conv2 사용 여부 |
| conv2_filters | number | 64 | Conv2 필터 수 |
| conv2_kernel | number | 3 | Conv2 커널 크기 |
| conv2_activation | string | "relu" | Conv2 활성함수 |
| use_dropout | checkbox | - | 드롭아웃 사용 여부 |
| dropout_p | number | 0.25 | 드롭아웃 비율 |
| dense_units | number | 128 | Dense 레이어 유닛 수 |
| dense_activation | string | "relu" | Dense 활성함수 |
| num_classes | number | 10 | 출력 클래스 수 |

##### 학습 블록 파라미터 (stage=train)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| loss_method | string | "CrossEntropy" | 손실함수: "CrossEntropy", "MSE" |
| optimizer_method | string | "Adam" | 옵티마이저: "Adam", "SGD", "RMSprop" |
| learning_rate | number | 0.00001 | 학습률 |
| epochs | number | 10 | 에폭 수 |
| batch_size | number | 64 | 배치 크기 |
| patience | number | 3 | Early stopping patience |

##### 평가 블록 파라미터 (stage=eval)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| metrics | checkbox[] | ["accuracy"] | 평가 메트릭 (복수 선택) |
| average | string | "macro" | 평균 방식 |
| topk_k | number | 3 | Top-K의 K값 |
| show_classification_report | checkbox | - | 분류 리포트 출력 |
| show_confusion_matrix | checkbox | - | 혼동 행렬 출력 |
| cm_normalize | checkbox | - | 혼동 행렬 정규화 |
| viz_samples | number | 10 | 시각화할 예측 샘플 수 |
| viz_mis | number | 5 | 시각화할 오분류 샘플 수 |
| eval_batch | number | 128 | 평가 배치 크기 |
| num_classes | number | 10 | 클래스 수 |
| class_names | string | - | 클래스 이름 (쉼표 구분) |
| force_cpu | checkbox | - | CPU 강제 사용 |

**응답**: HTML (index.html) - 생성된 코드와 함께 재렌더링

---

### 2.3 코드 실행 및 로그

#### POST `/run/<stage>`
**설명**: 특정 스테이지 코드 실행  
**경로 파라미터**: 
- `stage`: "pre", "model", "train", "eval"

**응답**:
```json
{
    "ok": true
}
```
또는
```json
{
    "error": "error message"
}
```

#### GET `/logs/stream`
**설명**: Server-Sent Events로 실행 로그 스트리밍  
**쿼리 파라미터**:
- `stage`: 로그를 볼 스테이지 (기본값: "train")

**응답**: text/event-stream
```
data: [pre][2025-01-01 12:00:00] Processing started...
data: [pre][2025-01-01 12:00:01] Loading data...
```

---

### 2.4 데이터 정보 조회

#### GET `/data-info`
**설명**: CSV 데이터셋 정보 조회  
**쿼리 파라미터**:
- `file`: CSV 파일명
- `type`: 정보 유형 ("shape", "structure", "sample", "images")
- `n`: 샘플/이미지 개수 (type=sample/images일 때)

**응답 예시**:

##### type=shape
```json
{
    "rows": 60000,
    "cols": 785
}
```

##### type=structure
```json
{
    "columns": [
        {"name": "label", "dtype": "int64"},
        {"name": "pixel0", "dtype": "int64"},
        ...
    ]
}
```

##### type=sample
```json
{
    "columns": ["label", "pixel0", ...],
    "sample": [
        [5, 0, 0, ...],
        [0, 0, 0, ...],
        ...
    ]
}
```

##### type=images
```json
{
    "images": [
        "data:image/png;base64,iVBORw0KGgo...",
        ...
    ]
}
```

---

### 2.5 파일 다운로드

#### GET `/download/<stage>`
**설명**: 생성된 코드 파일 다운로드  
**경로 파라미터**:
- `stage`: "pre", "model", "train", "eval", "all"

**응답**: 
- stage="pre": preprocessing.py
- stage="model": model.py
- stage="train": training.py
- stage="eval": evaluation.py
- stage="all": workspace_<uid>_<timestamp>.zip

---

## 3. 파일 구조 및 생성 규칙

### 3.1 워크스페이스 구조
```
workspace/<uid>/
├── preprocessing.py    # 전처리 코드
├── model.py           # 모델 정의
├── training.py        # 학습 코드
├── evaluation.py      # 평가 코드
├── inputs_pre.json    # 전처리 입력값 저장
├── inputs_model.json  # 모델 입력값 저장
├── inputs_train.json  # 학습 입력값 저장
├── inputs_eval.json   # 평가 입력값 저장
├── data/
│   └── dataset.pt     # 전처리된 데이터
└── artifacts/
    ├── best_model.pth # 학습된 모델
    ├── confusion_matrix.png
    ├── samples.png
    └── misclassified.png
```

### 3.2 코드 생성 규칙

#### 전처리 (preprocessing.py)
1. 데이터 로드 (필수)
2. 결측치 제거 (선택)
3. 라벨 필터링 (선택)
4. X/y 분리 (선택)
5. 이미지 리사이즈 (선택)
6. 데이터 증강 (선택)
7. 정규화 (선택)
8. 데이터 저장 (필수)

#### 모델 (model.py)
1. CNN 클래스 정의
2. Conv 레이어 (1~2개)
3. Pooling 레이어
4. Dropout (선택)
5. Flatten
6. Dense 레이어
7. Output 레이어

#### 학습 (training.py)
1. 데이터 로드
2. 모델 생성
3. 손실함수/옵티마이저 설정
4. 학습 루프
5. Early stopping
6. 체크포인트 저장

#### 평가 (evaluation.py)
1. 데이터/모델 로드
2. 추론
3. 메트릭 계산
4. 리포트 생성
5. 시각화

---

## 4. 에러 처리

### HTTP 상태 코드
- 200: 성공
- 302: 리다이렉트
- 400: 잘못된 요청 (unknown stage)
- 404: 파일 없음
- 500: 서버 에러

### 에러 응답 형식
```json
{
    "error": "error description"
}
```

---

## 5. 보안 고려사항

1. **경로 순회 방지**: 모든 파일 접근은 화이트리스트 기반
2. **세션 격리**: UUID 기반 사용자별 워크스페이스
3. **쿠키 보안**: httponly=True, samesite="Lax"
4. **입력 검증**: 모든 숫자 입력에 min/max 제한
5. **프로세스 격리**: 각 실행은 독립 프로세스

---

## 6. 확장 계획

### 추가 예정 기능
- [ ] RNN/LSTM 모델 지원
- [ ] 다양한 데이터셋 형식 지원 (이미지 폴더, JSON)
- [ ] 하이퍼파라미터 튜닝
- [ ] 모델 비교/앙상블
- [ ] 실시간 학습 그래프
- [ ] 코드 export (Jupyter Notebook)
- [ ] 클라우드 학습 연동

### API 버전 관리
- 현재 버전: v1.0
- 버전 업데이트 시 `/api/v2/` 경로 사용 예정





# AI 블록코딩 API 상세 명세서

## 목차
1. [기본 정보](#1-기본-정보)
2. [인증 및 세션](#2-인증-및-세션)
3. [API 엔드포인트 상세](#3-api-엔드포인트-상세)
4. [WebSocket/SSE 실시간 통신](#4-websocketsse-실시간-통신)
5. [에러 코드](#5-에러-코드)
6. [테스트 시나리오](#6-테스트-시나리오)

---

## 1. 기본 정보

### 1.1 서버 정보
- **Base URL**: `http://127.0.0.1:9011`
- **API Version**: v1.0
- **Content-Type**: `application/json` (API), `multipart/form-data` (폼 제출)
- **Encoding**: UTF-8

### 1.2 쿠키 정책
- **Name**: `uid`
- **Type**: UUID (32자 hex)
- **HttpOnly**: true
- **SameSite**: Lax
- **유효기간**: 세션

---

## 2. 인증 및 세션

### 2.1 UID 생성 및 관리
```bash
# 첫 접속 시 UID 자동 생성
curl -c cookies.txt http://127.0.0.1:9011/

# 쿠키 확인
cat cookies.txt
# 127.0.0.1	FALSE	/	FALSE	0	uid	a1b2c3d4e5f6...
```

### 2.2 세션 유지
```bash
# 쿠키를 사용한 요청
curl -b cookies.txt http://127.0.0.1:9011/app
```

---

## 3. API 엔드포인트 상세

### 3.1 페이지 렌더링

#### GET `/`
루트 페이지 - 앱으로 리다이렉트

**cURL 예제:**
```bash
curl -v http://127.0.0.1:9011/
# 302 Found -> /app
```

#### GET `/app`
메인 애플리케이션 페이지

**cURL 예제:**
```bash
curl -b cookies.txt http://127.0.0.1:9011/app
```

**응답 예제:**
```html
<!DOCTYPE html>
<html>
<head>...</head>
<body>
    <script id="app-state" type="application/json">
    {
        "dataset": "mnist_train.csv",
        "is_test": "false",
        "a": "80",
        ...
    }
    </script>
    ...
</body>
</html>
```

---

### 3.2 코드 생성

#### POST `/convert`
블록 설정을 Python 코드로 변환

**cURL 예제 - 전처리 코드 생성:**
```bash
curl -X POST http://127.0.0.1:9011/convert \
  -b cookies.txt \
  -F "stage=pre" \
  -F "dataset=mnist_train.csv" \
  -F "is_test=false" \
  -F "a=80" \
  -F "drop_na=on" \
  -F "split_xy=on" \
  -F "resize_n=28" \
  -F "normalize=0-1"
```

**cURL 예제 - 모델 코드 생성:**
```bash
curl -X POST http://127.0.0.1:9011/convert \
  -b cookies.txt \
  -F "stage=model" \
  -F "input_w=28" \
  -F "input_h=28" \
  -F "input_c=1" \
  -F "conv1_filters=32" \
  -F "conv1_kernel=3" \
  -F "conv1_padding=same" \
  -F "conv1_activation=relu" \
  -F "pool1_type=max" \
  -F "pool1_size=2" \
  -F "dense_units=128" \
  -F "num_classes=10"
```

**cURL 예제 - 학습 코드 생성:**
```bash
curl -X POST http://127.0.0.1:9011/convert \
  -b cookies.txt \
  -F "stage=train" \
  -F "loss_method=CrossEntropy" \
  -F "optimizer_method=Adam" \
  -F "learning_rate=0.001" \
  -F "epochs=10" \
  -F "batch_size=64" \
  -F "patience=3"
```

**cURL 예제 - 평가 코드 생성:**
```bash
curl -X POST http://127.0.0.1:9011/convert \
  -b cookies.txt \
  -F "stage=eval" \
  -F "metrics=accuracy" \
  -F "metrics=precision" \
  -F "metrics=recall" \
  -F "show_classification_report=on" \
  -F "show_confusion_matrix=on" \
  -F "viz_samples=10"
```

**cURL 예제 - 전체 파이프라인 생성:**
```bash
curl -X POST http://127.0.0.1:9011/convert \
  -b cookies.txt \
  -F "stage=all" \
  -F "dataset=mnist_train.csv" \
  -F "drop_na=on" \
  -F "split_xy=on" \
  -F "normalize=0-1" \
  -F "conv1_filters=32" \
  -F "dense_units=128" \
  -F "num_classes=10" \
  -F "loss_method=CrossEntropy" \
  -F "optimizer_method=Adam" \
  -F "learning_rate=0.001" \
  -F "epochs=10" \
  -F "metrics=accuracy"
```

---

### 3.3 코드 실행

#### POST `/run/<stage>`
생성된 코드 실행

**cURL 예제 - 전처리 실행:**
```bash
curl -X POST http://127.0.0.1:9011/run/pre \
  -b cookies.txt \
  -H "Content-Type: application/json"
```

**응답:**
```json
{
    "ok": true,
    "pid": 12345
}
```

**cURL 예제 - 학습 실행:**
```bash
curl -X POST http://127.0.0.1:9011/run/train \
  -b cookies.txt \
  -H "Content-Type: application/json"
```

**에러 응답:**
```json
{
    "error": "preprocessing.py not found"
}
```

---

### 3.4 로그 스트리밍

#### GET `/logs/stream`
Server-Sent Events로 실시간 로그 수신

**cURL 예제:**
```bash
# SSE 스트림 수신
curl -N http://127.0.0.1:9011/logs/stream?stage=train \
  -b cookies.txt \
  -H "Accept: text/event-stream"
```

**응답 예제:**
```
data: [train][2025-01-01 12:00:00] === TRAINING START ===

data: [train][2025-01-01 12:00:01] Using device: cuda

data: [train][2025-01-01 12:00:02] Data loaded: train=48000, test=12000

data: [train][2025-01-01 12:00:03] Epoch 1/10

data: [train][2025-01-01 12:00:04] Train Loss: 0.4523, Train Acc: 85.32%
```

**JavaScript 클라이언트 예제:**
```javascript
const eventSource = new EventSource('/logs/stream?stage=train');

eventSource.onmessage = (event) => {
    console.log('Log:', event.data);
    // UI에 로그 추가
    appendLog(event.data);
};

eventSource.onerror = (error) => {
    console.error('Stream error:', error);
    eventSource.close();
};

// 스트림 종료
eventSource.close();
```

---

### 3.5 데이터 정보 조회

#### GET `/data-info`
데이터셋 정보 조회

**cURL 예제 - 데이터 크기:**
```bash
curl "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=shape"
```

**응답:**
```json
{
    "rows": 60000,
    "cols": 785
}
```

**cURL 예제 - 데이터 구조:**
```bash
curl "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=structure"
```

**응답:**
```json
{
    "columns": [
        {"name": "label", "dtype": "int64"},
        {"name": "pixel0", "dtype": "int64"},
        {"name": "pixel1", "dtype": "int64"},
        ...
    ]
}
```

**cURL 예제 - 샘플 데이터:**
```bash
curl "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=sample&n=3"
```

**응답:**
```json
{
    "columns": ["label", "pixel0", "pixel1", ...],
    "sample": [
        [5, 0, 0, ...],
        [0, 0, 0, ...],
        [4, 0, 0, ...]
    ]
}
```

**cURL 예제 - 이미지 미리보기:**
```bash
curl "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=images&n=5"
```

**응답:**
```json
{
    "images": [
        "data:image/png;base64,iVBORw0KGgo...",
        "data:image/png;base64,iVBORw0KGgo...",
        ...
    ]
}
```

---

### 3.6 파일 다운로드

#### GET `/download/<stage>`
생성된 코드 다운로드

**cURL 예제 - 전처리 코드 다운로드:**
```bash
curl -O -J http://127.0.0.1:9011/download/pre \
  -b cookies.txt
# preprocessing.py 저장됨
```

**cURL 예제 - 전체 코드 ZIP 다운로드:**
```bash
curl -O -J http://127.0.0.1:9011/download/all \
  -b cookies.txt
# workspace_a1b2c3d4_1234567890.zip 저장됨
```

---

## 4. WebSocket/SSE 실시간 통신

### 4.1 SSE (Server-Sent Events) 구조

**이벤트 형식:**
```
data: [<stage>][<timestamp>] <message>\n\n
```

**스테이지별 로그 예제:**

전처리 로그:
```
data: [pre][2025-01-01 12:00:00] === PREPROCESSING START ===
data: [pre][2025-01-01 12:00:01] START: 데이터 선택/로딩
data: [pre][2025-01-01 12:00:02] Train size: 48000, Test size: 12000
data: [pre][2025-01-01 12:00:03] END  : 데이터 선택/로딩 (elapsed=1.234s)
data: [pre][2025-01-01 12:00:04] START: 결측치 행 삭제
data: [pre][2025-01-01 12:00:05] Removed 0 rows from train, 0 rows from test
data: [pre][2025-01-01 12:00:06] END  : 결측치 행 삭제 (elapsed=0.567s)
data: [pre][2025-01-01 12:00:07] === PREPROCESSING DONE ===
```

학습 로그:
```
data: [train][2025-01-01 12:05:00] === TRAINING START ===
data: [train][2025-01-01 12:05:01] Using device: cuda
data: [train][2025-01-01 12:05:02] Data loaded: train=48000, test=12000
data: [train][2025-01-01 12:05:03] 
data: [train][2025-01-01 12:05:03] Epoch 1/10
data: [train][2025-01-01 12:05:15] Train Loss: 0.4523, Train Acc: 85.32%
data: [train][2025-01-01 12:05:20] Val Loss: 0.3012, Val Acc: 91.23%
data: [train][2025-01-01 12:05:21] ✓ Best model saved (val_loss: 0.3012)
data: [train][2025-01-01 12:05:22] 
data: [train][2025-01-01 12:05:22] Epoch 2/10
data: [train][2025-01-01 12:05:34] Train Loss: 0.2834, Train Acc: 91.45%
data: [train][2025-01-01 12:05:39] Val Loss: 0.2456, Val Acc: 92.78%
data: [train][2025-01-01 12:05:40] ✓ Best model saved (val_loss: 0.2456)
```

---

## 5. 에러 코드

### 5.1 HTTP 상태 코드

| 코드 | 설명 | 대응 방법 |
|------|------|-----------|
| 200 | 성공 | - |
| 302 | 리다이렉트 | Location 헤더 따라가기 |
| 400 | 잘못된 요청 | 파라미터 확인 |
| 404 | 리소스 없음 | 경로 및 파일 존재 확인 |
| 500 | 서버 에러 | 서버 로그 확인 |

### 5.2 애플리케이션 에러

```json
{
    "error": "<error_type>",
    "message": "<detailed_message>",
    "code": "<error_code>"
}
```

**에러 타입:**

| 에러 타입 | 코드 | 설명 |
|-----------|------|------|
| `unknown_stage` | E001 | 알 수 없는 스테이지 |
| `file_not_found` | E002 | 파일이 존재하지 않음 |
| `dataset_not_found` | E003 | 데이터셋이 없음 |
| `process_already_running` | E004 | 이미 실행 중인 프로세스 |
| `invalid_parameters` | E005 | 잘못된 파라미터 |
| `workspace_error` | E006 | 워크스페이스 오류 |

---

## 6. 테스트 시나리오

### 6.1 전체 파이프라인 테스트

```bash
#!/bin/bash
# test_pipeline.sh

# 1. 쿠키 생성
echo "1. 세션 시작..."
curl -c cookies.txt http://127.0.0.1:9011/
echo ""

# 2. 전체 코드 생성
echo "2. 코드 생성..."
curl -X POST http://127.0.0.1:9011/convert \
  -b cookies.txt \
  -F "stage=all" \
  -F "dataset=mnist_train.csv" \
  -F "is_test=false" \
  -F "a=80" \
  -F "drop_na=on" \
  -F "split_xy=on" \
  -F "resize_n=28" \
  -F "normalize=0-1" \
  -F "conv1_filters=32" \
  -F "conv1_kernel=3" \
  -F "conv1_activation=relu" \
  -F "pool1_type=max" \
  -F "pool1_size=2" \
  -F "dense_units=128" \
  -F "num_classes=10" \
  -F "loss_method=CrossEntropy" \
  -F "optimizer_method=Adam" \
  -F "learning_rate=0.001" \
  -F "epochs=5" \
  -F "batch_size=64" \
  -F "metrics=accuracy" \
  -s -o /dev/null -w "%{http_code}\n"
echo ""

# 3. 전처리 실행
echo "3. 전처리 실행..."
curl -X POST http://127.0.0.1:9011/run/pre \
  -b cookies.txt \
  -H "Content-Type: application/json"
echo ""

# 4. 로그 확인 (5초간)
echo "4. 로그 스트리밍..."
timeout 5 curl -N http://127.0.0.1:9011/logs/stream?stage=pre \
  -b cookies.txt \
  -H "Accept: text/event-stream"
echo ""

# 5. 학습 실행
echo "5. 학습 실행..."
curl -X POST http://127.0.0.1:9011/run/train \
  -b cookies.txt \
  -H "Content-Type: application/json"
echo ""

# 6. 평가 실행
echo "6. 평가 실행..."
curl -X POST http://127.0.0.1:9011/run/eval \
  -b cookies.txt \
  -H "Content-Type: application/json"
echo ""

# 7. 코드 다운로드
echo "7. 코드 다운로드..."
curl -O -J http://127.0.0.1:9011/download/all \
  -b cookies.txt
echo "완료!"
```

### 6.2 데이터 조회 테스트

```bash
#!/bin/bash
# test_data.sh

# 데이터 크기 조회
echo "데이터 크기:"
curl -s "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=shape" | python -m json.tool

# 데이터 구조 조회
echo -e "\n데이터 구조 (처음 5개 컬럼):"
curl -s "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=structure" | \
  python -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data['columns'][:5], indent=2))"

# 샘플 데이터 조회
echo -e "\n샘플 데이터 (3개):"
curl -s "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=sample&n=3" | \
  python -c "import sys, json; data=json.load(sys.stdin); print(f\"Columns: {data['columns'][:5]}...\"); print(f\"Samples: {len(data['sample'])} rows\")"

# 이미지 조회
echo -e "\n이미지 미리보기:"
curl -s "http://127.0.0.1:9011/data-info?file=mnist_train.csv&type=images&n=2" | \
  python -c "import sys, json; data=json.load(sys.stdin); print(f\"Images: {len(data['images'])} generated\")"
```

### 6.3 JavaScript 클라이언트 테스트

```javascript
// test_client.js

class APITest {
    constructor() {
        this.baseURL = 'http://127.0.0.1:9011';
    }
    
    // 1. 코드 생성 테스트
    async testConvert() {
        console.log('Testing code conversion...');
        
        const formData = new FormData();
        formData.append('stage', 'pre');
        formData.append('dataset', 'mnist_train.csv');
        formData.append('drop_na', 'on');
        formData.append('normalize', '0-1');
        
        const response = await fetch(`${this.baseURL}/convert`, {
            method: 'POST',
            body: formData,
            credentials: 'same-origin'
        });
        
        console.log('Status:', response.status);
        const html = await response.text();
        console.log('Response length:', html.length);
        
        return response.ok;
    }
    
    // 2. 실행 테스트
    async testRun() {
        console.log('Testing code execution...');
        
        const response = await fetch(`${this.baseURL}/run/pre`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin'
        });
        
        const data = await response.json();
        console.log('Run result:', data);
        
        return data.ok;
    }
    
    // 3. SSE 로그 테스트
    testSSE(duration = 5000) {
        console.log('Testing SSE logs...');
        
        return new Promise((resolve) => {
            const eventSource = new EventSource(`${this.baseURL}/logs/stream?stage=pre`);
            const logs = [];
            
            eventSource.onmessage = (event) => {
                console.log('Log:', event.data);
                logs.push(event.data);
            };
            
            eventSource.onerror = (error) => {
                console.error('SSE Error:', error);
                eventSource.close();
                resolve(logs);
            };
            
            setTimeout(() => {
                eventSource.close();
                console.log(`Received ${logs.length} log messages`);
                resolve(logs);
            }, duration);
        });
    }
    
    // 4. 데이터 정보 테스트
    async testDataInfo() {
        console.log('Testing data info...');
        
        const types = ['shape', 'structure', 'sample', 'images'];
        const results = {};
        
        for (const type of types) {
            const url = `${this.baseURL}/data-info?file=mnist_train.csv&type=${type}&n=3`;
            const response = await fetch(url);
            const data = await response.json();
            
            results[type] = data;
            console.log(`${type}:`, data);
        }
        
        return results;
    }
    
    // 전체 테스트 실행
    async runAllTests() {
        console.log('=== Starting API Tests ===');
        
        try {
            // 1. 코드 생성
            await this.testConvert();
            console.log('✓ Code conversion test passed');
            
            // 2. 실행
            await this.testRun();
            console.log('✓ Code execution test passed');
            
            // 3. 로그 스트리밍
            await this.testSSE(3000);
            console.log('✓ SSE logging test passed');
            
            // 4. 데이터 정보
            await this.testDataInfo();
            console.log('✓ Data info test passed');
            
            console.log('\n=== All Tests Passed ===');
        } catch (error) {
            console.error('Test failed:', error);
        }
    }
}

// 테스트 실행
const tester = new APITest();
tester.runAllTests();
```

### 6.4 Python 클라이언트 테스트

```python
# test_api.py

import requests
import json
import time
from typing import Dict, Any

class AIBlockAPI:
    def __init__(self, base_url: str = "http://127.0.0.1:9011"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def init_session(self) -> str:
        """세션 초기화 및 UID 획득"""
        response = self.session.get(f"{self.base_url}/")
        uid = self.session.cookies.get('uid')
        print(f"Session initialized with UID: {uid}")
        return uid
    
    def convert_code(self, stage: str, params: Dict[str, Any]) -> bool:
        """코드 생성"""
        params['stage'] = stage
        response = self.session.post(
            f"{self.base_url}/convert",
            data=params
        )
        print(f"Convert {stage}: {response.status_code}")
        return response.ok
    
    def run_stage(self, stage: str) -> Dict[str, Any]:
        """스테이지 실행"""
        response = self.session.post(
            f"{self.base_url}/run/{stage}",
            headers={'Content-Type': 'application/json'}
        )
        result = response.json()
        print(f"Run {stage}: {result}")
        return result
    
    def get_data_info(self, dataset: str, info_type: str = "shape") -> Dict[str, Any]:
        """데이터 정보 조회"""
        response = self.session.get(
            f"{self.base_url}/data-info",
            params={'file': dataset, 'type': info_type}
        )
        return response.json()
    
    def download_code(self, stage: str, save_path: str = None) -> bool:
        """코드 다운로드"""
        response = self.session.get(f"{self.base_url}/download/{stage}")
        if response.ok:
            filename = save_path or f"{stage}_code.py"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Code saved to {filename}")
            return True
        return False
    
    def stream_logs(self, stage: str, duration: int = 5):
        """로그 스트리밍 (SSE)"""
        import sseclient
        
        response = self.session.get(
            f"{self.base_url}/logs/stream",
            params={'stage': stage},
            stream=True,
            headers={'Accept': 'text/event-stream'}
        )
        
        client = sseclient.SSEClient(response)
        start_time = time.time()
        
        for event in client.events():
            print(f"Log: {event.data}")
            if time.time() - start_time > duration:
                break

def test_full_pipeline():
    """전체 파이프라인 테스트"""
    api = AIBlockAPI()
    
    # 1. 세션 초기화
    uid = api.init_session()
    
    # 2. 전처리 코드 생성
    api.convert_code('pre', {
        'dataset': 'mnist_train.csv',
        'drop_na': 'on',
        'split_xy': 'on',
        'normalize': '0-1'
    })
    
    # 3. 모델 코드 생성
    api.convert_code('model', {
        'input_w': 28,
        'input_h': 28,
        'input_c': 1,
        'conv1_filters': 32,
        'dense_units': 128,
        'num_classes': 10
    })
    
    # 4. 데이터 정보 조회
    info = api.get_data_info('mnist_train.csv', 'shape')
    print(f"Dataset shape: {info}")
    
    # 5. 전처리 실행
    result = api.run_stage('pre')
    if result.get('ok'):
        print("Preprocessing started successfully")
        # api.stream_logs('pre', duration=3)
    
    # 6. 코드 다운로드
    api.download_code('all', f'workspace_{uid}.zip')
    
    print("Test completed!")

if __name__ == "__main__":
    test_full_pipeline()
```

---

## 7. 추가 정보

### 7.1 파일 구조
```
workspace/<uid>/
├── preprocessing.py      # 전처리 코드
├── model.py              # 모델 정의
├── training.py           # 학습 코드
├── evaluation.py         # 평가 코드
├── inputs_pre.json       # 전처리 입력값
├── inputs_model.json     # 모델 입력값
├── inputs_train.json     # 학습 입력값
├── inputs_eval.json      # 평가 입력값
├── data/
│   └── dataset.pt        # 전처리된 데이터
└── artifacts/
    ├── best_model.pth    # 최고 성능 모델
    ├── final_model.pth   # 최종 모델
    ├── training_history.json
    ├── evaluation_results.json
    ├── confusion_matrix.png
    ├── prediction_samples.png
    └── misclassified_samples.png
```

### 7.2 제한사항
- 최대 요청 크기: 16MB
- 동시 실행 프로세스: 1개 per UID
- SSE 연결 타임아웃: 30분
- 워크스페이스 보존: 7일

### 7.3 문의 및 지원