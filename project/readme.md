# 개선된 프론트엔드 구조

## 📁 새로운 디렉터리 구조

```
project/
├── templates/
│   ├── base.html           # 기본 레이아웃 (layout.html 대체)
│   ├── index.html          # 메인 페이지 (간소화)
│   └── components/         # 컴포넌트 분리
│       ├── sidebar.html    # 사이드바
│       ├── code_panel.html # 코드 패널
│       ├── data_panel.html # 데이터 패널
│       └── log_panel.html  # 로그 패널
│
├── static/
│   ├── css/
│   │   ├── main.css       # 메인 스타일
│   │   ├── blocks.css     # 블록 스타일
│   │   └── components.css # 컴포넌트 스타일
│   │
│   └── js/
│       ├── app.js         # 메인 애플리케이션
│       ├── state.js       # 상태 관리
│       ├── api.js         # API 통신
│       └── components/
│           ├── blocks.js   # 블록 관리
│           ├── tabs.js     # 탭 관리
│           ├── data.js     # 데이터 뷰어
│           └── logs.js     # 로그 스트리밍
```

## 주요 개선 사항

1. **컴포넌트 기반 구조**: 재사용 가능한 컴포넌트로 분리
2. **상태 관리 중앙화**: 모든 상태를 한 곳에서 관리
3. **API 레이어 분리**: 백엔드 통신 로직 독립
4. **CSS 모듈화**: 용도별로 스타일 파일 분리
5. **이벤트 위임**: 성능 최적화
