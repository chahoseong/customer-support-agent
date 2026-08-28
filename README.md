# Customer Support Agent

주문 관련 고객 문의를 처리하는 Tool Calling 기반 Customer Support Agent를 직접
구현하는 학습 프로젝트입니다.

Model의 Tool 사용 요청을 처리하는 Agent Loop부터 여러 Tool을 사용하는 고객지원
흐름, Evaluation과 Observability까지 단계적으로 확장하며 AI Agent의 실행 구조를
이해하고 검증하는 것을 목표로 합니다.

## 주요 기능

- **주문 조회** — 고객의 주문 목록과 개별 주문 상태를 확인합니다.
- **배송 안내** — 배송이 시작된 주문의 배송 상태를 안내합니다.
- **취소 안내** — 주문 상태와 취소 정책을 바탕으로 취소 가능 여부를 안내합니다.
- **대화형 지원** — 주문 정보가 부족하면 확인 질문을 하고 후속 대화를 이어갑니다.

## 빠른 시작

### 전제 조건

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

저장소 루트에서 프로젝트 의존성을 설치합니다.

```powershell
uv sync
```

저장소 루트에 `.env` 파일을 만들고 사용할 Model을 설정합니다.
`LLM_BASE_URL`에는 `/v1`까지 포함합니다.

```dotenv
LLM_BASE_URL=http://<openai-compatible-host>:8080/v1
LLM_MODEL_NAME=<model-name>
# LLM_API_KEY=<api-key>
```

로컬 Web Chat을 실행합니다.

```powershell
uv run --env-file .env python scripts/chat.py
```

환경변수와 E2E를 포함한 자세한 실행 방법은
[스크립트 실행 가이드](scripts/README.md)를 참고하세요.

## 주요 문서

- [Roadmap](docs/ROADMAP.md) — 프로젝트의 목표와 단계별 진행 방향을 설명합니다.
- [Architecture](docs/ARCHITECTURE.md) — 구성 요소의 책임, 의존 방향과 핵심 실행
  흐름을 설명합니다.
