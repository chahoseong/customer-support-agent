# Customer Support Agent

주문 관련 고객 문의를 처리하는 Tool Calling 기반 에이전트를 단계적으로 구현하는
프로젝트입니다.

## 설치

### 전제 조건

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

저장소 루트에서 프로젝트 의존성을 설치합니다.

```powershell
uv sync
```

## 문서

- [로드맵](docs/ROADMAP.md): 프로젝트의 단계별 목표와 앞으로의 진행 방향을 설명합니다.
- [Tool Calling 호환성을 판단하는 방법](docs/tool-calling-compatibility.md): 모델 서버의
  Tool Calling 지원 여부를 판단하고 검증하는 방법을 설명합니다.
- [Agent Loop with Tool Calling](docs/agent-loop-with-tool-calling.md): Agent Loop의 개념과
  Tool Calling을 처리하는 흐름, 이 프로젝트에서 구현한 범위를 설명합니다.
