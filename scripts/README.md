# 스크립트 실행 가이드

이 디렉터리의 스크립트는 실제 모델 서버와 연결해 Tool Calling과 Agent Loop를 수동으로
확인합니다.

## 환경 설정

저장소 루트의 `.env` 파일에 OpenAI 호환 API 정보를 입력합니다. `LLM_BASE_URL`에는
`/v1`까지 포함합니다.

```dotenv
LLM_BASE_URL=http://<openai-compatible-host>:8080/v1
LLM_MODEL_NAME=<model-name>
# LLM_API_KEY=<api-key>
```

인증이 필요한 서버에서만 `LLM_API_KEY`를 설정합니다.

아래 명령은 저장소 루트에서 실행합니다.

## check_tool_calling_compatibility.py

모델 서버가 OpenAI 호환 Tool Call 응답을 반환하는지 확인합니다.

```powershell
uv run --env-file .env python scripts/check_tool_calling_compatibility.py
```

## e2e/order_status.py

실제 모델과 기본 Customer Support Agent 구성으로 알려진 주문의 상태를
조회하는 전체 흐름을 확인합니다.

```powershell
uv run --env-file .env python scripts/e2e/order_status.py
```

이 스크립트는 다음을 확인합니다.

- 실제 모델이 `find_order`를 `order-001`로 호출합니다.
- Agent가 Tool을 실행해 `processing` 상태를 얻습니다.
- Tool Result가 원래 Tool Call ID와 연결된 채 후속 Model 요청에 전달됩니다.
- Agent가 최종 `AgentResult`를 반환합니다.

## e2e/order_not_found.py

실제 모델과 기본 Customer Support Agent 구성으로 존재하지 않는 주문을
조회하는 전체 흐름을 확인합니다.

```powershell
uv run --env-file .env python scripts/e2e/order_not_found.py
```

이 스크립트는 다음을 확인합니다.

- 실제 모델이 `find_order`를 `order-999`로 호출합니다.
- Agent가 Tool을 실행해 `order_not_found` 구조화 결과를 얻습니다.
- Tool Result가 Agent 실행을 중단시키지 않고, 원래 Tool Call ID와 연결된 채
  후속 Model 요청에 전달됩니다.
- Agent가 최종 `AgentResult`를 반환합니다.

## E2E 실행 결과

두 E2E 스크립트는 같은 형식으로 결과를 보여줍니다.

- `PASS`는 위에 기술한 Tool Call, Tool Result 재전달과 최종 `AgentResult`까지
  확인했다는 뜻입니다.
- `FAIL`은 누락된 Tool Call이나 Tool Result 등 확인하지 못한 내용을 함께
  출력합니다.
- `Final answer`는 실제 모델이 생성한 최종 응답을 사람이 확인할 수 있도록
  보여줍니다. 스크립트는 이 문구의 품질을 자동으로 채점하지 않습니다.

`check_tool_calling_compatibility.py`는 Model 서버가 기본 Tool Calling 형식을
지원하는지 확인합니다. E2E 스크립트는 여기서 더 나아가 실제 Customer
Support Agent가 Model, Tool, Tool Result와 최종 결과를 끝까지 연결하는지
확인합니다.
