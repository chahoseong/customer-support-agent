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

## e2e/customer_order_status.py

실제 모델을 사용해 고객 주문 상태 E2E 흐름을 확인합니다. `Agent`가
`get_customer_orders`, `find_order`, `find_shipment`을 순서대로 실행하고, 각 결과를
원래 Tool Call ID와 연결해 후속 Model 요청에 전달한 뒤 비어 있지 않은 최종 응답을
반환하는지 검증합니다.

```powershell
uv run --env-file .env python scripts/e2e/customer_order_status.py
```
