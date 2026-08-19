# Customer Support Agent

주문 관련 고객 문의를 처리하는 Tool Calling 기반 에이전트를 단계적으로 구현하는 프로젝트입니다.

## Tool Calling 호환성 검사

`scripts/check_tool_calling.py`는 에이전트 루프를 구현하기 전에 OpenAI Python SDK와
`llama-server`의 Chat Completions Tool Calling 호환성을 확인합니다.

이 검사는 이름이 지정된 단일 도구 호출을 강제한 뒤 다음 항목을 검증합니다.

- 요청이 OpenAI 호환 `/v1/chat/completions` 엔드포인트에 도달하는지
- 응답에 정확히 하나의 function tool call이 포함되는지
- 도구 이름이 `tool_calling_probe`인지
- 도구 인자가 유효한 JSON이며 `{"message": "tool-calling-ok"}`와 일치하는지

### llama-server 요구 조건

`llama-server`는 Tool Calling을 처리할 수 있는 Jinja 채팅 템플릿과 함께 실행해야 합니다.
llama.cpp 공식 문서는 function calling에 `--jinja`를 사용하고, `/props`의
`chat_template` 또는 `chat_template_tool_use`를 확인하도록 안내합니다.

```text
llama-server --jinja -m <model.gguf> --host <host> --port 8080
```

모델의 기본 템플릿이 Tool Calling을 지원하지 않으면 `--chat-template-file`로 적합한
템플릿을 지정해야 합니다. 자세한 내용은
[llama.cpp Function Calling 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)와
[llama.cpp HTTP Server 문서](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)를
참고합니다.

호환성 검사에 사용한 서버에서는 다음 조건을 확인했습니다.

- llama.cpp build: `b9982-99f3dc322`
- model alias: `google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0`
- `has_chat_template`: `true`
- `chat_template_mentions_tools`: `true`
- `chat_template_caps.supports_tools`: `true`
- `chat_template_caps.supports_tool_calls`: `true`
- `has_chat_template_tool_use`: `false`

별도의 `chat_template_tool_use`는 없었지만 기본 채팅 템플릿이 도구를 지원했고, 실제
Tool Call 응답까지 성공했습니다. 따라서 이 서버·모델 조합에서는
`has_chat_template_tool_use: true`가 필수 조건은 아닙니다.

### 실행 방법

저장소 루트에서 의존성을 동기화합니다.

```powershell
uv sync
```

현재 PowerShell 세션에 실행 설정을 제공합니다. `LLM_BASE_URL`에는 `/v1`까지 포함합니다.

```powershell
$env:LLM_BASE_URL = "http://<llama-server-host>:8080/v1"
$env:LLM_MODEL_NAME = "google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0"
```

위의 `http://` 예시는 loopback 또는 별도로 암호화되고 접근이 통제되는 개발
네트워크에서만 사용합니다. API 키를 사용하거나 신뢰할 수 없는 네트워크를 통과한다면
`https://`를 사용해야 합니다. `llama-server`는 OpenSSL 지원 빌드와
`--ssl-key-file`, `--ssl-cert-file` 옵션으로 TLS를 구성할 수 있습니다.

API 키가 필요한 서버라면 추가로 설정합니다. 키가 필요 없는 `llama-server`에서는 생략할
수 있으며, 검사 스크립트가 `no-api-key`를 대신 사용합니다.

```powershell
$env:LLM_API_KEY = "<api-key>"
```

검사를 실행합니다.

```powershell
uv run python scripts/check_tool_calling.py
```

성공하면 다음과 같은 결과가 출력됩니다.

```text
PASS: model='<model-name>', tool='tool_calling_probe', arguments={'message': 'tool-calling-ok'}
```

### 검사 범위

이 검사는 SDK와 서버 사이에서 도구 스키마 및 Tool Call 응답 형식이 호환됨을 확인합니다.
도구 선택을 강제하므로 모델의 자율적인 도구 선택 능력은 평가하지 않습니다. 또한 도구
실행, 결과 재전달, 반복 및 종료 조건을 포함하는 에이전트 루프는 이 검사의 범위가
아닙니다.
