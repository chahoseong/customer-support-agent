# Project Context

이 문서는 이 프로젝트에서 반복적으로 사용하는 핵심 용어의 의미를 정의합니다.

`Avoid`에는 해당 용어와 혼용하거나 대신 사용하지 않을 표현을 적습니다.

## Agent

사용자 요청을 처리하기 위해 Model을 호출하고 필요한 Tool을 실행하며, 최종 응답이나 종료 조건에 도달할 때까지 실행 흐름을 제어하는 애플리케이션 구성요소입니다.

**Avoid:** `Model`

## Model

메시지와 사용 가능한 Tool 정보를 입력받아 텍스트 응답이나 Tool 사용 요청을 생성합니다. Tool을 직접 실행하지 않습니다.

**Avoid:** `Agent`

## Agent Loop

한 번의 Agent 실행에서 Model 호출과 Tool 실행 결과 전달을 최종 응답 또는 중단 조건에 도달할 때까지 반복하는 제어 흐름입니다.

**Avoid:** `Conversation`

## Conversation

사용자와 Agent 사이에 이어지는 하나 이상의 메시지 교환입니다. 대화를 유지하거나 저장하는 구현 방식은 포함하지 않습니다.

**Avoid:** `Agent Loop`, `Session`

## Tool

Agent가 필요한 정보를 조회하거나 제한된 동작을 수행할 수 있도록 애플리케이션이 제공하는 호출 가능한 기능입니다. Model은 Tool 사용을 요청하고 Agent가 이를 검증하고 실행합니다.

**Avoid:** `Tool Call`, `Tool Result`
