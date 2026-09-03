# Project Context

이 문서는 설계와 구현을 논의할 때 같은 용어를 같은 의미로 사용하기 위한
기준입니다. 프로젝트에서 반복적으로 사용하는 핵심 용어만 정의합니다.

`Avoid`에는 해당 용어와 혼용하거나 대신 사용하지 않을 표현을 적습니다.

## Agent

Agent는 사용자 요청을 처리하기 위해 Model과 Tool을 사용하는 애플리케이션 구성 요소를 뜻합니다.

**Avoid:** `Model`

## Model

Model은 Agent가 전달한 메시지를 바탕으로 다음 출력을 생성하는 언어 모델을 뜻합니다.

**Avoid:** `Agent`

## Agent Loop

Agent Loop는 한 번의 Agent 실행에서 Model 호출, Tool 실행과 그 결과 전달을 최종 응답 또는 중단 조건에 도달할 때까지 반복하는 제어 흐름을 뜻합니다.

## Conversation

Conversation은 사용자와 Agent가 메시지를 주고받는 대화를 뜻합니다.

**Avoid:** `Session`

## Tool

Tool은 Agent가 사용할 수 있도록 애플리케이션이 제공하는 호출 가능한 기능을 뜻합니다.

## Evaluation

Evaluation은 Evaluation Dataset의 각 case를 사용해 Agent를 실행하고, Evaluator가 그 결과를 정해진 기준에 따라 판정하는 과정입니다.

## Trace

Trace는 한 번의 Agent 실행에서 발생한 Model 호출과 Tool 실행 등의 관계와 순서를 span으로 표현한 구조화된 실행 기록을 뜻합니다.

**Avoid:** `Log`, `Metric`
