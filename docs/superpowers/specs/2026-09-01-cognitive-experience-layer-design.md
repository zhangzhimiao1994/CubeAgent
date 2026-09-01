# Cognitive Experience Layer Design

Date: 2026-09-01

## Repository boundary

This design belongs to the current CubeAgent conversation-agent repository:

- Local path: `E:\code_x\mofangagent`
- GitHub repository: `https://github.com/zhangzhimiao1994/CubeAgent`

This repository remains a pure conversation-agent product line. The Cognitive Experience Layer must not introduce harness-level code execution, project mutation, Vibe Coding execution, or the future OpenAI/DeepSeek harness refactor.

The old product term "Evolution" should not be used for this feature. The correct boundary is "持续学习与经验成长", implemented as an independent cognitive layer on top of existing Hermes+/Memory capabilities.

## Goal

Add a system that lets the Agent learn from long-term interaction outcomes, abstract reusable experience, reason about confidence and contradictions, and use verified experience to improve future decisions.

The target loop is:

```text
Interaction → Episode → Memory → Experience → Reflection → Belief/Strategy → Skill → Future Decision → Outcome → New Reflection
```

The system is not ordinary long-term memory. A chat transcript is only raw material. It becomes experience only when it has durable value, repeated pattern evidence, explicit feedback, or a meaningful success/failure outcome.

## Current baseline

The repository already has useful primitives:

- `src/agent_hub/memory/types.py`
  - memory layer/category
  - confidence
  - heat
  - recall count
  - lock state
  - project/conversation scope
  - summary period
- `src/agent_hub/memory/service.py`
  - candidate add
  - search
  - edit/forget
  - lock/unlock
  - decay
  - consolidation
  - safety filters for secrets and prompt-like external content
- `src/agent_hub/hermes/advisor.py`
  - persistent Hermes advice
  - bounded outcome learning
  - confirmed-memory retrieval and runtime injection
  - separation between conversation advice and scheduler/runtime observations
- `src/agent_hub/runtime/hermes_context.py`
  - bounded prompt injection for confirmed Hermes+ memories
  - filters runtime/scheduler observations out of prompt context
- `src/agent_hub/context/builder.py`
  - structured context sections and token-budget selection
- `web/src/pages/HermesPage.tsx`
  - Chinese learning summary, confirmation, deletion, filtering
- `web/src/pages/RunsPage.tsx`
  - dispatch card and Hermes+ injected/skipped memory visibility

The new system should reuse these capabilities. It should not copy third-party memory frameworks into the repository or introduce a heavy vector store in the first implementation slice.

## Architecture

Add a new package:

```text
src/agent_hub/cognitive/
├── __init__.py
├── types.py
├── repository.py
├── service.py
├── router.py
├── reflection.py
├── beliefs.py
├── relationships.py
├── world_state.py
└── skills.py
```

The package is a Cognitive / Experience Layer between raw run history and runtime decision-making.

First-slice persistence uses the existing `AdminResourceRow.kind = "hermes"` bucket with namespaced resource ids:

- `cognitive_experience:<uuid>`
- `cognitive_reflection:<uuid>`
- `cognitive_belief:<uuid>`
- `cognitive_relationship:<user_id>`
- `cognitive_world:<scope_id>`
- `cognitive_skill:<uuid>`

This avoids a database check-constraint migration in the first slice. A later storage hardening pass can move these records to dedicated tables or new resource kinds with a non-destructive Alembic migration.

It consumes:

- run completion state
- run events and artifacts
- Hermes confirmed or pending learning records
- explicit user feedback
- memory records
- scheduler/runtime observations

It produces:

- reusable experiences
- reflections
- belief updates
- relationship signals
- world-state facts
- skill candidates
- bounded context advice for future runs

It must stay decoupled from:

- model provider implementations
- harness execution loops
- SOUL/persona core rules
- tool permission policy
- Feishu/OAuth identity binding
- database-specific storage choices beyond repository interfaces

## Core models

### Experience

An experience is a reusable lesson derived from one or more interactions.

Required fields:

- `id`
- `tenant_id`
- `user_id`
- `kind`
- `summary`
- `lesson`
- `strategy`
- `confidence`
- `evidence`
- `contradictions`
- `source_run_ids`
- `source_memory_ids`
- `tags`
- `applies_to_modes`
- `applies_to_agents`
- `use_count`
- `success_count`
- `failure_count`
- `last_used_at`
- `last_verified_at`
- `version`
- `status`
- `created_at`
- `updated_at`

Kinds:

- `user_preference`
- `project_fact`
- `workflow_strategy`
- `error_handling`
- `ui_rule`
- `communication_style`
- `tooling_strategy`
- `domain_pattern`

Statuses:

- `candidate`
- `confirmed`
- `active`
- `superseded`
- `deprecated`
- `rejected`

Only `confirmed` or `active` experiences can affect future runtime behavior.

### Reflection

A reflection explains why an outcome happened and what should change.

Required fields:

- `id`
- `tenant_id`
- `user_id`
- `source_run_id`
- `trigger`
- `outcome`
- `causal_analysis`
- `counterfactual`
- `positive_patterns`
- `negative_patterns`
- `proposed_experience_ids`
- `confidence`
- `created_at`

Triggers:

- `user_correction`
- `user_rejection`
- `user_confirmation`
- `user_satisfaction`
- `run_completed`
- `run_failed`
- `reviewer_failed`
- `timeout`
- `manual_feedback`

### Belief

A belief is a weighted judgment about the user, project, environment, or system behavior.

Required fields:

- `id`
- `tenant_id`
- `user_id`
- `subject`
- `claim`
- `confidence`
- `evidence`
- `contradictions`
- `last_verified_at`
- `status`
- `created_at`
- `updated_at`

Beliefs must not become permanent from one interaction. Repeated evidence raises confidence; contradictions lower confidence or create a competing belief.

### Relationship state

Relationship state tracks long-term interaction familiarity without pretending to be a person or changing safety behavior.

Fields:

- familiarity score
- preferred communication density
- preferred language
- preferred confirmation style
- recurring project names
- shared milestones
- recent friction points
- last_interaction_at

Relationship state can adjust tone and default workflow suggestions. It cannot modify security, permissions, or core persona.

### World state

World state tracks active facts and pending events.

Fields:

- active projects
- important people/entities
- open items
- future events
- event status
- last_verified_at
- source evidence

World state facts expire or require re-verification when time-sensitive.

### Skill library

The Skill Library promotes repeatedly successful experiences into reusable workflow/skill candidates.

Fields:

- skill name
- purpose
- steps
- required inputs
- output contract
- evidence
- use_count
- success_count
- failure_count
- version
- status

The first version should create skill candidates only. Actual Skill installation must continue using the existing Skill upload/approval flow.

## Memory and experience routing

Future runs should use a router instead of directly injecting all memory.

Input:

- current user request
- task mode
- selected or candidate agents
- workflow id
- project/conversation id
- confirmed memories
- active experiences
- beliefs
- relationship hints
- world-state facts

Output:

- selected context items
- skipped context items
- reasons
- conflict decisions
- confidence scores

Default thresholds:

- `score >= 0.70`: eligible for injection
- `0.45 <= score < 0.70`: visible as retrieved but skipped
- `score < 0.45`: ignored

Injection limits:

- maximum 3 active experience items
- maximum 2 belief/world-state items
- maximum 900 bytes for experience prompt block in the first implementation
- current user instruction always overrides older experience

## User confirmation boundary

The Agent may automatically:

- create ordinary candidate experiences
- merge duplicate candidates
- lower confidence after failed use
- mark stale candidates as deprecated
- propose skill candidates

The Agent may not automatically:

- edit SOUL/persona core identity
- change safety rules
- change tool permission policy
- install or enable a Skill
- bind external accounts
- promote one-off guesses into core user facts

User confirmation is required before a candidate affects future runtime behavior.

## UI boundary

Do not add another large management module. The console is already dense.

Preferred UI placement:

- Hermes page: add tabs or filters for `记忆`, `经验`, `反思`, `信念`
- Run dispatch card: show only a compact row such as `经验注入：2 条，跳过 1 条`
- Detail drawer: show why each item was used or skipped
- Confirmation queue: show Chinese one-sentence summary, type, evidence count, confidence, and suggested action

Avoid:

- long inline JSON
- raw event dumps inside cards
- new top-level navigation entry unless the module becomes large enough to justify it
- "Evolution" terminology

## Error handling

The cognitive layer must never block a normal conversation run.

Required behavior:

- reflection failure: log and continue
- experience retrieval timeout: skip retrieval and continue
- invalid experience payload: skip that item and record diagnostic
- contradiction detected: lower confidence or create competing candidate
- context budget exceeded: select highest-scoring items only
- storage unavailable: continue without cognitive updates

## First implementation slice

The first implementation should be intentionally small:

1. Create typed in-memory Cognitive models and service.
2. Convert explicit user feedback and terminal run outcomes into reflection records.
3. Create candidate experiences with confidence, evidence, contradictions, and version.
4. Add an Experience Router that returns selected/skipped items with reasons.
5. Persist records through the existing Hermes admin-resource bucket with `cognitive_*` resource-id prefixes.
6. Expose admin APIs for listing, confirming, rejecting, and deleting candidates.
7. Show compact experience rows in Hermes UI and run detail drawers.
8. Inject only confirmed active experiences through the existing Hermes runtime context path.

This slice proves the full loop without adding vector search, automated Skill installation, or personality mutation.

## Explicit Evolution isolation

The Cognitive Experience Layer must not reuse or revive the old Evolution feature path.

Implementation must not:

- import or call `agent_hub.evolution_hooks`
- write `AdminResourceRow.kind = "evolution"`
- call or create `/api/v1/admin/evolution-runs`
- generate `routing_decision.source = "evolution"`
- route cognitive skill candidates into old Evolution round execution

Cognitive skill candidates remain suggestions until the existing Skill upload/approval flow installs a real Skill.

## Non-goals

- No harness refactor in this repository.
- No direct code editing or Vibe Coding execution.
- No vector database or RAG in the first slice.
- No automatic SOUL/persona mutation.
- No automatic permission changes.
- No third-party memory framework code import.
- No resurrection of the old Evolution module.

## Acceptance criteria

- Candidate experience records are created from meaningful feedback/outcomes, not every chat message.
- Candidate records include confidence, evidence, contradictions, version, usage counts, and verification timestamps.
- Unconfirmed candidates do not affect runtime.
- Confirmed active experiences can be retrieved and injected through bounded context.
- Runtime/scheduler observations remain separated from conversation/user memory.
- A failed cognitive update does not fail the user conversation.
- UI shows Chinese one-sentence summaries and detail drawers without bloating the dispatch card.
- Tests cover positive learning, negative learning, contradiction handling, routing, prompt injection bounds, and UI confirmation.
