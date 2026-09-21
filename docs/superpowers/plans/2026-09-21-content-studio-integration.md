# Content Studio Integration

## Scope

Continue the existing Content Studio module in CubeAgent. Short-drama asset
quality work remains paused. Keep the existing orchestrator, capability router,
authorization, persistence, and media rendering boundaries.

## Acceptance

- Every entry point respects script, asset-rights, and final approval gates.
- Projects are bound to their authenticated tenant and owner.
- Retries preserve successful expensive work; changes invalidate dependencies.
- Demo results are explicitly labeled. Missing production adapters block with a
  useful reason; fabricated evidence or artifact IDs never count as production.
- A browser refresh can recover a project and its approval state.
- Real media adapters return playable artifacts and measure technical QC.
- Research adapters treat retrieved material as untrusted data and require
  verifiable evidence references before accepting factual statements.
- CI uses fake providers; external paid services are not invoked by these tests.

## Work Packages

1. Core state, approvals, targeted invalidation, checkpoints, and production mode.
   Owner: Russell. Files: content_studio/__init__.py and core/persistence tests.
2. API and repository tenant/owner isolation, ownership regression tests.
   Owner: Turing. Files: content_studio API, repository, API/repository tests.
3. Capability gateway identity propagation and explicit production wiring.
   Owner: Carson. Files: gateway, app/worker construction, gateway tests.
4. Recoverable workbench, approval controls, and UI tests.
   Owner: Boyle. Files: ContentStudioPage, client contracts, scoped styles/tests.
5. Test-runner diagnosis, broader regression evidence, integration guidance.
   Owner: Gibbs. Do not alter business code.
6. Local media adapter and measured QC with deterministic media tests.
   Owner: media implementer. Files: content_studio/media.py and media tests.
7. Research adapter with source validation, claim verification, and contracts.
   Owner: research implementer. Files: content_studio/research.py and tests.
8. Independent review, integration verification, concise handoff update.
   Owner: controller and Averroes.

## Decisions

- Keep the work in the existing dirty worktree and preserve unrelated changes.
- Reuse existing subagents inside this task; do not create top-level tasks.
- Default runtime behavior must not silently run demonstration providers.
- A local demonstration clip proves rendering and QC only; it does not prove
  factual research, natural voice generation, or real-provider production.
- Do not push or deploy incomplete Content Studio results as an accepted release.
