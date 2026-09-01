# Hermes Learning

Hermes is an experience layer, not online model training.

It stores safe lessons from previous runs and uses them to recommend:

- dispatch vs group-chat mode
- model candidates
- skill candidates
- risk and approval hints

Hermes never stores obvious secrets, never bypasses approvals, and never executes actions directly.

## Scope Model

Hermes and Cognitive experiences use two memory scopes:

- `user`: applies only to the user that created or confirmed the memory.
- `root`: applies to every user in the same tenant and should be reserved for stable project-wide rules, environment facts, or repeatedly verified operating lessons.

Runtime injection only considers confirmed records visible to the current actor: the actor's `user` memories plus tenant-level `root` memories. A confirmed memory is still advisory; it cannot override the current user request, safety policy, capability checks, model availability, or tool permissions.

When older Hermes records without scope metadata are confirmed, they are backfilled as `user` memories for the confirming user instead of becoming tenant-wide memory by accident.

## Runtime Acceptance

Full-chain tests should cover the complete loop:

1. Create a reusable experience candidate.
2. Confirm it.
3. Submit a new run whose task matches the experience.
4. Verify `routing_decision.hermes.injected_memories` contains the confirmed experience.
5. Execute the runtime and verify the bounded `HERMES_MEMORY_CONTEXT` appears in the model prompt.
