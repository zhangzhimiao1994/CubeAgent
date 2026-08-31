# Workspace Rules

- This repository is the active local continuation of CubeAgent/Agent Hub under `E:\code_x\mofangagent`.
- This project's only valid GitHub repository is `https://github.com/zhangzhimiao1994/CubeAgent` (`ssh://git@ssh.github.com:443/zhangzhimiao1994/CubeAgent.git` for SSH pushes). Do not push this project to `Cube-agent`, `mutilagent`, or any other repository.
- This repository is the pure conversation-agent product line. It may contain Hermes+, Chinese learning summaries, confirmed memory retrieval/injection, conversation routing, subagent observation UI, file delivery, and operational controls. It must not implement the future OpenAI/DeepSeek harness refactor or claim harness-level Vibe Coding execution capability.
- Harness work belongs in a separate future project/repository and must not be mixed into this CubeAgent repository.
- The production server remains the SSH host `prod-web-01`; the active release pointer is `/opt/agent-hub/current`.
- For runtime-affecting changes, deploy to `prod-web-01` and run a real feature-specific probe before pushing GitHub.
- Do not change public URL, port forwarding, secrets, provider credentials, or quota-consuming provider behavior without explicit user authorization.
- Keep `HANDOFF.md`, `task_plan.md`, `findings.md`, and `progress.md` local-only. Never commit secrets, virtual environments, caches, temporary probes, deployment packages, or local archives.
- Use test-driven development for features and bug fixes: add a focused regression, observe the expected failure, implement the smallest fix, then run focused and broader checks.
- After pushing GitHub, inspect the triggered run/check. If it fails, retrieve details, fix, verify, push, and repeat until green or a real external blocker is documented.
- If Docker, Docker Desktop, Docker Compose, or WSL2-backed containers are used, close Docker Desktop and shut down the WSL Docker backend after verification unless the user explicitly asks to keep them running.
- Before finishing Docker-related work, verify cleanup with `Get-Process '*docker*','vmmem*','wsl*'` and `wsl -l -v`; ask for approval if cleanup requires it.
- After every completed work slice, append a handoff entry containing current state, changes, verification, production details, remaining risks, and enough context to continue safely.
