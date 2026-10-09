# AGENTS.md

Durable working instructions for coding agents in this repository.

## Repository-level instructions
- When learning durable, repo-specific workflow/preferences, record them in this file.
- If something is likely a cross-repo preference, call it out explicitly as a **candidate for global custom instructions** in agent responses.
- If a missing tool/package/environment setup in the active conda env blocks progress or causes significant slowdown, report:
  - exactly what to install,
  - the install command(s), and
  - why it is needed for this repo.
- If user guidance appears technically incorrect, potentially harmful, or likely to produce a wrong fix, say so directly with technical reasoning before implementing.

## Task kickoff behavior
- At the start of each new task, ask clarifying questions only for uncertainties that materially affect:
  - correctness,
  - interfaces/public APIs,
  - edge-case behavior, or
  - design direction.
- Do **not** block on minor uncertainties. Make reasonable assumptions, state them briefly, and proceed.

## Notes
- Keep this file concise and focused on stable instructions that persist across tasks.
- Prefer updating this file over repeating the same meta-instructions in every conversation.
