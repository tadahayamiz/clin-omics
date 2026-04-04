# private_docs

This directory stores private planning and implementation-control documents for `clin-omics`.

These files are not part of the public library API.
They exist to keep implementation order, scope control, and handoff quality explicit during iterative development.

Current top-level roles:

- `IMPLEMENTATION_SCHEDULE.md`
  - source-of-truth progress ledger
  - current phase
  - next one-theme task
  - strict vs temporary distinctions
- `NEXT_CHAT_HANDOFF.md`
  - short operational handoff for the next chat
  - what was just decided
  - what to do next
  - what not to touch yet

Operational rules for this repo:

- 1 request = 1 theme
- prefer the smallest possible modification scope
- docs-only turns should remain docs-only
- temporary compatibility shims must be labeled explicitly
- completed items remain in the schedule as history
