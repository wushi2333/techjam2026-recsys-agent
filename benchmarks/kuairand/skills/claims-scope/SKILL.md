---
name: claims-scope
description: Treat every measurement as claim plus parent scope. Use when a 1-seed CI is negative or a flag looks banned. Do not promote a parent-scoped fail into a family ban.
arm: any
keys: ""
status: wired
---

# Claims with scope

Every run_facts falsified line is 1-seed on **that parent**. A new incumbent identity may retry the same patch.

Write (and read) cards as `{claim, evidence node, scope=parent, status}`. `tried_canonical_patches` is keyed by parent_id. Do not copy a 1K-FM fail onto Pure DIN.
