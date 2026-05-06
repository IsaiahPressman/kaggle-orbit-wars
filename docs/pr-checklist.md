# PR Checklist

Use this checklist before creating, recommending, or merging a PR. It is meant
to make the final agent review repeatable and repository-owned, following the
agent workflow guidance in OpenAI's harness-engineering writeup:
<https://openai.com/index/harness-engineering/>.

When using GitHub CLI commands, run all `gh` commands with elevated permissions;
they are not allowed in the sandbox.

## Final Review

1. Inspect the branch state:

```sh
git status --short --branch
git diff --stat
git diff --check
```

2. Review the actual patch:

```sh
git diff
git diff --cached
```

3. Confirm the changed files match the request. Do not include unrelated
   cleanup, local reference files, downloaded replay fixtures, or generated
   artifacts that are intentionally ignored.

4. Check doc freshness. Mapped code changes should update their mapped docs.
   If `just docs-fresh` fails but the docs are genuinely still current, rerun it
   as `DOCS_CURRENT=1 just docs-fresh` to acknowledge that review.

5. Review the patch for duplicated logic, especially copied rules-engine
   formulas, geometry, validation, scoring, and terminal-condition behavior.
   Prefer shared helpers for production code; keep intentionally independent
   oracle logic only when it is clearly serving parity validation.

6. Run the strongest relevant prepare target:

```sh
just prepare
```

Use `just py-prepare` or `just rs-prepare` only when the change is strictly
limited and a full prepare would not add signal. If only the doc freshness
portion of `just prepare` fails and the mapped docs are genuinely still current,
rerun it as `DOCS_CURRENT=1 just prepare`.

7. For rules-engine changes, confirm:
   - Replay fixtures are present in `tests/fixtures/orbit_wars_replays`.
   - `tests/fixtures/generation/reference_generation.json` is current if
     generation behavior changed.
   - `docs/rules-engine.md` describes the current implementation state.
   - `docs/rules-parity-coverage.md` still describes the real coverage and
     residual gaps.

8. For RL API changes, confirm `docs/rl-api-specs.md` still matches the public
   Python config shape, tensor shapes, channel order, and action semantics.

9. Summarize residual risk in the final response or PR body. If no meaningful
   risk remains, say so directly.

10. After a successful merge, switch back to the base branch and clean up the
    merged feature branch locally and remotely, as long as doing so will not
    disturb unrelated local work.

## Agent Review Gate

For non-trivial changes, run a final reviewer pass before PR creation. The
review should focus on bugs, missing tests, stale docs, orphaned code, fixture
freshness, and unintended files. Address concrete findings before declaring the
branch ready.
