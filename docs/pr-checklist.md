# PR Checklist

Use this checklist before creating, recommending, or merging a PR. It is meant
to make the final agent review repeatable and repository-owned, following the
agent workflow guidance in OpenAI's harness-engineering writeup:
<https://openai.com/index/harness-engineering/>.

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

4. Run the strongest relevant prepare target:

```sh
just prepare
```

Use `just py-prepare` or `just rs-prepare` only when the change is strictly
limited and a full prepare would not add signal.

5. For rules-engine changes, confirm:
   - Replay fixtures are present in `tests/fixtures/orbit_wars_replays`.
   - `tests/fixtures/generation/reference_generation.json` is current if
     generation behavior changed.
   - `docs/rules-engine-plan.md` describes the current implementation state, not
     only historical plans.
   - `docs/rules-parity-coverage.md` still describes the real coverage and
     residual gaps.

6. For RL API changes, confirm `docs/rl-api-specs.md` still matches the public
   Python config shape, tensor shapes, channel order, and action semantics.

7. Summarize residual risk in the final response or PR body. If no meaningful
   risk remains, say so directly.

## Agent Review Gate

For non-trivial changes, run a final reviewer pass before PR creation. The
review should focus on bugs, missing tests, stale docs, orphaned code, fixture
freshness, and unintended files. Address concrete findings before declaring the
branch ready.
