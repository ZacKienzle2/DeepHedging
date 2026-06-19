# Contributing

## Workflow

1. Branch from `main` using semantic names: `feat/<ticket-id>`, `fix/<ticket-id>`.
2. One logical change per commit, following Conventional Commits 1.0.0.
3. Open a pull request; reference the issue number so it auto-closes on merge.
4. CI must pass and the PR must be reviewed before merge.
5. Keep history linear. Rebase feature branches on `main`, and squash fixups with
   `git rebase -i` before merge. Never rebase shared branches.

## Commit Messages

```
<type>[(scope)][!]: <description>

[body]

[footer(s)]
```

- Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
  `chore`, `style`, `revert`.
- Description: imperative, lowercase, no trailing period.
- Breaking changes: `!` in the header or a `BREAKING CHANGE:` footer.
- ASCII only. Signed commits required.

## Code Standards

- Python: strict PEP 8, Google-style docstrings on public APIs, no inline
  comments, since code should be self-evident.
- Tests accompany every behavioural change (`pytest`).
- Profile before optimising; justify performance-motivated complexity.

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] New code covered by tests
- [ ] Docs updated where behaviour changed
- [ ] Commits follow Conventional Commits
