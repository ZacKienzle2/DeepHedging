# Contributing

Contributions accepted: bug reports, feature requests, documentation, tests, and
code. All contributors must follow the workflow below.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ground Rules](#ground-rules)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Workflow](#development-workflow)
- [Branching Strategy](#branching-strategy)
- [Commit Message Convention](#commit-message-convention)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Review](#code-review)
- [Release Process](#release-process)
- [License](#license)

## Code of Conduct

Governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report violations
privately to the maintainers via the channel in [SECURITY.md](SECURITY.md).

## Ground Rules

- Cross-platform compatibility: Linux, macOS, Windows.
- CI green before review request.
- Open an issue before non-trivial changes.
- One logical change per commit.
- Atomic PRs, preferably one feature per release.

## Getting Started

### Prerequisites

- Git 2.30 or newer
- [pre-commit](https://pre-commit.com), which installs every formatter and
  linter the repository uses into its own cache
- The toolchain the README names

### Local Setup

```bash
git clone https://github.com/ZacKienzle2/DeepHedging
cd DeepHedging
pre-commit install --install-hooks
```

The hooks run on every commit and refuse one that fails a check; they are also
what CI runs, so a commit that passes locally passes there.

## How to Contribute

### Reporting Bugs

Before submitting:

- Check the [issue tracker](https://github.com/ZacKienzle2/DeepHedging/issues)
  for duplicates.
- Reproduce against the latest `main`.
- Collect reproduction info.

Use the bug report template. Include:

- Descriptive title.
- Exact reproduction steps.
- Observed vs expected behaviour.
- Screenshots, logs, sample inputs where relevant.
- Environment: OS, toolchain version, project version or commit SHA.

### Suggesting Enhancements

Use the feature request template. Include:

- Descriptive title.
- Detailed description of the proposed enhancement.
- Motivation and use case.
- Alternatives considered.

### Your First Code Contribution

Issues tagged:

- `good first issue` - small scope, requires a few lines plus a test.
- `help wanted` - larger scope than `good first issue`.

### Pull Requests

1. Fork, branch from `main`.
2. Code changes ship with tests.
3. API changes ship with documentation.
4. Test suite passes.
5. Hooks pass.
6. Open PR, fill in the template.

## Development Workflow

1. Sync your fork with upstream `main`.
2. Create a feature branch with a semantic name.
3. Implement the change in small, atomic commits.
4. Rebase against `main` to resolve conflicts.
5. Open a pull request once the hooks pass locally.
6. Address review feedback with fixup commits, then squash before merge.

## Branching Strategy

- `main` - always deployable, protected, requires green CI and peer review.
- `feat/<ticket-id-or-slug>` - new features.
- `fix/<ticket-id-or-slug>` - bug fixes.
- `perf/<slug>` - performance improvements.
- `refactor/<slug>` - non-behavioural refactors.
- `docs/<slug>` - documentation only.
- `test/<slug>` - test additions or fixes.
- `chore/<slug>` - tooling, dependencies, housekeeping.

Never force-push to `main` or any shared branch.

## Commit Message Convention

This project follows
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
The commit-msg hooks refuse a message that does not. ASCII only. No emoji, smart
quotes, or em dashes.

Format:

```text
<type>(<scope>)[!]: <description>

[body]

[footer(s)]
```

Allowed types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
`chore`, `style`, `revert`. The scope is mandatory.

Rules:

- Subject: imperative, lowercase, no trailing period, 72 characters or fewer.
- Blank line between subject, body, and footers.
- Body: wrap at 72 characters. Explain motivation and contrast with prior
  behaviour. Do not restate the diff.
- One logical change per commit. `fix` only for real defects.
- Breaking changes: append `!` after the type or include a `BREAKING CHANGE:`
  footer.
- No tool attribution: no co-author trailer or generated-with line.

Example:

```text
feat(parser): support nested arrays

Extends the tokenizer to recognise bracket depth so deeply nested
literals parse without backtracking.

Refs #142
```

## Coding Standards

- Match the style of the surrounding code.
- The hooks format and lint before every commit; do not hand-format.
- Self-evident code. Avoid inline comments. Document public APIs with the
  project's chosen docstring style.
- Production-ready code on the first pass. No commented-out blocks, no debug
  prints, no TODOs without a referenced issue.
- ASCII only in committed text, code, and commit messages.

## Testing

- Every behavioural change ships with tests.
- Bug fixes include a regression test that fails before the fix and passes
  after.
- Keep tests deterministic. Avoid sleeps, network calls, and clock dependencies
  in unit tests.
- Mirror the source tree in the test tree.

## Documentation

- Update the README and any affected docs in the same commit as the code change.
- Document public APIs.
- Document breaking changes in the pull request description.

## Code Review

Reviewers check:

- Correctness, including edge cases and error handling.
- Test coverage of new behaviour and regressions.
- API design, naming, and backward compatibility.
- Performance implications in hot paths.
- Security implications, including input validation and secret handling.
- Documentation parity with the code change.

Authors:

- Respond to every comment, either with a change or with reasoning.
- Resolve conversations only after the reviewer is satisfied.
- Re-request review after addressing feedback.

## Release Process

Release notes are drafted by
[release-drafter](https://github.com/release-drafter/release-drafter) from pull
request labels as pull requests merge. `CHANGELOG.md` is generated from the
Conventional Commit history by [git-cliff](https://git-cliff.org/) (configured
in `cliff.toml`); preview the unreleased entries at any time with
`git cliff --unreleased`.

1. Bump the version using SemVer based on the commit history.
2. Tag the release commit with `vMAJOR.MINOR.PATCH`.
3. Pushing the tag runs the release workflow, which publishes the drafted
   release with the SBOMs and the regenerated changelog attached.

## License

Contributions licensed under the project's licence.
