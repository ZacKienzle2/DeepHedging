# Governance

Decision-making model for the project. Lightweight; evolves with the community.

## Roles

### Users

Use the project. Contribute via bug reports, feature requests, discussions.

### Contributors

Submit contributions (code, docs, tests, design). Recorded by git, and listed on
the repository's contributors page after first merge.

### Maintainers

Merge rights. Review PRs, triage issues, day-to-day decisions. Current list in
[CODEOWNERS](.github/CODEOWNERS).

Responsibilities:

- Uphold and enforce the [Code of Conduct](CODE_OF_CONDUCT.md).
- Review contributions promptly and fairly.
- Communicate decisions transparently.
- Mentor new contributors.

### Project Lead

Final say on contested decisions, technical direction, external representation.
Identified at top of [CODEOWNERS](.github/CODEOWNERS).

## Decision Making

Consensus preferred. Escalation path:

1. **Author proposes** in a pull request, issue, or discussion.
2. **Lazy consensus** - if no maintainer objects within 7 days, the proposal is
   accepted.
3. **Active discussion** - if objections arise, work toward consensus in the
   thread.
4. **Maintainer vote** - if consensus cannot be reached, maintainers vote.
   Simple majority wins. Ties go to the project lead.
5. **Project lead decides** - the lead may override in exceptional circumstances
   and must publish the reasoning.

Routine changes (bug fixes, refactors, docs): one maintainer approval, no vote.

Significant changes (architectural shifts, breaking API, new top-level features,
license-affecting deps): two maintainer approvals + 7-day comment period.

## Adding and Removing Maintainers

### Adding

Nomination by any maintainer after:

- 3+ months of high-quality contributions.
- Familiarity with codebase and conventions.
- Constructive review and discussion participation.

Simple majority vote of maintainers within 7 days.

### Stepping down

Notify project lead, update [CODEOWNERS](.github/CODEOWNERS).

### Removal

Sustained inactivity (12 months) or repeated Code of Conduct violations.
Two-thirds vote of remaining maintainers.

## Conflicts of Interest

Maintainers disclose before approving changes that materially benefit
themselves, employer, or represented organisation. Disclosure suffices; recusal
at maintainer's discretion unless direct and material.

## Code of Conduct Enforcement

Reports via the channel in [SECURITY.md](SECURITY.md). Project lead owns
enforcement decisions; may delegate.

## Changes

Two-thirds maintainer vote + 14-day public comment period.
