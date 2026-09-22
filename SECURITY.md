# Security Policy

## Table of Contents

- [Supported Versions](#supported-versions)
- [Reporting a Vulnerability](#reporting-a-vulnerability)
- [What to Include](#what-to-include)
- [Response Process](#response-process)
- [Disclosure Policy](#disclosure-policy)
- [Severity Classification](#severity-classification)
- [Scope](#scope)
- [Out of Scope](#out-of-scope)
- [Safe Harbour](#safe-harbour)
- [Recognition](#recognition)
- [Encrypted Communication](#encrypted-communication)

## Supported Versions

Only the listed versions receive security updates. Older versions will not
receive backports.

| Version | Supported |
| ------- | --------- |
| main    | Yes       |
| < main  | No        |

This table will be updated once tagged releases exist.

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues,
discussions, pull requests, or any other public channel.**

Private channels:

1. **GitHub Private Vulnerability Reporting** (preferred):
   [report a vulnerability](https://github.com/ZacKienzle2/DeepHedging/security/advisories/new),
   which is the repository's Security tab.
2. **Email**: maintainer's published security contact. Encrypt sensitive details
   with the key referenced in
   [Encrypted Communication](#encrypted-communication).

Acknowledgement within 7 days. Follow up if not received.

## What to Include

- A description of the vulnerability and its potential impact.
- The type of issue (buffer overflow, injection, auth bypass, cryptographic
  flaw, etc.).
- The full path of the source files related to the issue.
- The location of the affected source code (tag, branch, commit, or direct URL).
- The version, commit SHA, or release tag where the issue was discovered.
- Any special configuration required to reproduce.
- Step-by-step instructions to reproduce the issue.
- Proof-of-concept code or exploit, if available.
- Logs, screenshots, or network captures supporting the report.
- An assessment of the impact, including how an attacker might exploit it.
- Suggested mitigation or remediation, if you have one.
- Your contact information for follow-up.

## Response Process

Timeline:

| Stage                  | Target                                            |
| ---------------------- | ------------------------------------------------- |
| Acknowledgement        | Within 7 days                                     |
| Initial assessment     | Within 14 days                                    |
| Status update cadence  | Every 14 days                                     |
| Fix availability       | Severity-based, see below                         |
| Coordinated disclosure | Within 90 days of report, unless agreed otherwise |

Process:

1. Maintainer acknowledges receipt.
2. Maintainer reproduces and assesses severity.
3. Maintainer develops a fix in a private branch.
4. Maintainer coordinates a disclosure timeline with the reporter.
5. Fix is merged, released, and disclosed publicly.

## Disclosure Policy

- We follow a coordinated disclosure model.
- Default embargo: 90 days from report, or earlier upon release of a fix.
- Embargoes may be extended for complex fixes by mutual agreement.
- We will credit you in the advisory unless you request anonymity.
- We will request a CVE identifier where applicable.

## Severity Classification

We use the CVSS v3.1 framework to score vulnerabilities. Target remediation
windows:

| Severity | CVSS Range | Target Fix Window |
| -------- | ---------- | ----------------- |
| Critical | 9.0 - 10.0 | 7 days            |
| High     | 7.0 - 8.9  | 30 days           |
| Medium   | 4.0 - 6.9  | 60 days           |
| Low      | 0.1 - 3.9  | 90 days           |

## Scope

In scope:

- Source code in this repository.
- Build, release, and packaging artefacts produced by this repository.
- CI/CD configuration in this repository.
- Documentation that could mislead users into insecure configurations.

## Out of Scope

The following are generally out of scope and will be closed without action:

- Vulnerabilities requiring physical access to a user's device.
- Social engineering of project maintainers or users.
- Denial of service attacks, including volumetric and resource exhaustion,
  unless they expose data or escalate privileges.
- Vulnerabilities in third-party dependencies without a demonstrated,
  project-specific impact. Report those to the upstream project.
- Findings from automated scanners without a working proof of concept.
- Missing best-practice configurations without a demonstrated exploit (for
  example, missing HTTP security headers on non-sensitive endpoints).
- Issues affecting end-of-life or unsupported versions.
- Self-inflicted issues that require the victim to perform highly unusual
  actions.

## Safe Harbour

Security research conducted in good faith. No legal action against researchers
who:

- Make a good-faith effort to comply with this policy.
- Avoid privacy violations, destruction of data, and interruption or degradation
  of services.
- Only interact with accounts you own or have explicit permission to access.
- Do not exfiltrate data beyond the minimum required to demonstrate the
  vulnerability.
- Give us reasonable time to respond before any public disclosure.
- Do not exploit the vulnerability beyond what is necessary to confirm it.

If legal action is initiated by a third party against you for activity that
complied with this policy, this authorisation will be made known.

## Recognition

Researchers reporting previously unknown, valid vulnerabilities are credited in
the security advisory and listed in the Hall of Fame unless they request
otherwise. No monetary rewards.

## Encrypted Communication

For sensitive reports, request the project's PGP key by email before sending
exploit details. Key fingerprint and revocation status will be published in the
repository once the key is generated.
