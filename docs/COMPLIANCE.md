# Regulatory and industry-standards posture

DiaLog handles personal health information. This document states plainly what
standards inform its design, what it does and does not claim, and where the
gaps are.

**Nothing here is a claim of certification or regulatory clearance.** DiaLog has
not been assessed, audited, certified, or cleared by any regulator,
notified body, or accredited auditor. It carries no CE mark, no FDA clearance
or registration, and no SOC 2, ISO 27001, ISO 13485 or IEC 62304 attestation.
Statements below describe engineering practice, not conformity.

## Device classification

DiaLog is designed to stay outside the definition of a medical device, and this
is a deliberate product boundary rather than an accident of scope.

It does not diagnose, treat, cure, mitigate or prevent disease; it does not
calculate, recommend or adjust a medication dose; and it does not generate
alarms or alerts intended to prompt clinical action. It records what the user
enters or imports, computes descriptive statistics over that record, and
describes what it found.

The boundary is enforced in code, not only in policy:

- `lib/ai/guardrails.ts` applies deliberately over-broad medical-safety regexes
  to every assistant response. A match falls back to a safe template. A false
  positive costs a useful sentence; a false negative would be a dosing
  instruction reaching a patient, so the asymmetry is intentional.
- `MedicationEvent` in the schema is tracking-only — there is no dose
  calculation anywhere in the data model or the application.
- Every statistical finding is graded by sample size in
  `lib/domain/evidence.ts`, and findings below the minimum are never surfaced
  as confident claims.

**Anyone deploying DiaLog in a clinical setting, or extending it toward
decision support, must re-run this classification themselves.** Adding alarms,
dose guidance, diagnostic language, or clinician-facing interpretation is very
likely to change its regulatory status in most jurisdictions.

## Standards that inform the design

These are used as engineering references. Using a standard as a reference is
not conformance to it.

| Area                 | Reference                                                                     | How it shows up here                                                                                                                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accessibility        | **WCAG 2.2 Level AA**                                                         | The one standard DiaLog actively tests against: an `@axe-core/playwright` suite runs against every public and authenticated page in CI. See [ACCESSIBILITY.md](ACCESSIBILITY.md) for implementation and known gaps.                                      |
| Application security | **OWASP ASVS / Top 10**                                                       | Informs the threat model in [SECURITY.md](SECURITY.md): access control, session handling, input validation, dependency risk.                                                                                                                             |
| Health data privacy  | **HIPAA Security Rule**, **PIPEDA/PHIPA** (Canada), **GDPR** where applicable | Informs access control, audit logging, export and deletion. DiaLog is a self-tracking tool holding the user's own data, so the operator's obligations depend on their deployment and jurisdiction.                                                       |
| Software lifecycle   | **IEC 62304**, **ISO 14971**                                                  | Referenced for vocabulary and habits — traceable change history, documented invariants, risk-driven design — not followed as a formal process. There is no design history file, no formal risk management file, and no verification and validation plan. |
| Terminology          | **UCUM** units                                                                | Glucose stored canonically in mg/dL; mass in kg, volume in mL, duration in minutes. Conversion is centralised in `lib/domain/units.ts`.                                                                                                                  |

## Data-protection practices actually implemented

| Practice                                        | Status               | Where                                                                                                                                                    |
| ----------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-user data isolation on every read and write | Implemented, tested  | `lib/db/health-records.ts`; `tests/integration/authorization.test.ts` asserts by content, not count                                                      |
| Encryption in transit                           | Implemented          | HSTS with preload; secure cookies in production                                                                                                          |
| Encryption at rest                              | Deployment-dependent | Provided by the database host, not by the application                                                                                                    |
| Passwords hashed                                | Implemented          | bcrypt, cost 12                                                                                                                                          |
| Session revocation                              | Implemented          | `User.tokenVersion`; "sign out everywhere" and password change both invalidate outstanding cookies                                                       |
| Audit logging of security-relevant actions      | Implemented          | `lib/auth/audit.ts`. Never records health values or free text. Retained with a null user id after account deletion so the trail outlives the account     |
| Right of access / portability                   | Implemented          | Full JSON and per-type CSV export, scoped to the signed-in user                                                                                          |
| Right to erasure                                | Implemented          | Delete-all-records and full account deletion; see the deletion section of [DATA.md](DATA.md) for exactly what each clears                                |
| Data minimisation toward third parties          | Implemented          | The AI layer never receives raw health records — only an evidence-graded bundle — and free text is redacted before any external provider without consent |
| Breach detection and response                   | **Not implemented**  | No alerting, anomaly detection, or incident response process ships with the application                                                                  |
| Formal retention schedule                       | **Not implemented**  | Data is retained until the user deletes it; there is no automatic expiry                                                                                 |

## Known gaps

Stated explicitly so no one mistakes silence for coverage.

- **No third-party security assessment.** No penetration test, no code audit by
  an external party.
- **No third-party accessibility audit.** Automated axe-core coverage catches a
  meaningful subset of WCAG issues, not all of them; there has been no
  assistive-technology user testing.
- **No clinical review.** The guardrail pattern list, the evidence thresholds,
  and the wording of every user-facing statement have not been reviewed by a
  qualified clinician.
- **No formal quality management system.** No QMS, design history file, risk
  management file, or verification and validation plan exists.
- **Rate limiting is per instance.** The in-memory limiter does not coordinate
  across instances, so limits are weaker in a multi-instance deployment. See
  [SECURITY.md](SECURITY.md).
- **No account recovery.** There is no password-reset flow; a user who forgets
  their password and has no linked Google identity cannot recover the account.
- **No monitoring or detective controls.** Nothing detects credential stuffing,
  cross-account probing, or abuse in progress.

## If you are deploying this for real

Treat the list above as prerequisites, not as a backlog. At minimum: get an
independent security assessment, put a shared-store rate limiter in front of
the auth endpoints, establish monitoring and an incident response path, confirm
your database host encrypts at rest and that backups are in scope for
deletion requests, and take your own regulatory advice for your jurisdiction
and intended use. The deployment checklist in [SECURITY.md](SECURITY.md) is the
starting point, not the finish line.
