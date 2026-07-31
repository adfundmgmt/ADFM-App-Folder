# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or an accidental data
exposure. Contact the repository owners through the firm's established private
security channel and include the affected component, reproduction steps, impact,
and any safe remediation you have identified.

## Data handling

This repository must not contain credentials, client information, positions,
holdings, account identifiers, or other confidential portfolio data. Use
deployment secrets for provider credentials and approved private storage for
portfolio inputs. The repository ignores common local portfolio and position
paths as a last line of defense; those ignore rules do not replace a pre-commit
review.

Cached market and macro snapshots committed to the repository must contain only
public-source data, pass the repository snapshot validator, and include the
generated manifest. If confidential data is committed, treat it as exposed:
notify the repository owners, rotate affected credentials, and follow the firm's
incident-response process.

## Supported version

Security fixes are applied to the current `main` branch. Changes should enter
through a reviewed pull request with required CI checks.
