# Deployment runbook

## Current installation

The protected `adfm-python-api` Render web service runs the native GitHub migration branch in Virginia. It is on the free plan for staging. Deployment successfully started on September 7, 2026. Health and authentication smoke checks passed. The production website source has native `/tools` changes in development, not yet published.

The website uses server environment values `ADFM_API_ORIGIN`, `ADFM_GATEWAY_TOKEN`, `ADFM_TOOLS_ACCESS=private`, and `ADFM_TOOLS_ALLOWED_EMAILS`. The gateway token is generated randomly, stored as a secret in both hosts, and never shipped in browser bundles. Allowlist includes the owner's ChatGPT account `aryadeniz@yahoo.com` and work account `aryadeniz@adfundmgmt.com`. The existing website's dispatch-owned ChatGPT sign-in identifies the visitor; API authorization stays server-side. If hosting moves away from Sites, replace this identity boundary with verified OIDC/session middleware; never trust a browser-supplied identity header on a generic origin.

## Run locally

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-native-dev.txt
ADFM_ENV=development .venv/bin/python -m uvicorn adfm_api.main:app --host 127.0.0.1 --port 8080
```

Development without a gateway token is allowed only when `ADFM_ENV=development`. Production fails startup if the token is missing or shorter than 32 characters.

```bash
.venv/bin/python -m pytest native_tests -q
.venv/bin/python scripts/package_native_runtime.py
.venv/bin/python -m pip check
```

Website checkout: `npm ci`, `npm run build`, and `node --experimental-strip-types --test tests/analytics-gateway.test.mjs tests/rendered-html.test.mjs`. Use existing website build/hosting tooling. API routes are `/v1/rate-of-change`, `/v1/overview`, `/v1/leadership`, `/v1/relative-volatility`, `/v1/ratios`, `/v1/macro-regime`; browser equivalents use `/tools/api/`.

## Render runtime

`render.yaml` describes the service. Native build installs only `requirements-native.txt` and runs `scripts/package_native_runtime.py`. Startup changes into `.native-runtime` and runs Uvicorn on `0.0.0.0:$PORT`, one worker and a concurrency limit. The Dockerfile is a portable alternative with a non-root runtime and the same source allowlist. Do not install `requirements.txt` or start `Home.py` on this service.

The direct creation connector does not expose a health-check-path argument. The deployed service initially uses Render's default port check; the Blueprint specifies `/health/live`, which should be configured in the dashboard before final cutover. The endpoint is already implemented and verified. HTTP health proves process readiness only, not Yahoo/FRED availability; provider status belongs in page results and monitoring.

## Domain and paths

GoDaddy manages DNS, not URL paths. An A record or CNAME cannot route `/tools` separately. Keep the existing apex domain pointed at the existing website host. Its Worker handles `/tools` pages and `/tools/api/*`; the latter forwards to the Render HTTPS origin. No Streamlit redirect, iframe, embedded page or new user-facing subdomain is involved.

An optional `api.adfundmgmt.com` custom hostname would need a GoDaddy CNAME matching the value Render supplies, plus Render domain verification/TLS. It is unnecessary for this architecture: the gateway can use the existing `onrender.com` hostname privately. If moving the whole website later, preserve other site routes and use a reverse proxy with explicit `/tools/api/` forwarding.

## Secrets and scheduled data

| Variable | Consumer | Need |
|---|---|---|
| ADFM_GATEWAY_TOKEN | Website gateway and Python API | Required in production; same random value |
| ADFM_ENV | API | `production` on Render |
| ADFM_API_ORIGIN | Website server | HTTPS API origin only |
| ADFM_TOOLS_ACCESS | Website server | `private` for internal platform |
| ADFM_TOOLS_ALLOWED_EMAILS | Website server | Explicit comma-separated membership allowlist |
| FRED_API_KEY | Later CTE/provider adapters | Required where API-specific adapters use it; current FRED CSV slice has no key |
| ESTAT_APP_ID | Later CTE Japan adapter | Required for that adapter |
| ANTHROPIC_API_KEY | Optional CTE commentary | Optional, never required for core calculations |
| SEC_USER_AGENT / ADFM_SEC_USER_AGENT | Later SEC loaders | Identifiable SEC requests, preserving each adapter's variable mapping |
| ADFM_DATA_DIR | Later SEC bulk cache | Writable persistent directory or migrated object-store location |
| STOOQ_API_KEY / TRADING_ECONOMICS_API_KEY | Later credit sovereign loaders | Only for selected fallback providers |

The 15-page release uses a lightweight SQLite background-job queue and prepared SEC parquet archives. Set `ADFM_DATA_DIR` to a persistent disk mount for production. Run one process and one instance; the disk cannot be shared across instances. Keep temporary yfinance timezone/cookie caches on a writable temporary directory. Later CTE and SEC jobs should import the same analytics/data packages, publish validated immutable snapshots with timestamps and source hashes, and atomically advance a latest pointer. Retain a last-good version and surface age on failure. Use the existing scheduled GitHub Actions first where adequate; add a Render cron service only for jobs that need host-local credentials/resources. A simulation/watchlist database is added when those user workflows migrate, with user-scoped records and migrations.

## Cost planning

Budget approximately **$25–40 per month incremental infrastructure** for an always-on Python API with around 2 GB memory, small snapshot storage and light scheduled refreshes, subject to profiling the large pages. This is a planning allowance, not a quoted purchased plan. Existing website/domain costs and paid market-data subscriptions are additional. The current API staging service uses the free instance.

Render's official July 2026 example puts a Starter web service plus a small Postgres instance at roughly $13/month before growth, but the full ADFM workload should be memory-profiled before selecting such a small instance: [Render cost guide](https://render.com/articles/how-much-does-cloud-application-hosting-cost-for-small-businesses). Current plan/compute charges are published at [Render pricing](https://render.com/pricing).

Free services sleep after inactivity and have monthly instance-hour limits, so use them for staging rather than the final reliability target: [Render free-service limits](https://render.com/docs/free). Do not pay for Redis, a dedicated worker, and a database merely to deploy the first stateless pages.

## Release and rollback

Keep source and built website artifacts matched. Save reviewable website versions while migrating, then publish the complete version after all pages pass the agreed gates. Record the website version and API revision as one release manifest. API schema changes should remain backward compatible during rollout. Roll back to the prior matched native version on failure; retain original reference snapshots until validation is complete. Retire the Streamlit service only after the full native platform passes acceptance. Remove Streamlit from active deployment dependencies, workflows and runtime source at that point.

A September 7 source update did not start a deployment despite `autoDeploy=yes`; a clean-cache deployment was triggered and verified live at the updated revision. Verify the deployed commit explicitly during releases instead of assuming the GitHub webhook is connected.

## Revised release scope
See RELEASE-15.md: fifteen analytics pages plus Home are the agreed release. Other pages are deferred by the user. Publish this release after its gates pass.
