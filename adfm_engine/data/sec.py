from __future__ import annotations
import os,time,requests
from typing import Any,Mapping
from adfm_engine.analytics.sec_fundamentals import SecDataError,SEC_DATA_BASE,SEC_TICKER_URL,DEFAULT_SEC_USER_AGENT
class SecClient:
    """Small fair-access EDGAR client with retries and an explicit user agent."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        *,
        timeout_seconds: float = 20.0,
        retries: int = 3,
        retry_pause_seconds: float = 0.7,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.user_agent = (
            user_agent
            or os.getenv("SEC_USER_AGENT")
            or DEFAULT_SEC_USER_AGENT
        ).strip()
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, int(retries))
        self.retry_pause_seconds = max(0.0, float(retry_pause_seconds))
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }

    def get_json(self, url: str) -> Mapping[str, Any]:
        headers = dict(self.headers)
        if url.startswith(SEC_DATA_BASE):
            headers["Host"] = "data.sec.gov"
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise SecDataError(f"SEC returned a non-object payload for {url}")
                return payload
            except (requests.RequestException, ValueError, SecDataError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_pause_seconds * (attempt + 1))
        raise SecDataError(f"SEC request failed for {url}: {last_error}")

    def company_tickers(self) -> Mapping[str, Any]:
        return self.get_json(SEC_TICKER_URL)

    def company_facts(self, cik: int) -> Mapping[str, Any]:
        return self.get_json(
            f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
        )

    def submissions(self, cik: int) -> Mapping[str, Any]:
        return self.get_json(f"{SEC_DATA_BASE}/submissions/CIK{int(cik):010d}.json")
