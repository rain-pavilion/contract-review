"""Temporary accounts and real cookie sessions for legacy administrator routes."""

import http.cookiejar
import json
from pathlib import Path
import urllib.request

from zhilv_qiling_v2.contract_library import ContractLibrary


def prepare_admin(config, root):
    """Keep all three libraries in the test's temporary directory."""
    root = Path(root)
    config.contract_library_path = str(root / "contracts.sqlite")
    config.law_library_path = str(root / "laws.sqlite")
    config.user_library_path = str(root / "users.sqlite")
    store = ContractLibrary(config.contract_library_path, users_path=config.user_library_path)
    store.bootstrap_admin("fixture-admin", "Synthetic-fixture-admin-93825!")


def admin_session(base):
    """Log in through HTTP; neither authentication nor CSRF checks are mocked."""
    client = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    request = urllib.request.Request(base + "/api/auth/login",
        data=json.dumps({"username": "fixture-admin",
                         "password": "Synthetic-fixture-admin-93825!"}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with client.open(request, timeout=10) as response:
        payload = json.load(response)
        if response.status != 200 or not payload.get("csrf_token"):
            raise AssertionError("Temporary administrator login failed")
    return client, payload["csrf_token"]
