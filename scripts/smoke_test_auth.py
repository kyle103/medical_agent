from __future__ import annotations

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> None:
    phone = "13800000000"
    password = "Passw0rd!"

    # register (may fail if already registered)
    r = httpx.post(
        f"{BASE}/api/v1/user/register",
        json={"phone": phone, "password": password, "user_nickname": "test"},
        timeout=10,
    )
    print("register:", r.status_code)
    print(r.text)

    r2 = httpx.post(
        f"{BASE}/api/v1/user/login",
        json={"phone": phone, "password": password},
        timeout=10,
    )
    print("login:", r2.status_code)
    print(r2.text)


if __name__ == "__main__":
    main()
