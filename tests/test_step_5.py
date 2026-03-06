"""Тест Step 5: Litestar API."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests

BASE = "http://localhost:8050"

def test():
    print("=" * 55)
    print("ТЕСТ ШАГ 5: Litestar API")
    print("=" * 55)

    # Health
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"Health failed: {r.text}"
    print(f"✅ Health: {r.json()}")

    # Свечи
    r = requests.get(f"{BASE}/api/candles?ticker=SBER&interval=1h&days=7")
    assert r.status_code == 200, f"Candles failed ({r.status_code}): {r.text[:300]}"
    d = r.json()
    print(f"✅ Свечи: {len(d['candles'])} шт | последняя: {d['candles'][-1]['close']} ₽")

    # Формула — ок
    r = requests.post(f"{BASE}/api/formula", json={
        "ticker": "SBER", "interval": "1h", "days": 7,
        "formula": "RESULT = EMA(9) - EMA(21)", "name": "EMA Diff"
    })
    assert r.status_code == 201, f"Formula failed ({r.status_code}): {r.text[:300]}"
    d = r.json()
    print(f"✅ Формула: {d['name']} | {len(d['points'])} точек | last={d['last']}")

    # Формула — ошибка
    r = requests.post(f"{BASE}/api/formula", json={
        "ticker": "SBER", "interval": "1h", "days": 7,
        "formula": "BAD CODE +++", "name": "Bad"
    })
    d = r.json()
    print(f"✅ Ошибка формулы поймана: {d['error'][:50]}...")

    # OpenAPI docs
    r = requests.get(f"{BASE}/schema")
    print(f"✅ OpenAPI docs: {r.status_code == 200}")

    print("\n" + "=" * 55)
    print("🎉 ШАГ 5 ЗАВЕРШЕН!")
    print(f"   Браузер: {BASE}")
    print(f"   API docs: {BASE}/schema")
    print("=" * 55)


if __name__ == '__main__':
    test()
    