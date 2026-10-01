"""팀에서 준비한 정답 포함 CSV를 시뮬레이션 API로 전송한다."""
import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import load_rows


def main():
    parser = argparse.ArgumentParser(description="ViewPilot CSV 평가 시뮬레이션")
    parser.add_argument("csv", help="학습·검증에 쓰지 않은 영상 CSV")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    rows = load_rows(args.csv)
    # 30행씩 전송해 각 묶음의 평가와 연속 초과 상태를 확인한다.
    for start in range(0, len(rows), 30):
        response = requests.post(f"{args.url.rstrip('/')}/predict/batch-test",
                                 json={"videos": rows[start:start + 30]}, timeout=600)
        response.raise_for_status()
        print(json.dumps(response.json()["drift_check"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
