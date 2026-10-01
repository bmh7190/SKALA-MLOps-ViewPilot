"""정상 120개 → 드리프트 앞 60개 감지·재학습 → 뒤 60개 검증을 실행한다."""
import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import load_rows, validate_splits


def split_drift_rows(rows):
    """영상 ID를 섞지 않고 게시 순서로 60개씩 분리한다."""
    ordered = sorted(rows, key=lambda row: row["published_at"])
    if len(ordered) != 120:
        raise ValueError("드리프트 시나리오는 영상 120개가 필요합니다.")
    return validate_splits(ordered[:60], ordered[60:], retrospective=True)


def run_scenarios(url: str, data_dir: Path, scenario: str = "both") -> dict:
    if scenario not in {"normal", "drift", "both"}:
        raise ValueError("scenario는 normal, drift, both 중 하나여야 합니다.")
    training, validation = [], []
    if scenario != "normal":
        training, validation = split_drift_rows(load_rows(data_dir / "test_drift.csv"))
    response = requests.get(f"{url}/health", timeout=30)
    response.raise_for_status()
    health = response.json()
    if not health["model_ready"] or health["baseline_rmsle"] is None:
        raise ValueError("모델을 선택하고 해당 버전의 베이스라인을 설정한 뒤 서버를 실행하세요.")
    version = health["model_version"]
    report = {"scenario": scenario, "model_version": version, "baseline_rmsle": health["baseline_rmsle"],
              "threshold": health["drift_threshold"], "evaluation_mode": "retrospective",
              "retraining_video_ids": [row["video_id"] for row in training],
              "validation_video_ids": [row["video_id"] for row in validation], "events": []}
    names = ["test_normal", "test_drift"] if scenario == "both" else [f"test_{scenario}"]
    for name in names:
        rows = (training if name == "test_drift" else
                sorted(load_rows(data_dir / "test_normal.csv"), key=lambda row: row["published_at"]))
        if len(rows) % 30:
            raise ValueError("시나리오 CSV는 30개 단위의 완전한 묶음으로 준비하세요.")
        for start in range(0, len(rows), 30):
            payload = {"videos": rows[start:start + 30]}
            if name == "test_drift" and start == 30:
                # 검증 60개는 드리프트 묶음에 전송하지 않고 재학습 비교용으로만 전달한다.
                payload["retraining_validation"] = validation
            response = requests.post(f"{url}/api/v1/simulations/drift", json=payload, timeout=600)
            response.raise_for_status()
            check = response.json()["drift_check"]
            if check["model_version"] != version:
                raise ValueError("시나리오 실행 중 다른 모델 버전이 사용되었습니다.")
            if len(check["new_blocks"]) != 1:
                raise ValueError("이미 평가한 영상이거나 이전 미완료 묶음이 있습니다. 시연 모드에서는 서버를 재시작하고, 일반 모드에서는 새 runtime을 사용하세요.")
            event = {"file": name, "window": start // 30 + 1, **check}
            report["events"].append(event)
            print(json.dumps(event, ensure_ascii=False))
            # 교체가 성공하면 새 모델의 B를 다시 설정해야 하므로 여기서 중단한다.
            if check.get("retraining", {}).get("promoted"):
                report["stopped_after_promotion"] = True
                report["baseline_required"] = True
                return report
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["normal", "drift", "both"], default="both")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("runtime/scenario_report.json"))
    args = parser.parse_args()
    report = run_scenarios(args.url.rstrip("/"), args.data_dir, args.scenario)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
