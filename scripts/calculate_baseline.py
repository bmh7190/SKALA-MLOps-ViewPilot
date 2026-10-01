"""선택한 모델을 고정하고 30개 기준 영상의 RMSLE를 한 번 계산한다."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.baselines import calculate_baseline, save_baseline
from data.features import load_rows
from serving_app.model_loader import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/synthetic/baseline_normal.csv")
    args = parser.parse_args()
    result = calculate_baseline(get_model(), load_rows(args.csv))
    save_baseline(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n다음 두 줄을 .env에 직접 입력한 뒤 서버를 다시 실행하세요:")
    print(f"BASELINE_RMSLE={result['baseline_rmsle']!r}")
    print(f"BASELINE_MODEL_VERSION={result['model_version']}")


if __name__ == "__main__":
    main()
