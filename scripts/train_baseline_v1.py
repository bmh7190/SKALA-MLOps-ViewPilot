"""기존 실습 명령을 유지하는 최초 학습 진입점. 별도 학습 로직을 중복 작성하지 않는다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving_app.train_and_register import main

if __name__ == "__main__":
    main()
