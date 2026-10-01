# ViewPilot

영상 한 행의 첫 3일 지표로 7일 누적 조회수를 예측하는 LSTM/FastAPI 프로젝트입니다.
학습·검증·기준값 계산·정상 평가·드리프트 평가를 각각 분리합니다.

```text
train → validation으로 모델 선택 → baseline_normal로 B 계산 및 수동 설정
                                       ↓
                          test_normal → test_drift
                                       ↓
                         경보 → 별도 데이터로 재학습
                                       ↓
                         검증 통과 시 교체 / 실패 시 유지
```

대시보드는 변경하지 않았습니다. API 확인은 `/docs`를 사용하세요.

## 준비

Python 3.11을 사용합니다. 프로젝트 루트에서 실행하세요.

```bash
git clone https://github.com/bmh7190/SKALA-MLOps-ViewPilot.git viewpilot
cd viewpilot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

합성 데이터가 `data/synthetic/`에 포함되어 있습니다.

- `train.csv`: 3,000개, 모델 가중치와 전처리 학습
- `validation.csv`: 600개, 조기 종료와 최적 가중치 선택
- `baseline_normal.csv`: 30개, 선택된 모델의 기준 RMSLE 계산
- `test_normal.csv`: 120개, 정상 운영 30개 묶음 4개
- `test_drift.csv`: 120개, 드리프트 운영 30개 묶음 4개

원본 데이터 안내와 생성 보고서도 같은 폴더에 포함했습니다. 실제 YouTube 수집 데이터가 아닌
합성 데이터입니다. 해당 폴더의 `smoke_test_report.json`은 데이터 제작자의 **선형회귀** 점검 결과이며
이 프로젝트의 LSTM 결과와 구분해야 합니다. LSTM 실행 결과는 `docs/scenario_results.json`에 있습니다.

## 최초 학습과 모델 선택

```bash
python -m serving_app.train_and_register \
  --train data/synthetic/train.csv \
  --validation data/synthetic/validation.csv \
  --epochs 100 --patience 10
```

`train`으로만 스케일러·카테고리 사전·가중치를 학습합니다. `validation`은 가중치 갱신에 사용하지 않습니다.
검증 로그 조회수 MSE인 `val_loss`가 10 epoch 동안 개선되지 않으면 멈추고, 가장 좋았던 가중치를 복원합니다.
최대 epoch에 도달한 경우에도 가장 좋았던 가중치를 복원합니다. 최종 RMSLE는 조회수로 복원·보정한 예측으로 계산합니다.

최초 학습에는 B가 필요 없습니다. `DEPLOY_RMSLE_GATE`를 설정하지 않았다면 검증으로 선택한 모델을
활성화합니다. 이는 별도의 서비스 품질 상한을 통과했다는 의미는 아닙니다. 상한을 지정하면 함께 적용합니다.
이미 활성 모델이 있으면 최초 학습으로 덮어쓰지 않고 오류를 반환합니다.

## 베이스라인 수동 계산과 설정

```bash
python scripts/calculate_baseline.py --csv data/synthetic/baseline_normal.csv
```

스크립트가 출력하는 다음 두 값을 `.env`에 직접 입력합니다. 아래 빈 칸에 계산 결과를 사용하세요.

```dotenv
BASELINE_RMSLE=
BASELINE_MODEL_VERSION=
```

30개 영상 ID와 계산 결과는 `runtime/baselines/<version>.json`에 보관합니다.
이 영상들은 학습·운영 평가에 재사용하지 못하도록 검사합니다. 스크립트는 `.env`를 자동 변경하거나
모델을 학습하지 않습니다. 저장된 결과와 설정값이 다른 경우 평가를 거절합니다.

모델을 교체하면 기존 B를 새 버전에 적용하지 않습니다. 새 모델의 기준값을 다시 계산·설정해야 합니다.

## 서버 실행

```bash
set -a
source .env
set +a
python -m uvicorn serving_app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

API 문서: http://localhost:8000/docs

설정 변경 후에는 환경변수를 다시 읽고 서버를 다시 실행하세요. 서버의 예측 기록, 평가 묶음,
모델과 MLflow 데이터는 `runtime/`에 보존합니다. 가상환경·실행 산출물·`.env`는 Git에 포함하지 않습니다.
한 프로세스의 실습 범위이며, CLI 학습과 API 학습을 동시에 실행하지 마세요.

## 시나리오 1 정상 운영과 시나리오 2 드리프트

서버가 실행된 상태에서 새 터미널을 열고 프로젝트 가상환경을 활성화한 뒤 실행합니다.

```bash
python scripts/run_scenarios.py --url http://localhost:8000
```

스크립트는 `test_normal` 4묶음 다음 `test_drift` 4묶음을 전송합니다.
결과를 `runtime/scenario_report.json`에 저장합니다.

- 정상: 묶음별 RMSLE가 임계값 이내이면 경보·재학습 없이 기존 모델 유지
- 드리프트 첫 초과: `warning`
- 두 묶음 연속 초과: `retrain_review`, 자동 재학습이 켜져 있으면 후보 학습 시도
- 기준 이하 복귀: 연속 초과 횟수 초기화
- 새 모델 교체 성공: 새 기준값 설정이 필요하므로 시나리오 실행 중단

임계값은 `T = max(B × 1.3, B + 0.1)`입니다. 같은 모델 버전에서 실제값이 등록된 순서로
30개씩 평가하며 이미 사용한 영상은 다음 묶음에 재사용하지 않습니다.

이 명령은 정답이 있는 과거 CSV로 수행하는 배치 시뮬레이션입니다. 실제 72시간·168시간을 기다리는
시간 이벤트 재생은 하지 않으며, 운영의 예측·실제값 저장 공간과 분리해 집계합니다.
실제 운영은 `/predict`에서 먼저 예측을 저장하고 `/observations`에서 나중에 실제값을 등록합니다.

같은 runtime에서 같은 영상을 다시 보내도 중복 평가하지 않습니다. 전체 실험을 재현하려면
새 터미널에서 `export VIEWPILOT_RUNTIME_DIR="$PWD/runtime/replay-2"`로 새 저장 위치를 지정하고
학습부터 반복하세요. 학습·기준값 계산·서버 실행에 같은 저장 위치를 사용해야 합니다.

## 재학습 데이터와 교체 판정

**제공된 5개 파일에는 별도의 변화 후 재학습·검증 데이터가 없습니다.**
따라서 기본 시나리오에서는 드리프트를 감지한 뒤 `retraining.status=deferred`와 준비할 데이터 안내가 나옵니다.
테스트 영상을 학습에 자동 편입하거나, 같은 영상으로 다시 평가해 개선됐다고 판정하지 않습니다.

새 데이터를 준비하면 `.env`에 다음 경로를 지정합니다. 두 파일 모두 기존 14개 컬럼을 사용합니다.

```dotenv
AUTO_RETRAIN=true
RETRAIN_TRAIN_CSV=data/adaptation/train.csv
RETRAIN_VALIDATION_CSV=data/adaptation/validation.csv
```

또는 API로 `retrain_train`, `retrain_validation` 용도의 CSV를 각각 업로드할 수 있습니다.
환경변수에 경로를 지정했다면 그 파일이 업로드보다 우선합니다. Git에 포함하려는 추가 합성 CSV는
`data/` 아래에 두고, API 업로드 결과는 `runtime/`에만 보관합니다.

재학습 파일은 기존 학습·검증 범위보다 이후 영상으로 구성하고, 서로 중복되지 않게 준비해야 합니다.
베이스라인이나 이미 운영·시뮬레이션에서 평가한 영상도 재사용할 수 없습니다.
학습 30개·검증 10개 이상이며, 최초 검증 예측 시점에 학습 정답이 확보될 수 있도록 시간 간격을 검사합니다.
이 최소 행 수는 실행 조건이며 모델 품질을 보장하지 않습니다.

새 묶음의 경보가 발생하면 기존 모델을 복제하고, 전처리를 고정한 채 학습률 `1e-4`로 최대 10 epoch 추가 학습합니다.
후보와 기존 모델을 같은 별도 검증 영상으로 비교해 두 조건을 모두 충족해야 교체합니다.

```text
후보 RMSLE < 기존 모델 RMSLE
후보 RMSLE ≤ 배포 상한
```

기본 배포 상한은 현재 모델 B의 `max(B*1.3, B+0.1)`입니다. `DEPLOY_RMSLE_GATE`로 따로 지정할 수 있습니다.
실패하면 기존 모델·가중치·활성 버전을 유지합니다. 데이터 보충 후 즉시 재시도하려면
`POST /training`에 `mode: fine_tune`을 보내거나 다음 새 평가 묶음을 사용합니다.

## CSV와 모델 입력

컬럼 예시는 `data/sample_videos.csv`에 있습니다. 이 3행 파일은 형식 확인용입니다.

- 한 행은 독립된 영상 하나입니다. 다른 영상을 연결해 시퀀스를 만들지 않습니다.
- 1·2·3일차는 게시 후 0~24, 24~48, 48~72시간이며 각 구간의 조회수·노출 수를 사용합니다.
- 평균 시청 비율은 45.5처럼 퍼센트로 전달합니다. 일반 검증에서는 100 초과도 허용합니다.
- `video_id`, `published_at`은 식별·시간 검증용이며 모델 입력에서 제외합니다.
- `target_views_day7`은 0~168시간 누적 조회수이고, 첫 3일 합보다 작을 수 없습니다.
- 예측 CSV는 정답 컬럼이 없어도 됩니다. 정답이 있더라도 피처에서는 제외합니다.

수량은 `log1p`, 시청 비율은 `/100` 후 학습 세트 기준으로 표준화합니다.
카테고리는 train에서 사전을 만들며 제공 데이터의 순서는 gaming, education, entertainment, lifestyle입니다.
미등록 카테고리는 all-zero로 처리합니다. 구독자 수와 카테고리를 세 시점에 반복합니다.

```text
입력 (N, 3, 8)
→ LSTM 32 → LSTM 32 → LSTM 16 → Dense 16 → Dense 1
```

정답 `log1p(7일 조회수)`를 MSE로 학습합니다. `expm1`으로 복원하고 정수로 반올림한 뒤
첫 3일 조회수 합 이상으로 보정합니다. 학습 평가·API·베이스라인·드리프트에서 같은 후처리를 씁니다.

## API

- `POST /data/upload?purpose=train`: 학습용 CSV
- `POST /data/upload?purpose=validation`: 검증용 CSV
- `POST /data/upload?purpose=retrain_train`: 별도 재학습용 CSV
- `POST /data/upload?purpose=retrain_validation`: 별도 재학습 검증 CSV
- `GET /data/status?purpose=train`: 해당 용도의 최신 파일 상태
- `POST /training`: `{"mode":"initial","epochs":100,"patience":10}` 또는 `fine_tune`
- `POST /predict`: 정답을 제외한 영상 한 행을 JSON으로 전달
- `POST /predict/csv`: 예측용 CSV
- `POST /observations`: `{"prediction_id":"...","target_views_day7":12000}`
- `POST /predict/batch-test`: `{"videos":[정답 포함 영상 행, ...]}`
- `POST /predict/batch-test/csv`: 정답 포함 시뮬레이션 CSV
- `GET /health`, `GET /logs/aiops.log`: 모델·설정 상태와 로그

CSV 업로드는 multipart `file` 필드, UTF-8, 최대 10MB입니다. JSON 배치 최대 영상 수는 1,000개입니다.
운영 예측은 게시 후 72시간, 실제값 등록은 168시간이 지나야 합니다. 처음 예측한 값과 모델 버전을 저장하고
같은 영상·버전·모드의 중복 요청은 최초 결과를 유지합니다. 서버 처리 시간은 콘솔에 `duration=...ms`로 출력합니다.

## 저장 구조와 테스트

- `data/features.py`: 입력 검증·분리 검사·전처리
- `data/baselines.py`: 수동 기준값과 기준 영상 보관
- `data/observations.py`: SQLite 예측·실제값·평가 묶음
- `serving_app/train_and_register.py`: 학습·조기 종료·모델 선택
- `serving_app/model_loader.py`: 모델·전처리 버전 로딩과 캐시 교체
- `serving_app/monitoring/`: 드리프트 판정과 별도 데이터 재학습 연결

각 MLflow run별로 `model.keras`, `preprocessor.json`, `metadata.json`을 저장합니다.
검증 후 `production.json`을 원자적으로 갱신하며, 다음 요청에서 새 모델을 완전히 로드한 뒤 캐시를 교체합니다.

```bash
python -m unittest discover -s tests -v
python -m tests.smoke_training
```

테스트는 임시 runtime을 사용합니다. smoke test는 실제 TensorFlow 최초·추가 학습과 MLflow 로딩을 확인하고,
교체 성공 분기에는 재현 가능한 고정 출력 후보를 사용합니다. 실제 데이터에서 재학습이 개선됐다는 증거는 아닙니다.

Docker 구성 확인 및 실행:

```bash
docker compose --env-file .env -f serving_app/docker-compose.yml up --build
```

이미지 빌드에서 학습하지 않습니다. 컨테이너 안에서 같은 학습·기준값 계산 명령을 실행하거나 API로 진행하세요.
호스트에서 만든 MLflow DB에는 절대 아티팩트 경로가 남으므로, 호스트 학습 runtime을 컨테이너로 옮겨 재사용하지 마세요.

## 원본 안내

기존 HAIC 실습의 FastAPI, LSTM과 MLflow 구성을 기반으로 변경했습니다.
원본 안내: 다음 실습 코드는 학습 목적으로만 사용 바랍니다.
문의: architect@sk.com, audit@korea.ac.kr 임성열 Ph.D.
