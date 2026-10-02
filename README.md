# ViewPilot

영상 한 행의 첫 3일 지표로 7일 누적 조회수를 예측하는 LSTM/FastAPI 프로젝트입니다.
학습·검증·기준값 계산·정상 평가·드리프트 평가를 각각 분리합니다.

데이터가 예측과 재학습으로 이어지는 과정을 쉽게 풀어쓴 [구현 상세 설명](docs/ViewPilot_구현상세.md)을 함께 읽어보세요.

```text
train → validation으로 모델 선택 → baseline_normal로 B 계산 및 수동 설정
                                       ↓
                          test_normal → test_drift
                                       ↓
                         경보 → test_drift 앞 60개로 재학습
                                       ↓
                         뒤 60개에서 RMSLE 개선 시 교체 / 아니면 유지
```

서버 실행 후 `http://localhost:8000/`에서 연동된 대시보드를 엽니다. API 문서는 `/docs`에서 확인합니다.

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
- `test_drift.csv`: 120개, 앞 60개는 감지·재학습, 뒤 60개는 검증

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

최초 학습에는 드리프트 기준값 B가 필요 없습니다. 검증으로 선택한 LSTM의 RMSLE가 같은 검증 데이터의
**Naive baseline RMSLE보다 엄격히 낮을 때만** 활성화합니다. `DEPLOY_RMSLE_GATE`를 설정하면
그 절대 상한 이하인지도 함께 검사합니다. 미설정이어도 Naive 비교는 생략하지 않습니다.
이미 활성 모델이 있으면 최초 학습으로 덮어쓰지 않고 오류를 반환합니다.


### Naive baseline 배포 게이트

Naive baseline은 **첫 3일의 일평균 조회수가 7일까지 유지된다**고 가정합니다.
예측식은 `(day1_views + day2_views + day3_views) / 3 * 7`이며 중간에 정수 반올림하지 않습니다.
정답이나 이후 날짜의 지표를 예측식에 사용하지 않고, 별도 학습·튜닝도 하지 않습니다.

- 최초 배포: `candidate RMSLE < naive RMSLE`, 설정한 경우 `candidate RMSLE <= DEPLOY_RMSLE_GATE`도 만족
- 재학습 교체: `candidate RMSLE < naive RMSLE` **그리고** `candidate RMSLE < current RMSLE`
- 실패: 후보의 실험 결과는 남기되 최초 활성화는 보류하고, 재학습에서는 기존 모델·가중치·포인터를 유지

비교는 항상 같은 검증 영상에서 수행합니다. API 결과와 모델 metadata에는 `naive_rmsle`,
`naive_method`, `deployment_checks`, `deployment_rule`을 남깁니다. MLflow에도 Naive RMSLE와
규칙·조건별 통과 여부를 기록합니다. 화면의 실행 응답 JSON에서 함께 확인할 수 있습니다.

Naive baseline은 배포 후보와 비교하는 **단순 예측 방식**입니다. `baseline_normal.csv`로 계산하는
**드리프트 기준값 B**와 다르며, 30개 묶음·두 번 연속 초과 조건은 그대로 유지합니다.
단순 규칙과 기존 모델을 이겼다는 상대 평가이므로 실제 서비스의 절대 품질을 보장하지는 않습니다.

2026-10-02 별도 브랜치와 임시 runtime에서 제공된 합성 데이터를 변경하지 않고 재평가했습니다.
최초 검증 RMSLE는 LSTM 0.1008 / Naive 0.2161, 정상 4묶음은 모두 정상,
드리프트 재학습 검증은 후보 0.1141 / 기존 0.5005 / Naive 0.5378로 게이트를 통과했습니다.
상세 조건과 수치는 [Naive 게이트 검증 결과](docs/naive_gate_evaluation.json)를 참고하세요.

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

## 서버 재시작으로 초기화하는 시연 모드

최초 모델 학습과 `calculate_baseline.py` 실행은 `DEMO_MODE=false`에서 한 번 완료합니다.
그다음 `.env`의 `DEMO_MODE=true`, `AUTO_RETRAIN=true`를 설정하고 서버를 실행합니다.

```bash
set -a
source .env
set +a
python -m uvicorn serving_app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

시연 모드에서는 서버 시작마다 원본 `runtime/models/production.json`이 가리키는 모델과 그 버전의
저장된 기준값을 `runtime/demo/`로 복사합니다. 시연 중의 모델 교체·평가 DB·로그·업로드·MLflow 기록은
이 폴더에만 저장합니다. 서버를 끄고 다시 켜면 시연 폴더를 복원하고 평가 기록은 비워집니다.
원본 모델과 원본 관측 기록은 유지하며, 서버를 켤 때마다 최초 모델을 다시 학습하지 않습니다.
기준값과 버전은 저장된 원본 결과에서 자동으로 읽습니다. `.env` 파일을 덮어쓰지는 않습니다.

대시보드의 두 버튼은 **본문 없이** 아래 API를 각각 호출합니다. 터미널에서도 같은 요청을 보낼 수 있습니다.

```bash
# 정상 버튼: 정상 데이터 120개를 평가하고 결과 반환
curl -X POST http://localhost:8000/api/v1/simulations/normal/run

# 드리프트 버튼: 감지 → 앞 60개 재학습 → 뒤 60개 검증 → 모델 교체 판단
curl -X POST http://localhost:8000/api/v1/simulations/drift/run
```

각 버튼은 독립적으로 실행할 수 있습니다. 정상 버튼 후 드리프트 버튼을 실행해도 됩니다.
드리프트로 모델이 교체되면 새 기준값이 없으므로 서버를 재시작한 뒤 다른 시연을 진행하세요.
같은 시나리오를 다시 누르면 409와 재시작 안내를 반환합니다. 두 버튼을 동시에 실행해도 409입니다.
초기화는 서버 시작 때 일어나며 API 요청마다 원본으로 되돌리지는 않습니다.

응답의 `events`에 30개 묶음별 결과가 순서대로 들어갑니다. `active_model_version`으로 최종 모델을,
`baseline_required`로 새 기준값 필요 여부를 확인합니다. 요청은 시나리오가 끝난 후 응답하므로
화면은 실행 중 경과 시간을 표시하고 중복 요청을 막습니다. 응답 대기 제한은 10분입니다.
단계별 실시간 스트리밍은 제공하지 않으며, 응답 후 결과와 모델 상태·이력·로그를 갱신합니다.

시연 모드는 로컬 SQLite MLflow 복사본과 로컬 모델을 사용합니다. 외부 MLflow 설정 대신 시연 DB에 기록합니다.
시연 서버는 하나만, worker는 1개만 실행하고 `--reload`는 사용하지 마세요.
`VIEWPILOT_RUNTIME_DIR`를 지정했다면 그 폴더가 원본이며, 그 안의 `demo/`가 초기화 대상입니다.
일반 모드(`DEMO_MODE=false`)는 서버를 재시작해도 기록을 보존합니다.

## 시나리오 1 정상 운영과 시나리오 2 드리프트

서버가 실행된 상태에서 새 터미널을 열고 프로젝트 가상환경을 활성화한 뒤 실행합니다.

```bash
python scripts/run_scenarios.py --url http://localhost:8000
# 정상 또는 드리프트만 실행하려면 다음 중 하나 사용
python scripts/run_scenarios.py --scenario normal
python scripts/run_scenarios.py --scenario drift
```

스크립트는 `test_normal` 4묶음 다음 `test_drift` 앞 60개를 2묶음으로 전송합니다.
두 번째 드리프트 요청에는 뒤 60개를 `retraining_validation`으로 따로 전달합니다.
이 검증 영상은 드리프트 평가 묶음에 포함하지 않습니다.
결과를 `runtime/scenario_report.json`에 저장합니다.

- 정상: 묶음별 RMSLE가 임계값 이내이면 경보·재학습 없이 기존 모델 유지
- 드리프트 첫 초과: `warning`
- 두 묶음 연속 초과: `retrain_review`, 앞 60개로 실제 재학습하고 뒤 60개에서 모델 비교
- `AUTO_RETRAIN=false`이면 `manual_review`로 종료하고 재학습하지 않음
- 기준 이하 복귀: 연속 초과 횟수 초기화
- 새 모델 교체 성공: 새 기준값 설정이 필요하므로 시나리오 실행 중단

임계값은 `T = max(B × 1.3, B + 0.1)`입니다. 같은 모델 버전에서 실제값이 등록된 순서로
30개씩 평가하며 이미 사용한 영상은 다음 묶음에 재사용하지 않습니다.

이 명령은 정답이 있는 과거 CSV로 수행하는 배치 시뮬레이션입니다. 실제 72시간·168시간을 기다리는
시간 이벤트 재생은 하지 않으며, 운영의 예측·실제값 저장 공간과 분리해 집계합니다.
실제 운영은 `/api/v1/videos/predictions`에서 먼저 예측을 저장하고 `/api/v1/videos/actuals`에서 나중에 실제값을 등록합니다.

같은 runtime에서 같은 영상을 다시 보내도 중복 평가하지 않습니다. 시연 모드에서는 서버를 재시작하면 다시 실행할 수 있습니다. 아래 새 경로 방식은 일반 모드에서 사용합니다. 모델이 교체된 뒤에도 같은 120개로 성능을 재평가하지 않습니다. 전체 실험을 재현하려면
새 터미널에서 `export VIEWPILOT_RUNTIME_DIR="$PWD/runtime/replay-2"`로 새 저장 위치를 지정하고
학습부터 반복하세요. 학습·기준값 계산·서버 실행에 같은 저장 위치를 사용해야 합니다.

## 재학습 데이터와 교체 판정

기본 시나리오는 추가 CSV 없이 실행됩니다. `test_drift.csv`를 게시 시점 순으로 나눕니다.

1. 앞 60개를 기존 모델로 평가하고, 두 묶음 연속 초과하면 이 60개로 추가 학습합니다.
2. 뒤 60개는 가중치 학습에 넣지 않습니다. 검증 손실로 최적 가중치를 선택하고 두 모델을 비교합니다.
3. 같은 검증 60개에서 후보 RMSLE가 **Naive baseline과 기존 모델 모두보다 엄격히 낮으면** 교체합니다. 어느 하나와 동점이거나 더 나쁘면 기존 모델을 유지합니다.

기존 모델을 복제하고 전처리를 고정한 채 학습률 `1e-4`로 최대 10 epoch 추가 학습합니다.
드리프트 임계값은 재학습을 시작하는 신호이며 배포 상한으로 사용하지 않습니다.
`DEPLOY_RMSLE_GATE`는 최초 모델 선택에만 적용되고 재학습 교체에는 적용되지 않습니다.

**이 실험은 과거 데이터로 수행하는 재학습·모델 선택 시나리오입니다.**
제공 CSV에서 60번째 영상은 2025-09-17 09:00 UTC, 61번째 영상은 같은 날 12:00 UTC에 게시됐습니다.
첫 검증 영상의 3일차에는 마지막 학습 영상의 7일 정답이 아직 없으므로, 60/60 분할을
실제 당시의 예측 실험으로 해석할 수 없습니다. 모든 정답이 확보된 후 비교하며, 검증 세트는
가중치 선택에도 사용되므로 교체 후의 독립적인 성능 평가를 의미하지 않습니다.

교체하면 응답에 새 버전과 `baseline_required=true`가 표시됩니다. 시나리오는 여기서 끝나며
새 모델에 맞는 대표 영상 30개로 B를 수동 계산·설정한 뒤 운영 모니터링을 재개합니다.
교체된 모델에 이전 B를 자동 적용하지 않습니다. 새 모델의 기준값이나 교체 후 성능을
학습·검증에 사용한 120개로 다시 측정하지 않습니다.

일반 운영용 재학습은 기존처럼 `RETRAIN_TRAIN_CSV`, `RETRAIN_VALIDATION_CSV` 또는
`retrain_train`, `retrain_validation` 용도 업로드를 사용할 수 있습니다. 두 파일 모두 14개 컬럼입니다.
실제값이 확보된 관측 영상은 저장된 입력·정답과 일치하면 재학습에 사용할 수 있습니다.
검증에는 아직 관측·학습·기준값 계산에 사용하지 않은 영상을 지정합니다.
일반 재학습은 첫 검증 예측 시점에 학습 정답을 알 수 있도록 게시 시점 간격도 검사합니다.
학습 최소 30개·검증 최소 10개이며, 이 수량이 품질을 보장하는 것은 아닙니다.

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

## 대시보드 사용과 파일 구조

HTML 파일을 직접 열지 않고 서버의 `http://localhost:8000/`에 접속합니다.
최초 모델과 기준값을 준비한 시연 모드에서 **정상 시나리오 실행 → 드리프트 시나리오 실행**을 누르면 됩니다.

- 상단 카드: 현재 모델 버전, 기준 RMSLE, 임계값, 기준값 필요 여부
- 시나리오 결과: 묶음별 판정, 기존·후보의 검증 RMSLE, 모델 교체 결과
- 저장된 평가와 예측: 모드·모델 버전 필터, 페이지 이동, RMSLE 그래프, 예측·실제값 목록
- 새 영상 CSV 예측: 예측 저장 후 운영 예측 목록으로 이동. 미등록 행의 **실제값 입력**으로 정답 등록
- 학습 데이터 업로드·수동 학습: 용도별 업로드, 최신 파일 확인, 최초/추가 학습
- 운영 로그: 서버 로그 조회와 새로고침

브라우저 새로고침 후 예측·평가 이력은 서버에서 다시 읽습니다. 이번 실행의 상세 재학습 비교 응답은
페이지 메모리에만 표시하므로 새로고침하면 사라집니다. 시연 모드의 서버 재시작은 시연 이력 자체를 초기화합니다.
기준값 계산·설정은 기존 CLI 방식으로 유지합니다.

디자인 변경과 API 수정을 분리할 수 있도록 파일을 나눴습니다.

- `serving_app/static/index.html`: 화면 구조와 입력 폼
- `serving_app/static/dashboard.css`: 색상·배치·반응형 스타일
- `serving_app/static/api.js`: API 경로, JSON/파일 전송, 오류·타임아웃 처리
- `serving_app/static/dashboard.js`: 버튼 이벤트, 실행 잠금, 응답 렌더링, 이력·그래프 갱신

외부 프레임워크나 차트 CDN 없이 동작합니다. 디자인을 바꿀 때는 요소 ID와 이벤트 연결도 함께 확인하세요.

## API

서비스 API는 `/api/v1` 아래에 모았습니다. `/health`, `/docs`, `/openapi.json`은 기존 경로를 유지합니다.
요청·응답 형식과 `purpose` 값은 그대로이며, 이전 API 경로는 제공하지 않습니다.
대시보드는 아래 API를 같은 서버의 상대 경로로 호출합니다. API 호출은 `serving_app/static/api.js`에 모았습니다.

- `POST /api/v1/datasets?purpose=train`: 학습용 CSV
- `POST /api/v1/datasets?purpose=validation`: 검증용 CSV
- `POST /api/v1/datasets?purpose=retrain_train`: 별도 재학습용 CSV
- `POST /api/v1/datasets?purpose=retrain_validation`: 별도 재학습 검증 CSV
- `GET /api/v1/datasets/latest?purpose=train`: 해당 용도의 최신 파일 상태
- `POST /api/v1/models/training`: `{"mode":"initial","epochs":100,"patience":10}` 또는 `fine_tune`
- `GET /api/v1/videos/predictions`: 영상별 예측·실제값 이력 조회
- `GET /api/v1/monitoring/drift`: 완료된 30개 묶음의 드리프트 평가 이력 조회
- `POST /api/v1/videos/predictions`: 정답을 제외한 영상 한 행을 JSON으로 전달
- `POST /api/v1/videos/predictions/csv`: 예측용 CSV
- `POST /api/v1/videos/actuals`: `{"prediction_id":"...","target_views_day7":12000}`
- `POST /api/v1/simulations/normal/run`: 정상 버튼용, 본문 없음, 시연 모드 전용
- `POST /api/v1/simulations/drift/run`: 드리프트 버튼용, 본문 없음, 시연 모드 전용
- `POST /api/v1/simulations/drift`: `{"videos":[정답 포함 영상 행, ...]}`; 선택 필드 `retraining_validation`에 별도 검증 행 전달
- `POST /api/v1/simulations/drift/csv`: 정답 포함 시뮬레이션 CSV
- `GET /api/v1/logs`: 운영 로그 파일 목록
- `GET /health`, `GET /api/v1/logs/aiops.log`: 모델·설정 상태와 로그

CSV 업로드는 multipart `file` 필드, UTF-8, 최대 10MB입니다. JSON 배치 최대 영상 수는 1,000개입니다.
운영 예측은 게시 후 72시간, 실제값 등록은 168시간이 지나야 합니다. 처음 예측한 값과 모델 버전을 저장하고
같은 영상·버전·모드의 중복 요청은 최초 결과를 유지합니다. 서버 처리 시간은 콘솔에 `duration=...ms`로 출력합니다.

### 대시보드 이력 조회

새로고침 후에는 아래 조회 API로 저장된 목록과 그래프를 다시 구성할 수 있습니다.

```bash
curl 'http://localhost:8000/api/v1/videos/predictions?mode=simulation&page=1&page_size=20'
curl 'http://localhost:8000/api/v1/monitoring/drift?mode=simulation&model_version=1&page=1&page_size=20'
```

두 API의 공통 파라미터는 `page`(기본 1, 1 이상), `page_size`(기본 20, 1~100),
`mode`(`live` 또는 `simulation`), `model_version`(문자열)입니다.
필터를 생략하면 모든 모드·버전을 반환하며, 두 필터를 함께 지정하면 둘 다 일치하는 기록만 반환합니다.
현재 활성 모델 버전으로 자동 제한하지 않으므로 모델 교체 후에도 v1 이력을 조회할 수 있습니다.

응답은 모두 `items`, `total`, `page`, `page_size`를 갖습니다. `total`은 필터 적용 후 전체 개수입니다.
이력이 없거나 페이지 범위를 넘으면 `items=[]`이며, 잘못된 파라미터에는 422를 반환합니다.

예측 이력의 각 항목에는 다음 필드가 있습니다.

- `prediction_id`, `video_id`, `model_version`, `mode`
- `predicted_views_day7`: 당시 저장한 예상 조회수
- `target_views_day7`: 등록된 실제 조회수, 아직 없으면 `null`
- `predicted_at`, `observed_at`: 예측·실제값 등록 시각, 미등록 실제값 시각은 `null`
- `block_id`: 포함된 평가 묶음 ID, 아직 평가하지 않았으면 `null`

최신 예측부터 정렬하고 동일 시각이면 나중에 저장한 기록부터 반환합니다.
동일 영상도 모델 버전이나 모드가 다르면 별도 예측 기록으로 표시됩니다.

드리프트 이력의 각 항목에는 다음 필드가 있습니다.

- `block_id`, `model_version`, `mode`, `video_ids`
- `baseline_rmsle`, `rmsle`, `threshold`
- `status`: `ok`, `warning`, `retrain_review`
- `consecutive_exceeds`: 그 묶음까지의 연속 초과 횟수

최신 `block_id`부터 반환합니다. 그래프는 필요한 페이지를 모은 뒤 `block_id` 오름차순으로 정렬하세요.
완료된 평가만 반환하며 대기 중 영상과 재학습 결과는 포함하지 않습니다. 과거 기준값·판정을
현재 모델 설정으로 다시 계산하지 않습니다. 재학습 전후 수치와 교체 여부는 기존 실행 API의 응답을 사용하세요.

조회는 저장된 SQLite를 읽기만 하며 모델 추론·드리프트 평가·재학습을 실행하지 않습니다.
새 DB를 만들지도 않습니다. 시연 모드에서는 화면 새로고침 후 이력이 유지되지만,
서버 재시작으로 시연 폴더를 초기화하면 조회 결과도 빈 목록으로 돌아갑니다.

## 저장 구조와 테스트

- `data/features.py`: 입력 검증·분리 검사·전처리
- `data/baselines.py`: 수동 기준값과 기준 영상 보관
- `data/observations.py`: SQLite 예측·실제값·평가 묶음
- `serving_app/train_and_register.py`: 학습·조기 종료·모델 선택
- `serving_app/deployment.py`: Naive·기존 모델 비교와 배포 게이트
- `serving_app/model_loader.py`: 모델·전처리 버전 로딩과 캐시 교체
- `serving_app/demo.py`: 시작 시 원본 모델·기준값을 시연 폴더로 복원
- `serving_app/routers/simulations.py`: 두 버튼용 시나리오 API
- `serving_app/monitoring/`: 드리프트 판정과 재학습·검증 연결

각 MLflow run별로 `model.keras`, `preprocessor.json`, `metadata.json`을 저장합니다.
검증 후 `production.json`을 원자적으로 갱신하며, 다음 요청에서 새 모델을 완전히 로드한 뒤 캐시를 교체합니다.

```bash
python -m unittest discover -s tests -v
python -m tests.smoke_training
```

테스트는 임시 runtime을 사용합니다. smoke test는 Naive 미달 최초 후보와 개선됐지만 Naive에 못 미치는 재학습 후보의 차단,
실제 TensorFlow 최초 학습과 MLflow 로딩, 동점 후보의 교체 거절을 확인하고,
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
