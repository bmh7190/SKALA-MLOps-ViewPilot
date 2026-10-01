"""업로드된 CSV로 시작하는 학습. 최초 학습과 재학습은 같은 함수를 사용한다."""
from fastapi import APIRouter

from serving_app.monitoring.retrain_trigger import retraining_data
from serving_app.schemas import TrainingRequest
from serving_app.train_and_register import train_and_register

router = APIRouter()


@router.post("/training")
def train(request: TrainingRequest):
    options = {"mode": request.mode, "epochs": request.epochs, "patience": request.patience}
    if request.mode == "fine_tune":
        training, validation = retraining_data()
        options.update(training_rows=training, validation_rows=validation)
    return train_and_register(**options)
