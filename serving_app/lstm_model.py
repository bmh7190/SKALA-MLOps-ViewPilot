"""기존 LSTM 층 구성은 유지하고 영상의 3일 입력을 받는다."""
from tensorflow import keras

from data.features import SEQ_LEN


def build_model(n_features: int) -> keras.Model:
    # 카테고리 사전 크기가 정해진 뒤 실제 입력 너비를 전달한다.
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, n_features)),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
