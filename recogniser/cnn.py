import os
from typing import Optional

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

INIT_LR = 1e-3
EPOCHS = 10
BS = 128


class SudokuNet:
    def __init__(
        self,
        width: Optional[int] = 28,
        height: Optional[int] = 28,
        depth: Optional[int] = 1,
        classes: Optional[int] = 10,
        name: Optional[str] = "digit_classifier",
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.depth: int = depth
        self.classes: int = classes
        self.model_name: str = name
        self.model_path: str = os.path.join("files\digit_model.h5")  # add the required path
        self.model = None

    def predict(self, digit):
        self._load_model()
        return self.model.predict(digit).argmax(axis=1)[0]

    def predict_(self, digit):
        self._load_model()

        raw_predictions = self.model.predict(digit)
        predicted_class_index = raw_predictions.argmax(axis=1)[0]
        confidence = raw_predictions[0, predicted_class_index]

        return predicted_class_index, confidence

    def train_model(self) -> None:
        self._load_data()
        self._build_model()
        self._compile_model()

        self.model.fit(
            self.train_data,
            self.train_labels,
            validation_data=(self.test_data, self.test_labels),
            batch_size=BS,
            epochs=EPOCHS,
            verbose=1,
        )

        self._evaluate_model()
        self._serialize_model()

    def _serialize_model(self):
        self.model.save(self.model_path)

    def _evaluate_model(self) -> None:
        predictions = self.model.predict(self.test_data)

    def _compile_model(self) -> None:
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=INIT_LR),
            metrics=["accuracy"],
        )

    def _load_model(self):
        if self.model is None:
            self.model = load_model(self.model_path)

    def _load_data(self) -> None:
        (
            (self.train_data, self.train_labels),
            (self.test_data, self.test_labels),
        ) = mnist.load_data()

        trainData = self.train_data.reshape((self.train_data.shape[0], 28, 28, 1))
        testData = self.test_data.reshape((self.test_data.shape[0], 28, 28, 1))

        trainData = trainData.astype("float32") / 255.0
        testData = testData.astype("float32") / 255.0

        self.le = LabelBinarizer()
        self.train_labels = self.le.fit_transform(self.train_labels)
        self.test_labels = self.le.transform(self.test_labels)

    def _build_model(self) -> None:
        model = Sequential()
        inputShape = (self.height, self.width, self.depth)

        model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(self.classes))
        model.add(Activation("softmax"))

        self.model = model
