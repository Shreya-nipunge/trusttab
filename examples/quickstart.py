"""Small offline example of the TrustTab public API."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

from trusttab.api import (
    build_model,
    evaluate_classifier,
    inject_symmetric_label_noise,
)


def main() -> None:
    features, labels = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=12,
    )
    X = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(8)])
    y = pd.Series(labels, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=21,
        stratify=y,
    )

    noisy_y_train = inject_symmetric_label_noise(y_train, noise_rate=0.1, seed=33)
    model = build_model(
        "logistic_regression",
        features=X_train,
        random_state=44,
        model_params={"max_iter": 1000},
    )
    model.fit(X_train, noisy_y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    positive_index = list(model.classes_).index(1)
    metrics = evaluate_classifier(
        y_test,
        predictions,
        probabilities[:, positive_index],
    )
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
