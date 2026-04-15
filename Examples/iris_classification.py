from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_models() -> dict[str, object]:
    return {
        "Karar Agaci": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Destek Vektor Makinasi": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale"),
        ),
        "Rastgele Orman": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }


def print_table(rows: list[dict[str, float]]) -> None:
    header = (
        f"{'Model':<26}"
        f"{'Accuracy':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Kappa':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['name']:<26}"
            f"{row['accuracy']:>10.3f}"
            f"{row['precision']:>12.3f}"
            f"{row['recall']:>10.3f}"
            f"{row['f1']:>10.3f}"
            f"{row['kappa']:>10.3f}"
        )


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "kappa": make_scorer(cohen_kappa_score),
    }

    rows: list[dict[str, float]] = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        rows.append(
            {
                "name": name,
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "kappa": scores["test_kappa"].mean(),
            }
        )

    rows.sort(key=lambda row: row["accuracy"], reverse=True)
    print("10 katli capraz dogrulama ortalamalari")
    print_table(rows)

    best_model_name = rows[0]["name"]
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\nSecilen model:", best_model_name)
    print("Test kumesi Cohen kappa:", round(cohen_kappa_score(y_test, y_pred), 3))
    print("\nSiniflandirma raporu")
    print(classification_report(y_test, y_pred, target_names=iris.target_names, digits=3))
    print("Karisiklik matrisi")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
