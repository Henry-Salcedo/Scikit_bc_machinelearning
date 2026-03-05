"""
main.py
=======
Demonstrates the use of the three breast-cancer ML model classes defined in
ml_models.py.

Run
---
    python main.py
"""

from ml_models import DecisionTreeModel, LogisticRegressionModel, RandomForestModel


def print_results(name: str, results: dict) -> None:
    """Pretty-print evaluation results for a model."""
    separator = "-" * 60
    print(f"\n{separator}")
    print(f"Model: {name:<20}")
    print(separator)
    print(f"Accuracy : {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results["report"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])


def main() -> None:
    models = [
        ("Logistic Regression", LogisticRegressionModel()),
        ("Decision Tree", DecisionTreeModel(max_depth=5)),
        ("Random Forest", RandomForestModel(n_estimators=100)),
    ]

    for name, model in models:
        results = model.run()
        print_results(name, results)


if __name__ == "__main__":
    main()
