"""
ml_models.py
============
Class design and implementation for three machine learning models applied to
the scikit-learn breast cancer dataset.

Each class documents its attributes, methods, and known limitations.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Abstract base class shared by all breast-cancer ML models.

    Attributes
    ----------
    model : sklearn estimator or None
        The underlying scikit-learn estimator.  Set to ``None`` until
        :meth:`build` is called.
    X_train : ndarray of shape (n_train, n_features)
        Training feature matrix.  Available after :meth:`load_data`.
    X_test : ndarray of shape (n_test, n_features)
        Test feature matrix.  Available after :meth:`load_data`.
    y_train : ndarray of shape (n_train,)
        Training labels.  Available after :meth:`load_data`.
    y_test : ndarray of shape (n_test,)
        Test labels.  Available after :meth:`load_data`.
    scaler : StandardScaler
        Fitted scaler used to standardise the features.
    test_size : float
        Proportion of the dataset to reserve for testing (default 0.20).
    random_state : int
        Seed for all random operations to ensure reproducibility (default 42).

    Limitations
    -----------
    * Designed exclusively for the sklearn breast cancer dataset; the
      ``load_data`` method will not accept arbitrary datasets.
    * No hyperparameter-tuning pipeline is included.  Models use either
      default or manually specified hyperparameters.
    * Feature scaling is performed with ``StandardScaler``; non-Gaussian
      features may benefit from a different scaler.
    """

    def __init__(self, test_size: float = 0.20, random_state: int = 42) -> None:
        self.model = None
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load the breast cancer dataset and create train/test splits.

        The raw features are standardised via :class:`~sklearn.preprocessing.StandardScaler`
        fitted *only* on the training split to prevent data leakage.

        Returns
        -------
        None

        Side effects
        ------------
        Sets ``X_train``, ``X_test``, ``y_train``, and ``y_test`` on the instance.
        """
        data = load_breast_cancer()
        X, y = data.data, data.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self) -> None:
        """Instantiate and configure the underlying scikit-learn estimator.

        Subclasses must override this method to set ``self.model``.
        """

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Fit the model on the training data.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If :meth:`build` or :meth:`load_data` have not been called first.
        """
        if self.model is None:
            raise RuntimeError("Call build() before train().")
        if self.X_train is None:
            raise RuntimeError("Call load_data() before train().")
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix to predict.  Must already be scaled.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If :meth:`train` has not been called first.
        """
        if self.model is None:
            raise RuntimeError("Call train() before predict().")
        return self.model.predict(X)

    def evaluate(self) -> dict:
        """Evaluate the trained model on the held-out test set.

        Returns
        -------
        results : dict with keys
            * ``accuracy``  – float, overall accuracy on the test split.
            * ``report``    – str, full classification report.
            * ``confusion_matrix`` – ndarray, confusion matrix.

        Raises
        ------
        RuntimeError
            If :meth:`train` has not been called first.
        """
        y_pred = self.predict(self.X_test)
        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "report": classification_report(self.y_test, y_pred,
                                             target_names=["malignant", "benign"]),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
        }

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full pipeline: load data → build → train → evaluate.

        Returns
        -------
        results : dict
            The evaluation results returned by :meth:`evaluate`.
        """
        self.load_data()
        self.build()
        self.train()
        return self.evaluate()


# ---------------------------------------------------------------------------
# Logistic Regression model
# ---------------------------------------------------------------------------

class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier for the breast cancer dataset.

    Attributes
    ----------
    max_iter : int
        Maximum number of solver iterations (default 10 000).
    C : float
        Inverse of regularisation strength; smaller values specify stronger
        regularisation (default 1.0).
    solver : str
        Algorithm to use in the optimisation problem (default ``'lbfgs'``).
    model : LogisticRegression or None
        The underlying :class:`~sklearn.linear_model.LogisticRegression`
        estimator.  ``None`` until :meth:`build` is called.

    Limitations
    -----------
    * Assumes a (roughly) linear decision boundary between classes.
    * Performance degrades when features are highly correlated
      (multicollinearity).
    * Requires feature scaling for well-conditioned optimisation; the
      parent class applies ``StandardScaler`` automatically.
    * Does not natively handle missing values.
    * Binary-only by default; multi-class extension uses a one-vs-rest or
      multinomial strategy.

    Examples
    --------
    >>> lr = LogisticRegressionModel()
    >>> results = lr.run()
    >>> print(results["accuracy"])
    """

    def __init__(self, max_iter: int = 10_000, C: float = 1.0,
                 solver: str = "lbfgs", **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.C = C
        self.solver = solver

    def build(self) -> None:
        """Instantiate the LogisticRegression estimator.

        Sets ``self.model`` to a configured
        :class:`~sklearn.linear_model.LogisticRegression` instance.
        """
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            solver=self.solver,
            random_state=self.random_state,
        )


# ---------------------------------------------------------------------------
# Decision Tree model
# ---------------------------------------------------------------------------

class DecisionTreeModel(BaseModel):
    """Decision Tree classifier for the breast cancer dataset.

    Attributes
    ----------
    max_depth : int or None
        Maximum depth of the tree.  ``None`` means nodes are expanded until
        all leaves are pure or contain fewer than ``min_samples_split``
        samples (default ``None``).
    criterion : str
        Function to measure split quality; ``'gini'`` or ``'entropy'``
        (default ``'gini'``).
    min_samples_split : int
        Minimum number of samples required to split an internal node
        (default 2).
    model : DecisionTreeClassifier or None
        The underlying :class:`~sklearn.tree.DecisionTreeClassifier`
        estimator.  ``None`` until :meth:`build` is called.

    Limitations
    -----------
    * Prone to overfitting, especially when ``max_depth`` is unconstrained.
    * High variance: small changes in the data can produce very different trees.
    * Not robust to outliers in continuous features.
    * Feature scaling has no effect on tree splits (though the parent class
      still applies it for consistency with sibling classes).
    * Greedy split selection does not guarantee a globally optimal tree.

    Examples
    --------
    >>> dt = DecisionTreeModel(max_depth=5)
    >>> results = dt.run()
    >>> print(results["accuracy"])
    """

    def __init__(self, max_depth: int | None = None, criterion: str = "gini",
                 min_samples_split: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split

    def build(self) -> None:
        """Instantiate the DecisionTreeClassifier estimator.

        Sets ``self.model`` to a configured
        :class:`~sklearn.tree.DecisionTreeClassifier` instance.
        """
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )


# ---------------------------------------------------------------------------
# Random Forest model
# ---------------------------------------------------------------------------

class RandomForestModel(BaseModel):
    """Random Forest classifier for the breast cancer dataset.

    Attributes
    ----------
    n_estimators : int
        Number of trees in the forest (default 100).
    max_depth : int or None
        Maximum depth of each individual tree.  ``None`` means trees are
        grown until all leaves are pure (default ``None``).
    max_features : str or int or float
        Number of features to consider when looking for the best split
        (default ``'sqrt'``).
    model : RandomForestClassifier or None
        The underlying :class:`~sklearn.ensemble.RandomForestClassifier`
        estimator.  ``None`` until :meth:`build` is called.

    Limitations
    -----------
    * More computationally expensive and memory-intensive than a single tree
      or logistic regression.
    * Predictions are harder to interpret than a single decision tree.
    * For very high-dimensional or very sparse data, extremely randomised
      forests may be preferred.
    * Training time grows linearly with ``n_estimators``; choosing a very
      large forest may be impractical for time-sensitive pipelines.
    * Does not natively handle missing values.

    Examples
    --------
    >>> rf = RandomForestModel(n_estimators=200)
    >>> results = rf.run()
    >>> print(results["accuracy"])
    """

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None,
                 max_features: str = "sqrt", **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features

    def build(self) -> None:
        """Instantiate the RandomForestClassifier estimator.

        Sets ``self.model`` to a configured
        :class:`~sklearn.ensemble.RandomForestClassifier` instance.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
        )
