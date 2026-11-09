"""Models, tuning grids and imbalance strategies, shared by Phase 3 (unified) and Phase 4 (region-specific).

Phase 4 rule: same model families, same grids, same CV protocol, same imbalance strategy. Keep them HERE so that
cannot drift apart.

    pipe = build("linsvm", "balanced", {"C": 0.1}, d["col_weights"])      # unfitted pipeline
    s = scores(pipe.fit(Xtr, ytr), Xva)                                    # larger = more likely fracture

Every pipeline is  prep (StandardScaler + equal family weight) -> [SelectKBest] -> [SMOTE] -> classifier,
so scaling, feature selection and oversampling are all fitted on the training part of a fold only.
Imbalance strategies: "none", "balanced" (class_weight="balanced"), "smote" (SMOTE inside the pipeline).
The 4th strategy of the plan, threshold tuning, is a post-processing step (evaluate.best_threshold).
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from feature_utils import make_preprocessor

IMBALANCE = ("none", "balanced", "smote")


class PrecomputedRBFSVC(ClassifierMixin, BaseEstimator):
    """RBF SVM that builds the kernel matrix with BLAS (float32 matrix product) and hands it to libsvm.
    libsvm's own dense RBF kernel does not use BLAS, which is very slow at 8,641 dimensions.
    Same model as sklearn's SVC(kernel="rbf", gamma=gamma): only the kernel computation differs.
    Keeps only the support vectors after fitting."""

    def __init__(self, C=1.0, gamma=0.2, class_weight=None):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight

    @staticmethod
    def _rbf(A, B, gamma):
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
        np.maximum(d2, 0.0, out=d2)
        return np.exp(-gamma * d2, dtype=np.float32)

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        svc = SVC(C=self.C, kernel="precomputed", class_weight=self.class_weight, cache_size=500)
        svc.fit(self._rbf(X, X, self.gamma), y)
        self.classes_ = svc.classes_
        self.sv_ = X[svc.support_]
        self.dual_coef_ = svc.dual_coef_[0].astype(np.float32)
        self.intercept_ = float(svc.intercept_[0])
        return self

    def decision_function(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return (self._rbf(X, self.sv_, self.gamma) @ self.dual_coef_ + self.intercept_).astype(np.float64)


@dataclass
class ModelSpec:
    name: str
    label: str
    grid: dict                       # classifier keyword -> list of values (full cartesian product is searched)
    make: callable                   # (params, class_weight, seed) -> classifier
    default_thr: float = 0.0         # "default" decision threshold on scores(): 0 for margins, 0.5 for probabilities
    kbest: int = None                # SelectKBest(f_classif, k) before the classifier (speed, see notes)
    supports_cw: bool = True
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _logreg(l1_ratio, solver):
    def make(p, cw, seed):
        return LogisticRegression(C=p["C"], l1_ratio=l1_ratio, solver=solver, class_weight=cw, max_iter=1000,
                                  tol=1e-3 if solver == "liblinear" else 1e-4, random_state=seed)
    return make


SPECS = {s.name: s for s in [
    # Features are block-weighted (each family has total weight 1, so ||x||^2 ~ 5), which moves the useful C
    # range from the plan's {0.01..100} to roughly 0.03-1 for the linear SVM (Phase 2: best C = 0.1).
    ModelSpec("logreg_l2", "Logistic regression (L2)", {"C": [0.03, 0.1, 0.3, 1, 3, 10]},
              _logreg(0.0, "lbfgs"), notes="lbfgs"),
    ModelSpec("logreg_l1", "Logistic regression (L1)", {"C": [0.1, 0.3, 1, 3, 10, 30]},
              _logreg(1.0, "liblinear"), notes="liblinear; L1 needs a larger C than L2 because block weighting shrinks inputs"),
    ModelSpec("linsvm", "Linear SVM", {"C": [0.01, 0.03, 0.1, 0.3, 1]},
              lambda p, cw, seed: LinearSVC(C=p["C"], class_weight=cw, dual="auto", max_iter=20000, random_state=seed)),
    # RBF: with block weighting E||x - x'||^2 ~ 10, so sklearn's gamma="scale" is ~0.2. The plan's {1e-3, 1e-4} assume
    # unweighted features (||x||^2 ~ 8,641) and would be ~linear here, so the grid is expressed around gamma="scale".
    ModelSpec("rbfsvm", "RBF SVM", {"C": [0.1, 1, 10, 100], "gamma": [0.8, 0.2, 0.05, 0.0125]},
              lambda p, cw, seed: PrecomputedRBFSVC(C=p["C"], gamma=p["gamma"], class_weight=cw),
              notes="gamma 0.2 ~ sklearn 'scale' after block weighting; 0.8 = 4x scale (added after the first nested run chose 0.2 at the upper edge in 4/5 folds); 0.05, 0.0125 = scale/4, scale/16"),
    # n_estimators is not tuned: more trees only reduce variance, so it is fixed at 500 (the plan's grid also lists 200).
    ModelSpec("rf", "Random forest", {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5]},
              lambda p, cw, seed: RandomForestClassifier(
                  n_estimators=500, max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                  class_weight=("balanced_subsample" if cw else None), n_jobs=1, random_state=seed),
              default_thr=0.5, kbest=1000, notes="n_estimators fixed at 500"),
    ModelSpec("hgb", "Gradient boosting (HistGB)",
              {"learning_rate": [0.03, 0.1], "max_depth": [3, 5], "max_iter": [200, 400]},
              lambda p, cw, seed: HistGradientBoostingClassifier(
                  learning_rate=p["learning_rate"], max_depth=p["max_depth"], max_iter=p["max_iter"],
                  class_weight=cw, early_stopping=False, random_state=seed),
              kbest=300, notes="kbest 300 (boosting cost grows with the number of columns)"),
    # optional extra from the plan ("optionally k-NN"); no class weights, so only imbalance = none / smote
    ModelSpec("knn", "k-NN", {"n_neighbors": [5, 15, 45, 135]},
              lambda p, cw, seed: KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights="distance", n_jobs=1),
              default_thr=0.5, supports_cw=False),
]}
DEFAULT_MODELS = ["logreg_l2", "logreg_l1", "linsvm", "rbfsvm", "rf", "hgb"]


# --fast: coarser grids for the two slow tree models (same families, fewer settings)
FAST_GRIDS = {"rf": {"max_depth": [None, 10], "min_samples_leaf": [1, 5]},
              "hgb": {"learning_rate": [0.1], "max_depth": [3, 5], "max_iter": [200, 400]}}


def grid_for(model, fast=False):
    return FAST_GRIDS[model] if fast and model in FAST_GRIDS else SPECS[model].grid


def expand_grid(model, grid=None):
    """List of parameter dicts (cartesian product, in the order the grid is written: simplest first)."""
    g = grid or SPECS[model].grid
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]


def build(model, imbalance, params, col_weights, seed=42, kbest="spec"):
    """Unfitted pipeline. `kbest`: "spec" = the model's default, None = no selection, int = SelectKBest k."""
    spec = SPECS[model]
    if imbalance not in IMBALANCE:
        raise ValueError(f"imbalance must be one of {IMBALANCE}")
    if imbalance == "balanced" and not spec.supports_cw:
        raise ValueError(f"{model} has no class weights; use imbalance 'none' or 'smote'")
    k = spec.kbest if kbest == "spec" else kbest
    steps = list(make_preprocessor(col_weights).steps)        # scaler, block weight (imblearn forbids nested pipelines)
    if k:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", spec.make(params, "balanced" if imbalance == "balanced" else None, seed)))
    return ImbPipeline(steps)


def scores(fitted, X):
    """Fracture score, larger = more likely fractured (decision function if the model has one, else probability)."""
    if hasattr(fitted, "decision_function"):
        return np.asarray(fitted.decision_function(X), dtype=float)
    return fitted.predict_proba(X)[:, 1].astype(float)


def default_threshold(model):
    return SPECS[model].default_thr
