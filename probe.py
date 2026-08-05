"""Linear-probe measurement instrument for the self-distillation ablation.

This is the co-primary-metric machinery: given a frozen/adapted representation
(pooled features per clip), it reports both
  * TASK decodability  — ecotype / call-type, grouped by hydrophone so the probe
    can't cheat through the recording-condition shortcut (GroupKFold), and
  * NUISANCE decodability — hydrophone / date / SR, plain stratified CV (the
    nuisance *is* the target here, so grouping by it makes no sense).

A win for a given representation = task up AND nuisance down. Split membership
(which rows are train vs test) is passed in, so this file is independent of the
still-unsettled DCLDE hydrophone split — see [[project_dclde_split_v2]].

Feature extraction (backbone -> pooled features) lives elsewhere; this operates
on numpy arrays only, so it runs anywhere sklearn is installed.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


def _clf(C=1.0, seed=59):
    # standardize then multinomial logistic regression — the standard linear
    # evaluation protocol. class_weight balanced so majority classes don't
    # dominate balanced-accuracy via a degenerate constant predictor.
    return make_pipeline(
        StandardScaler(),
        # multinomial is the default for multiclass in modern sklearn (the
        # multi_class arg was removed in 1.8) — don't pass it.
        LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", random_state=seed,
        ),
    )


def _majority_baseline(y):
    _, counts = np.unique(y, return_counts=True)
    # balanced-accuracy of always predicting the majority class = 1/n_classes
    return 1.0 / len(counts)


def probe_cv(X, y, groups=None, cv="group", n_splits=5, C=1.0, seed=59):
    """Cross-validated linear-probe balanced accuracy.

    cv="group": GroupKFold on `groups` (use for TASK probes — group by hydrophone
                so recording condition can't leak across folds).
    cv="stratified": StratifiedKFold (use for NUISANCE probes — predicting the
                hydrophone/date/SR itself).
    Returns dict with mean/std balanced-acc, per-fold scores, majority baseline.
    """
    X = np.asarray(X); y = np.asarray(y)
    if cv == "group":
        assert groups is not None, "group CV needs `groups`"
        g = np.asarray(groups)
        k = min(n_splits, len(np.unique(g)))
        splitter = GroupKFold(n_splits=k).split(X, y, g)
    elif cv == "stratified":
        k = min(n_splits, np.min(np.unique(y, return_counts=True)[1]))
        splitter = StratifiedKFold(n_splits=max(k, 2), shuffle=True,
                                   random_state=seed).split(X, y)
    else:
        raise ValueError(f"unknown cv={cv!r}")

    scores = []
    for tr, te in splitter:
        clf = _clf(C=C, seed=seed).fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    scores = np.array(scores)
    return {
        "balanced_acc": float(scores.mean()),
        "std": float(scores.std()),
        "per_fold": scores.tolist(),
        "n_folds": len(scores),
        "majority_baseline": _majority_baseline(y),
    }


def probe_fixed_split(X, y, train_mask, test_mask, C=1.0, seed=59):
    """Fit on train rows, evaluate on test rows (no CV).

    This is the protocol chosen for the CALL-TYPE probe: train on the train-split
    call-type labels, evaluate on the held-out test split (decision 1.a). Also the
    natural readout for any 'train hydros -> test hydros' generalization check.
    """
    X = np.asarray(X); y = np.asarray(y)
    tr = np.asarray(train_mask, dtype=bool); te = np.asarray(test_mask, dtype=bool)
    clf = _clf(C=C, seed=seed).fit(X[tr], y[tr])
    return {
        "balanced_acc": float(balanced_accuracy_score(y[te], clf.predict(X[te]))),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "majority_baseline": _majority_baseline(y[te]),
    }


def diagnose(X, task_labels, nuisance_labels, hydrophone):
    """One-shot co-primary report for a representation.

    task_labels     : dict name -> array (e.g. {"ecotype": y_eco, "calltype": y_ct})
                      probed with GroupKFold-by-hydrophone.
    nuisance_labels : dict name -> array (e.g. {"hydro": h, "date": d, "sr": s})
                      probed with StratifiedKFold.
    hydrophone      : array of hydrophone ids, used as the group for task probes.
    Returns dict name -> probe result. Read it as: task ↑ good, nuisance ↓ good.
    """
    out = {}
    for name, y in task_labels.items():
        out[f"task/{name}"] = probe_cv(X, y, groups=hydrophone, cv="group")
    for name, y in nuisance_labels.items():
        out[f"nuisance/{name}"] = probe_cv(X, y, cv="stratified")
    return out


def split_tags(datasets, test_hydros, val_hydros, low_sr_hydros):
    """Map each row's hydrophone (`Dataset`) to its split tag.

    Returns an object array of 'test'/'val'/'low_sr'/'train' aligned to `datasets`.
    Build the #7 probe masks from it, e.g. for the call-type probe (train-on-train /
    eval-on-test, decision 1.a):
        tags = split_tags(df["Dataset"], TEST, VAL, LOW_SR)
        probe_fixed_split(X, y, train_mask=(tags=="train"), test_mask=(tags=="test"))
    Single source of truth = the finalized DCLDE split (see project_dclde_split_current):
        TEST=[StraitofGeorgia, BarkleyCanyon], VAL=[CarmanahPt].
    """
    test, val, low = set(test_hydros), set(val_hydros), set(low_sr_hydros)
    def tag(h):
        return ("test" if h in test else "val" if h in val
                else "low_sr" if h in low else "train")
    return np.array([tag(h) for h in np.asarray(datasets)], dtype=object)
