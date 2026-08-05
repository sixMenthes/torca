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


def select_C(X, y, groups, C_grid=(0.01, 0.1, 1.0, 10.0), n_splits=5, seed=59):
    """Pick C by GroupKFold-by-hydrophone WITHIN the given (training) rows.

    Inner loop of a nested cross-validation: the hyperparameter is chosen using only
    training hydrophones, so the test split is touched exactly once, for the final
    number. Selecting on 13 train hydrophones generalises across recording conditions
    far better than selecting on the single val hydrophone (CarmanahPt) would — and
    val cannot support GroupKFold at all, since n_splits=1 raises.

    Returns (best_C, mean score) and falls back to C=1.0 when there aren't enough
    groups to split.
    """
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    k = min(n_splits, len(np.unique(groups)))
    if k < 2 or len(np.unique(y)) < 2:
        return 1.0, None

    best = (None, -1.0)
    for C in C_grid:
        scores = []
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            clf = _clf(C=C, seed=seed).fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
        mean = float(np.mean(scores))
        if mean > best[1]:
            best = (C, mean)
    return best[0], best[1]


def probe_split_protocol(X, y, tags, hydrophone=None, C_grid=(0.01, 0.1, 1.0, 10.0),
                         seed=59, include_low_sr=False, c_selection="train_cv"):
    """Fit on train hydros, evaluate once on test. THE task protocol.

    Replaces GroupKFold-over-everything for the task metric, because that protocol
    leaks: some folds put adaptation-TRAIN hydrophones in the probe's test fold, and
    the adapted backbone has already seen those recording conditions while the frozen
    control has not. The adapted cell then wins for a reason that has nothing to do
    with representation quality, which is precisely the comparison being made.

    Here the probe's test set is the adaptation's held-out test hydrophones, so every
    cell is scored on conditions none of them adapted on.

    c_selection="train_cv" (default): choose C by GroupKFold within train — 13
    hydrophones, and the same procedure the call-type probe uses.
    c_selection="val": choose C on the val split instead (CarmanahPt alone). Kept as
    an option, but selecting on one recording condition is thin.

    Returns test balanced-acc plus a per-hydrophone breakdown — with only two test
    hydrophones the headline number is high-variance, and one of them carrying the
    result is something you want to see rather than average away.
    """
    X = np.asarray(X); y = np.asarray(y); tags = np.asarray(tags)

    train_m = tags == "train"
    if include_low_sr:
        train_m = train_m | (tags == "low_sr")
    val_m = tags == "val"
    test_m = tags == "test"

    if test_m.sum() == 0:
        raise ValueError("no test rows — check test_hydros against the manifest")
    if train_m.sum() == 0:
        raise ValueError("no train rows")

    best_C, sel_score = 1.0, None
    if c_selection == "train_cv" and hydrophone is not None:
        best_C, sel_score = select_C(X[train_m], y[train_m],
                                     np.asarray(hydrophone)[train_m],
                                     C_grid=C_grid, seed=seed)
    elif c_selection == "val" and val_m.sum() > 0 and len(np.unique(y[val_m])) > 1:
        scored = []
        for C in C_grid:
            clf = _clf(C=C, seed=seed).fit(X[train_m], y[train_m])
            scored.append((balanced_accuracy_score(y[val_m], clf.predict(X[val_m])), C))
        sel_score, best_C = max(scored)

    clf = _clf(C=best_C, seed=seed).fit(X[train_m], y[train_m])
    pred = clf.predict(X[test_m])

    per_hydro = {}
    if hydrophone is not None:
        h = np.asarray(hydrophone)[test_m]
        y_te = y[test_m]
        for name in np.unique(h):
            sel = h == name
            if len(np.unique(y_te[sel])) > 1:
                per_hydro[str(name)] = float(
                    balanced_accuracy_score(y_te[sel], pred[sel])
                )

    return {
        "balanced_acc": float(balanced_accuracy_score(y[test_m], pred)),
        "C": best_C,
        "selection_score": sel_score,
        "per_hydrophone": per_hydro,
        "n_train": int(train_m.sum()),
        "n_val": int(val_m.sum()),
        "n_test": int(test_m.sum()),
        "majority_baseline": _majority_baseline(y[test_m]),
    }


def probe_nuisance_background(X, hydrophone, tags, is_background,
                              min_per_hydro=50, n_splits=5, C=1.0, seed=59):
    """THE domain-confound diagnostic: hydrophone decodability from BACKGROUND clips.

    Restricting to Background is what makes this a measurement of *channel* rather
    than of content. Hydrophone identity correlates with ecotype in this dataset
    (Cpe_Elz is 89% TKW), so a site probe run over all clips can score high by reading
    the vocalisation instead of the recording condition. With no orca present, the
    only thing separating sites is instrument response, noise floor, depth and
    self-noise — the confound itself.

    Run on TRAIN, for frozen and adapted features, and read the pair: the drop is the
    result. Train (not test) because 13 hydrophones give a multi-class problem with
    real headroom (chance ~0.1) where a 2-hydrophone test split would be binary at
    chance 0.5 and would likely saturate at 1.0 for every cell. It also leaves test
    sealed.

    Caveat for interpretation: a drop here is not automatically attributable to the
    invariance objective, since adapting on in-domain audio reorganises features
    regardless. The `augmentations.student.background.p=0.0` run is the control that
    separates those.

    Hydrophones with fewer than `min_per_hydro` Background clips are dropped — a
    class with 3 examples destabilises the stratified CV without adding information.
    """
    tags = np.asarray(tags)
    h = np.asarray(hydrophone)
    m = (tags == "train") & np.asarray(is_background, dtype=bool)

    keep_h = {name for name in np.unique(h[m]) if (h[m] == name).sum() >= min_per_hydro}
    m = m & np.isin(h, list(keep_h))

    if len(keep_h) < 2:
        raise ValueError(
            f"only {len(keep_h)} hydrophone(s) with >={min_per_hydro} Background "
            f"clips — lower min_per_hydro or check the Background label"
        )

    out = probe_cv(np.asarray(X)[m], h[m], cv="stratified",
                   n_splits=n_splits, C=C, seed=seed)
    out["n_hydrophones"] = len(keep_h)
    out["n_clips"] = int(m.sum())
    return out


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
