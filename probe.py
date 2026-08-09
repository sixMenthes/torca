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
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

# stdlib logging, NOT util.pylogger: this module's docstring promises it "runs anywhere
# sklearn is installed", and pylogger imports pytorch_lightning, which would quietly
# make that false and break anyone using probe.py from a bare analysis environment.
import logging

log = logging.getLogger(__name__)


def _fit_score(X, y, tr, te, C, seed):
    """One fold, fitted and scored. The unit of parallelism.

    Folds are independent, so the CV loop is embarrassingly parallel and was
    nonetheless serial until 2026-08-09. That was the dominant cost of a probe run:
    lbfgs on 21k rows of 768 features, multinomial over four classes, up to
    max_iter=2000, five times over — and again for the twelve-class nuisance probe.
    The C2 cell ran past twenty minutes of pure sklearn, against three minutes of GPU
    feature extraction.

    threadpool_limits(1) is not optional here. Without it every worker process would
    also try to use every core through BLAS, and the oversubscription costs more than
    the parallelism gains. One thread per worker, one worker per fold.

    Nothing numerical changes: same folds, same seed, same solver, same tolerance. This
    is purely a scheduling change, so results stay comparable with cells measured before
    it — which matters, because C4's 0.5388 and 0.9474 were measured serially.
    """
    with threadpool_limits(limits=1):
        clf = _clf(C=C, seed=seed).fit(X[tr], y[tr])
        score = balanced_accuracy_score(y[te], clf.predict(X[te]))
    # n_iter_ is the honest answer to "why is this slow". Hitting max_iter means lbfgs
    # never converged and the fold burned the full budget for a result that is not even
    # at an optimum, which is worth knowing rather than guessing at.
    n_iter = int(np.max(clf[-1].n_iter_)) if hasattr(clf[-1], "n_iter_") else -1
    return score, n_iter


def _run_folds(X, y, splits, C, seed, n_jobs):
    """Fit every fold, in parallel when asked, and report non-convergence once."""
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(X, y, tr, te, C, seed) for tr, te in splits
    )
    scores = np.array([s for s, _ in out])
    iters = [n for _, n in out]
    cap = _clf(C=C, seed=seed)[-1].max_iter
    if iters and max(iters) >= cap:
        log.warning(
            f"logistic regression hit max_iter={cap} on "
            f"{sum(1 for n in iters if n >= cap)}/{len(iters)} folds "
            f"(iterations {iters}) — the fit did not converge, and this is where the "
            f"probe's wall-clock goes"
        )
    return scores


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


def probe_cv(X, y, groups=None, cv="group", n_splits=5, C=1.0, seed=59, n_jobs=1):
    """Cross-validated linear-probe balanced accuracy.

    cv="group": GroupKFold on `groups` (use for TASK probes — group by hydrophone
                so recording condition can't leak across folds).
    cv="stratified": StratifiedKFold (use for NUISANCE probes — predicting the
                hydrophone/date/SR itself).
    n_jobs: folds fitted in parallel. 1, the default, keeps the old serial behaviour,
                which is what the ONLINE callbacks want — they run inside a training
                loop and should not fork a process per fold. Offline runs should pass
                their core count. Results are identical either way; see _fit_score.
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

    scores = _run_folds(X, y, list(splitter), C, seed, n_jobs)
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


def select_C(X, y, groups, C_grid=(0.01, 0.1, 1.0, 10.0), n_splits=5, seed=59, n_jobs=1):
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

    # len(C_grid) x k fits, so 20 at the defaults — the single most expensive thing in a
    # reported (non-sealed) probe run. Flattened into ONE parallel map rather than a
    # parallel inner loop per C, so all 20 fits are scheduled together and no core sits
    # idle waiting for the slowest fold of the current C.
    splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    jobs = [(C, tr, te) for C in C_grid for tr, te in splits]
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(X, y, tr, te, C, seed) for C, tr, te in jobs
    )

    best = (None, -1.0)
    for i, C in enumerate(C_grid):
        mean = float(np.mean([s for s, _ in out[i * len(splits):(i + 1) * len(splits)]]))
        if mean > best[1]:
            best = (C, mean)
    return best[0], best[1]


def probe_split_protocol(X, y, tags, hydrophone=None, C_grid=(0.01, 0.1, 1.0, 10.0),
                         seed=59, include_low_sr=False, c_selection="train_cv",
                         n_jobs=1):
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
                                     C_grid=C_grid, seed=seed, n_jobs=n_jobs)
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
                              min_per_hydro=50, n_splits=5, C=1.0, seed=59, n_jobs=1):
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
                   n_splits=n_splits, C=C, seed=seed, n_jobs=n_jobs)
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
