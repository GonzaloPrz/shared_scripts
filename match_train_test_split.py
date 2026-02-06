import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind


def _safe_std(values):
    std = float(np.nanstd(values, ddof=0))
    return std if std > 0 else 1.0


def _compute_numeric_smd(group_a, group_b, column):
    mean_a = float(group_a[column].mean())
    mean_b = float(group_b[column].mean())
    std_a = float(group_a[column].std(ddof=0))
    std_b = float(group_b[column].std(ddof=0))
    pooled = np.sqrt((std_a**2 + std_b**2) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float(abs(mean_a - mean_b) / pooled)


def _compute_category_gap(group_a, group_b, column):
    categories = sorted(
        set(group_a[column].dropna().unique()).union(set(group_b[column].dropna().unique()))
    )
    if not categories:
        return 0.0
    prop_a = (
        group_a[column]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )
    prop_b = (
        group_b[column]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )
    return float(np.max(np.abs(prop_a - prop_b)))


def _compute_internal_balance(data, group_col, numeric_covariates, categorical_covariates):
    groups = sorted(data[group_col].unique())
    if len(groups) != 2:
        raise ValueError(f"{group_col} debe tener exactamente dos clases para emparejar.")

    group_a = data[data[group_col] == groups[0]]
    group_b = data[data[group_col] == groups[1]]

    numeric_smd = {
        col: _compute_numeric_smd(group_a, group_b, col) for col in numeric_covariates
    }
    categorical_gap = {
        col: _compute_category_gap(group_a, group_b, col) for col in categorical_covariates
    }

    return {
        "numeric_smd": numeric_smd,
        "categorical_gap": categorical_gap,
        "max_numeric_smd": float(max(numeric_smd.values(), default=0.0)),
        "max_categorical_gap": float(max(categorical_gap.values(), default=0.0)),
    }


def _ttest_pvalue(sample_a, sample_b):
    a = pd.Series(sample_a).dropna().astype(float).to_numpy()
    b = pd.Series(sample_b).dropna().astype(float).to_numpy()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    if np.allclose(a, a[0]) and np.allclose(b, b[0]):
        return 1.0 if np.isclose(a[0], b[0]) else 0.0

    _, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    if np.isnan(p):
        return 1.0 if np.isclose(np.nanmean(a), np.nanmean(b)) else 0.0
    return float(p)


def _categorical_pvalue(sample_a, sample_b):
    a = pd.Series(sample_a).dropna()
    b = pd.Series(sample_b).dropna()
    categories = sorted(set(a.unique()).union(set(b.unique())))
    if len(categories) <= 1:
        return 1.0

    counts_a = a.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=int)
    counts_b = b.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=int)
    contingency = np.vstack([counts_a, counts_b])

    if contingency.shape == (2, 2):
        _, p_chi2, _, expected = chi2_contingency(contingency, correction=False)
        if np.any(expected < 5):
            _, p_fisher = fisher_exact(contingency)
            return float(p_fisher)
        return float(p_chi2)

    _, p, _, _ = chi2_contingency(contingency, correction=False)
    return float(p)


def _collect_pvalues(significance_dict):
    pvals = []
    for key in ["numeric_pvalues", "categorical_pvalues"]:
        values = significance_dict.get(key, {})
        for val in values.values():
            if val is not None and not np.isnan(val):
                pvals.append(float(val))
    return pvals


def _all_non_significant(significance_dict, alpha):
    pvals = _collect_pvalues(significance_dict)
    if not pvals:
        return False
    return all(p >= float(alpha) for p in pvals)


def _compute_internal_significance(data, group_col, numeric_covariates, categorical_covariates):
    groups = sorted(data[group_col].unique())
    if len(groups) != 2:
        raise ValueError(f"{group_col} debe tener exactamente dos clases para emparejar.")

    group_a = data[data[group_col] == groups[0]]
    group_b = data[data[group_col] == groups[1]]

    numeric_pvalues = {
        col: _ttest_pvalue(group_a[col], group_b[col]) for col in numeric_covariates
    }
    categorical_pvalues = {
        col: _categorical_pvalue(group_a[col], group_b[col]) for col in categorical_covariates
    }

    pvals = [
        float(x)
        for x in list(numeric_pvalues.values()) + list(categorical_pvalues.values())
        if x is not None and not np.isnan(x)
    ]

    return {
        "numeric_pvalues": numeric_pvalues,
        "categorical_pvalues": categorical_pvalues,
        "min_pvalue": float(min(pvals)) if pvals else np.nan,
    }


def _build_optimal_pairs(
    data,
    group_col,
    numeric_covariates,
    exact_match_covariates,
    mismatch_penalty=1e6,
):
    groups = sorted(data[group_col].unique())
    if len(groups) != 2:
        raise ValueError(f"{group_col} debe tener exactamente dos clases para emparejar.")

    group_a = data[data[group_col] == groups[0]].copy()
    group_b = data[data[group_col] == groups[1]].copy()

    if len(group_a) <= len(group_b):
        minority, majority = group_a, group_b
    else:
        minority, majority = group_b, group_a

    n_pairs = len(minority)
    if n_pairs < 2:
        raise ValueError("No hay suficientes muestras para crear pares.")

    if numeric_covariates:
        full_numeric = pd.concat(
            [minority[numeric_covariates], majority[numeric_covariates]], axis=0
        )
        means = full_numeric.mean()
        stds = full_numeric.std(ddof=0).replace(0, 1)
        minor_num = ((minority[numeric_covariates] - means) / stds).to_numpy(dtype=float)
        major_num = ((majority[numeric_covariates] - means) / stds).to_numpy(dtype=float)
        distances = np.sqrt(((minor_num[:, None, :] - major_num[None, :, :]) ** 2).sum(axis=2))
    else:
        distances = np.zeros((len(minority), len(majority)), dtype=float)

    for col in exact_match_covariates:
        mismatch = (
            minority[col].to_numpy()[:, None] != majority[col].to_numpy()[None, :]
        ).astype(float)
        distances += mismatch * mismatch_penalty

    row_idx, col_idx = linear_sum_assignment(distances)
    pairs = []
    for pair_id, (i, j) in enumerate(zip(row_idx, col_idx)):
        row_minor = minority.iloc[i]
        row_major = majority.iloc[j]
        pairs.append(
            {
                "pair_id": int(pair_id),
                "distance": float(distances[i, j]),
                "row_ids": [int(row_minor["_row_id"]), int(row_major["_row_id"])],
            }
        )

    return sorted(pairs, key=lambda x: x["distance"])


def _pairs_to_frame(data, pairs):
    row_to_pair = {}
    row_to_distance = {}
    for pair in pairs:
        for row_id in pair["row_ids"]:
            row_to_pair[row_id] = pair["pair_id"]
            row_to_distance[row_id] = pair["distance"]

    selected = data[data["_row_id"].isin(row_to_pair.keys())].copy()
    selected["pair_id"] = selected["_row_id"].map(row_to_pair).astype(int)
    selected["pair_distance"] = selected["_row_id"].map(row_to_distance).astype(float)
    return selected


def _choose_pair_subset(
    data,
    all_pairs,
    group_col,
    numeric_covariates,
    categorical_covariates,
    alpha=0.05,
    min_pair_retention=0.7,
    retention_penalty=0.2,
):
    total_pairs = len(all_pairs)
    min_pairs = max(2, int(np.ceil(total_pairs * min_pair_retention)))

    best_fallback = None
    for n_pairs in range(total_pairs, min_pairs - 1, -1):
        candidate_pairs = all_pairs[:n_pairs]
        candidate_data = _pairs_to_frame(data, candidate_pairs)
        significance = _compute_internal_significance(
            candidate_data, group_col, numeric_covariates, categorical_covariates
        )

        if _all_non_significant(significance, alpha):
            return candidate_pairs, significance, "all_pvalues_above_alpha"

        pvals = _collect_pvalues(significance)
        deficits = float(sum(max(0.0, float(alpha) - p) for p in pvals))
        min_p = float(min(pvals)) if pvals else 0.0
        objective = deficits + retention_penalty * (1.0 - (n_pairs / total_pairs)) - 0.01 * min_p
        if best_fallback is None or objective < best_fallback["objective"]:
            best_fallback = {
                "objective": float(objective),
                "pairs": candidate_pairs,
                "significance": significance,
            }

    if best_fallback is None:
        candidate_pairs = all_pairs
        candidate_data = _pairs_to_frame(data, candidate_pairs)
        significance = _compute_internal_significance(
            candidate_data, group_col, numeric_covariates, categorical_covariates
        )
        return candidate_pairs, significance, "all_pairs"

    return (
        best_fallback["pairs"],
        best_fallback["significance"],
        "fallback_best_pvalues",
    )


def _between_set_gaps(
    train_data,
    test_data,
    group_col,
    numeric_covariates,
    categorical_covariates,
):
    groups = sorted(train_data[group_col].unique())
    numeric_gap = {}
    categorical_gap = {}

    for group_value in groups:
        train_group = train_data[train_data[group_col] == group_value]
        test_group = test_data[test_data[group_col] == group_value]

        for col in numeric_covariates:
            key = f"{col}|group={group_value}"
            scale = _safe_std(pd.concat([train_data[col], test_data[col]], axis=0))
            numeric_gap[key] = float(abs(train_group[col].mean() - test_group[col].mean()) / scale)

        for col in categorical_covariates:
            key = f"{col}|group={group_value}"
            categories = sorted(
                set(train_data[col].dropna().unique()).union(set(test_data[col].dropna().unique()))
            )
            if not categories:
                categorical_gap[key] = 0.0
                continue
            prop_train = (
                train_group[col]
                .value_counts(normalize=True)
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
            prop_test = (
                test_group[col]
                .value_counts(normalize=True)
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
            categorical_gap[key] = float(np.max(np.abs(prop_train - prop_test)))

    return {
        "numeric_gap": numeric_gap,
        "categorical_gap": categorical_gap,
        "max_numeric_gap": float(max(numeric_gap.values(), default=0.0)),
        "max_categorical_gap": float(max(categorical_gap.values(), default=0.0)),
    }


def _compute_between_set_significance(
    train_data,
    test_data,
    group_col,
    numeric_covariates,
    categorical_covariates,
):
    groups = sorted(train_data[group_col].unique())
    numeric_pvalues = {}
    categorical_pvalues = {}

    for group_value in groups:
        train_group = train_data[train_data[group_col] == group_value]
        test_group = test_data[test_data[group_col] == group_value]

        for col in numeric_covariates:
            key = f"{col}|group={group_value}"
            numeric_pvalues[key] = _ttest_pvalue(train_group[col], test_group[col])

        for col in categorical_covariates:
            key = f"{col}|group={group_value}"
            categorical_pvalues[key] = _categorical_pvalue(train_group[col], test_group[col])

    pvals = [
        float(x)
        for x in list(numeric_pvalues.values()) + list(categorical_pvalues.values())
        if x is not None and not np.isnan(x)
    ]
    return {
        "numeric_pvalues": numeric_pvalues,
        "categorical_pvalues": categorical_pvalues,
        "min_pvalue": float(min(pvals)) if pvals else np.nan,
    }


def _split_objective(
    train_data,
    test_data,
    group_col,
    numeric_covariates,
    categorical_covariates,
    alpha,
):
    train_significance = _compute_internal_significance(
        train_data, group_col, numeric_covariates, categorical_covariates
    )
    test_significance = _compute_internal_significance(
        test_data, group_col, numeric_covariates, categorical_covariates
    )
    between_significance = _compute_between_set_significance(
        train_data, test_data, group_col, numeric_covariates, categorical_covariates
    )

    train_balance = _compute_internal_balance(
        train_data, group_col, numeric_covariates, categorical_covariates
    )
    test_balance = _compute_internal_balance(
        test_data, group_col, numeric_covariates, categorical_covariates
    )
    between_gap = _between_set_gaps(
        train_data, test_data, group_col, numeric_covariates, categorical_covariates
    )

    all_pvals = (
        _collect_pvalues(train_significance)
        + _collect_pvalues(test_significance)
        + _collect_pvalues(between_significance)
    )
    deficits = float(sum(max(0.0, float(alpha) - p) for p in all_pvals))
    min_p = float(min(all_pvals)) if all_pvals else 0.0
    score = deficits - 0.01 * min_p

    all_non_significant = (
        _all_non_significant(train_significance, alpha)
        and _all_non_significant(test_significance, alpha)
        and _all_non_significant(between_significance, alpha)
    )

    return (
        float(score),
        all_non_significant,
        train_significance,
        test_significance,
        between_significance,
        train_balance,
        test_balance,
        between_gap,
    )


def _optimize_train_test_split(
    matched_data,
    group_col,
    numeric_covariates,
    categorical_covariates,
    test_size,
    alpha=0.05,
    random_state=42,
    n_trials=3000,
):
    pair_ids = np.array(sorted(matched_data["pair_id"].unique()))
    n_pairs = len(pair_ids)
    if n_pairs < 2:
        raise ValueError("Se requieren al menos 2 pares para dividir en train/test.")

    n_test = int(round(n_pairs * float(test_size)))
    n_test = max(1, min(n_pairs - 1, n_test))

    rng = np.random.default_rng(int(random_state))
    best = None

    for _ in range(max(1, int(n_trials))):
        test_pair_ids = rng.choice(pair_ids, size=n_test, replace=False)
        test_pair_set = set(int(x) for x in test_pair_ids)
        test_data = matched_data[matched_data["pair_id"].isin(test_pair_set)]
        train_data = matched_data[~matched_data["pair_id"].isin(test_pair_set)]

        (
            score,
            all_non_significant,
            train_significance,
            test_significance,
            between_significance,
            train_balance,
            test_balance,
            between_gap,
        ) = _split_objective(
            train_data,
            test_data,
            group_col,
            numeric_covariates,
            categorical_covariates,
            alpha,
        )

        candidate = {
            "score": float(score),
            "all_non_significant": bool(all_non_significant),
            "train_pair_ids": sorted(int(x) for x in train_data["pair_id"].unique()),
            "test_pair_ids": sorted(int(x) for x in test_data["pair_id"].unique()),
            "train_significance": train_significance,
            "test_significance": test_significance,
            "between_significance": between_significance,
            "train_balance": train_balance,
            "test_balance": test_balance,
            "between_gap": between_gap,
        }

        if best is None:
            best = candidate
            continue

        if candidate["all_non_significant"] and not best["all_non_significant"]:
            best = candidate
            continue

        if candidate["all_non_significant"] == best["all_non_significant"]:
            if candidate["score"] < best["score"]:
                best = candidate

    return best


def generate_matched_train_test(
    data,
    group_col="group",
    set_col="set",
    covariates=None,
    categorical_covariates=None,
    test_size=None,
    random_state=42,
    n_trials=3000,
    alpha=0.05,
    min_pair_retention=0.7,
    retention_penalty=0.2,
    mismatch_penalty=1e6,
):
    covariates = list(covariates or ["age", "sex", "education"])
    categorical_covariates = list(categorical_covariates or ["sex"])
    numeric_covariates = [c for c in covariates if c not in categorical_covariates]

    required_columns = [group_col] + covariates
    missing_columns = [c for c in required_columns if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    full_data = data.copy().reset_index(drop=True)
    full_data["_row_id"] = np.arange(len(full_data))

    complete_mask = full_data[required_columns].notna().all(axis=1)
    clean_data = full_data[complete_mask].copy().reset_index(drop=True)
    if clean_data.shape[0] < 4:
        raise ValueError("Muy pocas filas completas para hacer emparejamiento.")

    clean_data[group_col] = pd.to_numeric(clean_data[group_col], errors="coerce")
    clean_data = clean_data[clean_data[group_col].isin([0, 1])].copy()
    clean_data[group_col] = clean_data[group_col].astype(int)
    if clean_data[group_col].nunique() != 2:
        raise ValueError(f"{group_col} debe tener exactamente dos clases (0 y 1).")

    if test_size is None:
        if set_col in clean_data.columns:
            set_str = clean_data[set_col].astype(str).str.lower()
            n_train = int((set_str == "train").sum())
            n_test = int((set_str == "test").sum())
            if (n_train + n_test) > 0:
                test_size = n_test / (n_train + n_test)
        if test_size is None:
            test_size = 0.3

    all_pairs = _build_optimal_pairs(
        clean_data,
        group_col=group_col,
        numeric_covariates=numeric_covariates,
        exact_match_covariates=categorical_covariates,
        mismatch_penalty=mismatch_penalty,
    )

    selected_pairs, overall_significance, pair_selection_mode = _choose_pair_subset(
        clean_data,
        all_pairs,
        group_col=group_col,
        numeric_covariates=numeric_covariates,
        categorical_covariates=categorical_covariates,
        alpha=float(alpha),
        min_pair_retention=float(min_pair_retention),
        retention_penalty=float(retention_penalty),
    )

    matched = _pairs_to_frame(clean_data, selected_pairs)
    split = _optimize_train_test_split(
        matched_data=matched,
        group_col=group_col,
        numeric_covariates=numeric_covariates,
        categorical_covariates=categorical_covariates,
        test_size=float(test_size),
        alpha=float(alpha),
        random_state=int(random_state),
        n_trials=int(n_trials),
    )

    train_pair_set = set(split["train_pair_ids"])
    matched["set_matched"] = np.where(matched["pair_id"].isin(train_pair_set), "train", "test")

    if set_col in matched.columns:
        matched[f"{set_col}_original"] = matched[set_col]
        matched[set_col] = matched["set_matched"]
    else:
        matched[set_col] = matched["set_matched"]

    train_data = matched[matched["set_matched"] == "train"].copy()
    test_data = matched[matched["set_matched"] == "test"].copy()

    matched_row_ids = set(matched["_row_id"].tolist())
    unmatched_data = full_data[~full_data["_row_id"].isin(matched_row_ids)].copy()

    drop_columns = ["_row_id", "set_matched"]
    matched_final = matched.drop(columns=[c for c in drop_columns if c in matched.columns])
    train_final = train_data.drop(columns=[c for c in drop_columns if c in train_data.columns])
    test_final = test_data.drop(columns=[c for c in drop_columns if c in test_data.columns])
    unmatched_final = unmatched_data.drop(columns=[c for c in ["_row_id"] if c in unmatched_data.columns])

    diagnostics = {
        "alpha": float(alpha),
        "input_rows": int(len(full_data)),
        "complete_rows_used": int(len(clean_data)),
        "all_pairs_count": int(len(all_pairs)),
        "selected_pairs_count": int(len(selected_pairs)),
        "selected_pair_distance_mean": float(np.mean([p["distance"] for p in selected_pairs])),
        "selected_pair_distance_max": float(np.max([p["distance"] for p in selected_pairs])),
        "pair_selection_mode": pair_selection_mode,
        "test_size_used": float(test_size),
        "rows_in_matched": int(len(matched_final)),
        "rows_in_train": int(len(train_final)),
        "rows_in_test": int(len(test_final)),
        "rows_unmatched": int(len(unmatched_final)),
        "overall_internal_significance": overall_significance,
        "overall_internal_balance": _compute_internal_balance(
            matched, group_col, numeric_covariates, categorical_covariates
        ),
        "train_internal_significance": split["train_significance"],
        "test_internal_significance": split["test_significance"],
        "between_train_test_significance": split["between_significance"],
        "train_internal_balance": split["train_balance"],
        "test_internal_balance": split["test_balance"],
        "between_train_test_balance": split["between_gap"],
        "split_all_non_significant": bool(split["all_non_significant"]),
        "split_objective_score": float(split["score"]),
    }

    return matched_final, train_final, test_final, unmatched_final, diagnostics


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Genera train/test emparejados usando p-values (t-test/chi-cuadrado)."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta del CSV de entrada.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio de salida. Por defecto, mismo directorio del input.",
    )
    parser.add_argument("--group_col", type=str, default="group")
    parser.add_argument("--set_col", type=str, default="set")
    parser.add_argument("--covariates", nargs="+", default=["age", "sex", "education"])
    parser.add_argument("--categorical_covariates", nargs="+", default=["sex"])
    parser.add_argument("--test_size", type=float, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min_pair_retention", type=float, default=0.7)
    parser.add_argument("--retention_penalty", type=float, default=0.2)
    parser.add_argument("--mismatch_penalty", type=float, default=1e6)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    matched, train, test, unmatched, diagnostics = generate_matched_train_test(
        data=data,
        group_col=args.group_col,
        set_col=args.set_col,
        covariates=args.covariates,
        categorical_covariates=args.categorical_covariates,
        test_size=args.test_size,
        random_state=args.random_state,
        n_trials=args.n_trials,
        alpha=args.alpha,
        min_pair_retention=args.min_pair_retention,
        retention_penalty=args.retention_penalty,
        mismatch_penalty=args.mismatch_penalty,
    )

    stem = input_path.stem
    matched_path = output_dir / f"{stem}_matched_train_test.csv"
    train_path = output_dir / f"{stem}_matched_train.csv"
    test_path = output_dir / f"{stem}_matched_test.csv"
    unmatched_path = output_dir / f"{stem}_unmatched.csv"
    report_path = output_dir / f"{stem}_matching_report.json"

    matched.to_csv(matched_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_to_native(diagnostics), f, indent=2, ensure_ascii=False)

    print(f"Archivo combinado: {matched_path}")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print(f"No seleccionados: {unmatched_path}")
    print(f"Reporte: {report_path}")
    print(
        "Resumen -> "
        f"train={diagnostics['rows_in_train']}, "
        f"test={diagnostics['rows_in_test']}, "
        f"no_seleccionados={diagnostics['rows_unmatched']}, "
        f"all_non_significant={diagnostics['split_all_non_significant']}, "
        f"min_p_train={diagnostics['train_internal_significance']['min_pvalue']:.4f}, "
        f"min_p_test={diagnostics['test_internal_significance']['min_pvalue']:.4f}, "
        f"min_p_between={diagnostics['between_train_test_significance']['min_pvalue']:.4f}"
    )


if __name__ == "__main__":
    main()
