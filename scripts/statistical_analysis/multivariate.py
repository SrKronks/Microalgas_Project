from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing.data_loader import DataSchema


def multivariate_analysis(
    df: pd.DataFrame,
    schema: DataSchema,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = df[schema.analysis_columns].apply(pd.to_numeric, errors="coerce")
    saved: dict[str, Path] = {}

    for method in ("pearson", "spearman", "kendall"):
        path = output_dir / f"correlation_{method}.csv"
        try:
            numeric.corr(method=method).to_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            logger.info("Skipping %s correlation: %s", method, exc)
            pd.DataFrame({"status": ["skipped"], "reason": [str(exc)]}).to_csv(path, index=False, encoding="utf-8-sig")
        saved[f"correlation_{method}"] = path

    covariance_path = output_dir / "covariance_matrix.csv"
    numeric.cov().to_csv(covariance_path, encoding="utf-8-sig")
    saved["covariance"] = covariance_path

    pca_path = output_dir / "pca_scores.csv"
    _pca(numeric, pca_path, logger)
    saved["pca"] = pca_path

    _optional_embedding(numeric, output_dir / "tsne_scores.csv", "tsne", logger)
    _optional_embedding(numeric, output_dir / "umap_scores.csv", "umap", logger)
    _optional_clustering(numeric, output_dir / "temporal_clustering.csv", logger)
    return saved


def _pca(numeric: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.shape[0] < 3 or clean.shape[1] < 2:
        pd.DataFrame({"status": ["skipped"], "reason": ["too_few_complete_rows"]}).to_csv(path, index=False)
        return
    try:
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        scaled = StandardScaler().fit_transform(clean)
        pca = PCA(n_components=min(3, clean.shape[1], clean.shape[0])).fit(scaled)
        scores = pca.transform(scaled)
        result = pd.DataFrame(scores, index=clean.index, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
        for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
            result[f"PC{i}_explained_variance_ratio"] = ratio
        result.to_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("Falling back to numpy PCA: %s", exc)
        values = clean.to_numpy(dtype=float)
        std = values.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        values = (values - values.mean(axis=0)) / std
        values = np.nan_to_num(values)
        _, _, vt = np.linalg.svd(values, full_matrices=False)
        scores = values @ vt[: min(3, vt.shape[0])].T
        pd.DataFrame(scores, index=clean.index, columns=[f"PC{i+1}" for i in range(scores.shape[1])]).to_csv(
            path,
            encoding="utf-8-sig",
        )


def _optional_embedding(numeric: pd.DataFrame, path: Path, kind: str, logger: logging.Logger) -> None:
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.shape[0] < 8 or clean.shape[1] < 2:
        pd.DataFrame({"status": ["skipped"], "reason": ["too_few_complete_rows"]}).to_csv(path, index=False)
        return
    try:
        if kind == "tsne":
            from sklearn.manifold import TSNE  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore

            scaled = StandardScaler().fit_transform(clean)
            scores = TSNE(n_components=2, perplexity=min(5, clean.shape[0] - 1), init="pca", learning_rate="auto").fit_transform(scaled)
        else:
            import umap  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore

            scaled = StandardScaler().fit_transform(clean)
            scores = umap.UMAP(n_components=2, random_state=42).fit_transform(scaled)
        pd.DataFrame(scores, index=clean.index, columns=[f"{kind}_1", f"{kind}_2"]).to_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        logger.info("Skipping %s embedding: %s", kind, exc)
        pd.DataFrame({"status": ["skipped"], "reason": [str(exc)]}).to_csv(path, index=False)


def _optional_clustering(numeric: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.shape[0] < 4:
        pd.DataFrame({"status": ["skipped"], "reason": ["too_few_complete_rows"]}).to_csv(path, index=False)
        return
    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        scaled = StandardScaler().fit_transform(clean)
        labels = KMeans(n_clusters=min(3, clean.shape[0]), random_state=42, n_init="auto").fit_predict(scaled)
        pd.DataFrame({"cluster": labels}, index=clean.index).to_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        logger.info("Skipping clustering: %s", exc)
        pd.DataFrame({"status": ["skipped"], "reason": [str(exc)]}).to_csv(path, index=False)
