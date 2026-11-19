from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from VmaxBuilder.database_retrieval.classes.kcat import (
    CATPRED_KCAT_CACHE_NAMESPACE,
    KCAT_NAMESPACE_TO_SCHEMA_VERSION,
    CatPredKcatPrediction,
    catpred_kcat_cache_key,
    catpred_kcat_prediction_to_dict,
)
from VmaxBuilder.utils.lookup_cache import LookupCache, get_default_cache_dir

from .service import run_inprocess_prediction_pipeline
from .types import PredictionRequest

_REQUIRED_INPUT_COLUMNS = {"SMILES", "sequence", "pdbpath"}
_CACHE_VALUE_COLUMNS = [
    "Prediction_log10",
    "SD_total",
    "SD_epistemic",
    "SD_aleatoric",
]


@dataclass(frozen=True)
class KcatInferenceResult:
    predictions_df: pd.DataFrame
    cache_file: Path
    newly_inferred_count: int
    cached_count: int


def _normalise_output_columns(prediction_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = prediction_df.copy()

    if "Prediction_log10" not in normalized_df.columns:
        prediction_column = "Prediction_(s^(-1))"
        if prediction_column in normalized_df.columns:
            normalized_df["Prediction_log10"] = normalized_df[prediction_column].astype(float)
            positive_mask = normalized_df["Prediction_log10"] > 0
            normalized_df.loc[positive_mask, "Prediction_log10"] = normalized_df.loc[
                positive_mask, "Prediction_log10"
            ].map(lambda value: np.log10(value))
            normalized_df.loc[~positive_mask, "Prediction_log10"] = 0.0
        else:
            raise ValueError(
                "CatPred output is missing both 'Prediction_log10' and "
                "'Prediction_(s^(-1))' columns."
            )

    if "pdbpath" not in normalized_df.columns:
        raise ValueError("CatPred output is missing required 'pdbpath' column.")

    if "SD_total" not in normalized_df.columns:
        normalized_df["SD_total"] = 0.0
    if "SD_epistemic" not in normalized_df.columns:
        normalized_df["SD_epistemic"] = 0.0
    if "SD_aleatoric" not in normalized_df.columns:
        normalized_df["SD_aleatoric"] = 0.0

    for column in _CACHE_VALUE_COLUMNS:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce").fillna(0.0)

    return normalized_df


def _cache_key_for_row(
    *,
    smiles: str,
    sequence: str,
    pdbpath: str,
    checkpoint_dir: str,
    use_gpu: bool,
) -> str:
    return catpred_kcat_cache_key(
        smiles=smiles,
        sequence=sequence,
        pdbpath=pdbpath,
        checkpoint_dir=checkpoint_dir,
        use_gpu=use_gpu,
    )


def _prediction_from_payload(payload: dict[str, Any]) -> CatPredKcatPrediction | None:
    try:
        return CatPredKcatPrediction(**payload)
    except Exception:
        return None


def _cached_predictions_to_dataframe(
    lookup_rows_df: pd.DataFrame,
    cache_map: dict[str, CatPredKcatPrediction],
    checkpoint_dir: str,
    use_gpu: bool,
) -> pd.DataFrame:
    if lookup_rows_df.empty:
        return pd.DataFrame(
            columns=["SMILES", "sequence", "pdbpath", "Prediction_log10", *_CACHE_VALUE_COLUMNS[1:]]
        )

    rows: list[dict[str, Any]] = []
    for row in lookup_rows_df.itertuples(index=False):
        key = _cache_key_for_row(
            smiles=str(row.SMILES),
            sequence=str(row.sequence),
            pdbpath=str(row.pdbpath),
            checkpoint_dir=checkpoint_dir,
            use_gpu=use_gpu,
        )
        cached = cache_map.get(key)
        if cached is None:
            continue
        rows.append(
            {
                "SMILES": str(row.SMILES),
                "sequence": str(row.sequence),
                "pdbpath": str(row.pdbpath),
                "Prediction_log10": float(cached.prediction_log10),
                "SD_total": float(cached.sd_total),
                "SD_epistemic": float(cached.sd_epistemic),
                "SD_aleatoric": float(cached.sd_aleatoric),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["SMILES", "sequence", "pdbpath", "Prediction_log10", *_CACHE_VALUE_COLUMNS[1:]]
        )

    return pd.DataFrame(rows)


def _inferred_predictions_to_cache_updates(
    inferred_df: pd.DataFrame,
    checkpoint_dir: str,
    use_gpu: bool,
) -> dict[str, dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}
    for row in inferred_df.itertuples(index=False):
        key = _cache_key_for_row(
            smiles=str(row.SMILES),
            sequence=str(row.sequence),
            pdbpath=str(row.pdbpath),
            checkpoint_dir=checkpoint_dir,
            use_gpu=use_gpu,
        )
        prediction = CatPredKcatPrediction(
            smiles=str(row.SMILES),
            sequence=str(row.sequence),
            pdbpath=str(row.pdbpath),
            checkpoint_dir=str(checkpoint_dir),
            use_gpu=bool(use_gpu),
            prediction_log10=float(row.Prediction_log10),
            sd_total=float(row.SD_total),
            sd_epistemic=float(row.SD_epistemic),
            sd_aleatoric=float(row.SD_aleatoric),
            error=None,
            cache_schema_version=KCAT_NAMESPACE_TO_SCHEMA_VERSION[
                CATPRED_KCAT_CACHE_NAMESPACE
            ],
        )
        updates[key] = catpred_kcat_prediction_to_dict(prediction)
    return updates


def infer_kcats(
    input_df: pd.DataFrame,
    checkpoint_dir: str,
    *,
    results_dir: str | Path = "../results",
    repo_root: str | Path | None = None,
    use_gpu: bool = False,
    cache_file: str | Path | None = None,
) -> KcatInferenceResult:
    missing_columns = _REQUIRED_INPUT_COLUMNS.difference(input_df.columns)
    if missing_columns:
        raise ValueError(
            "infer_kcats input_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    if cache_file is None:
        cache_directory = get_default_cache_dir()
    else:
        cache_directory = Path(cache_file).resolve().parent
        cache_directory.mkdir(parents=True, exist_ok=True)

    prepared_input_df = input_df[["SMILES", "sequence", "pdbpath"]].copy()
    prepared_input_df["SMILES"] = prepared_input_df["SMILES"].astype(str)
    prepared_input_df["sequence"] = prepared_input_df["sequence"].astype(str)
    prepared_input_df["pdbpath"] = prepared_input_df["pdbpath"].astype(str)

    cache = LookupCache(cache_directory, CATPRED_KCAT_CACHE_NAMESPACE)
    cache_version = KCAT_NAMESPACE_TO_SCHEMA_VERSION[CATPRED_KCAT_CACHE_NAMESPACE]

    lookup_keys: list[str] = [
        _cache_key_for_row(
            smiles=str(row.SMILES),
            sequence=str(row.sequence),
            pdbpath=str(row.pdbpath),
            checkpoint_dir=str(checkpoint_dir),
            use_gpu=bool(use_gpu),
        )
        for row in prepared_input_df.itertuples(index=False)
    ]
    deduped_lookup_keys = list(dict.fromkeys(lookup_keys))

    try:
        valid_payload: dict[str, dict[str, Any]] = {}
        keys_to_invalidate: list[str] = []
        for key in deduped_lookup_keys:
            entry = cache.get(key)
            if not isinstance(entry, dict):
                continue
            if entry.get("cache_schema_version") != cache_version:
                continue

            error_value = entry.get("error")
            if error_value in (None, "Not Found"):
                valid_payload[key] = entry
            else:
                keys_to_invalidate.append(key)

        if keys_to_invalidate:
            cache.invalidate_many(keys_to_invalidate)

        ids_to_lookup = [key for key in deduped_lookup_keys if key not in valid_payload]

        cached_map: dict[str, CatPredKcatPrediction] = {}
        for key, value in valid_payload.items():
            if not isinstance(value, dict):
                continue
            parsed = _prediction_from_payload(value)
            if parsed is not None:
                cached_map[key] = parsed

        missing_key_set = set(ids_to_lookup)
        pending_df = prepared_input_df.loc[
            [key in missing_key_set for key in lookup_keys],
            ["SMILES", "sequence", "pdbpath"],
        ].copy()

        cached_df = _cached_predictions_to_dataframe(
            lookup_rows_df=prepared_input_df,
            cache_map=cached_map,
            checkpoint_dir=str(checkpoint_dir),
            use_gpu=bool(use_gpu),
        )

        inferred_df = pd.DataFrame(
            columns=["SMILES", "sequence", "pdbpath", "Prediction_log10", "SD_total", "SD_epistemic", "SD_aleatoric"]
        )
        if not pending_df.empty:
            pending_input_path = results_path / "kcat_pending_input.csv"
            pending_df.to_csv(pending_input_path, index=False)

            request = PredictionRequest(
                parameter="kcat",
                input_file=str(pending_input_path),
                checkpoint_dir=str(checkpoint_dir),
                use_gpu=bool(use_gpu),
                repo_root=None if repo_root is None else str(repo_root),
            )
            output_path = run_inprocess_prediction_pipeline(
                request=request,
                results_dir=str(results_path),
            )
            raw_output_df = pd.read_csv(output_path)
            normalized_output_df = _normalise_output_columns(raw_output_df)

            inferred_df = pending_df.merge(
                normalized_output_df[
                    [
                        "pdbpath",
                        "Prediction_log10",
                        "SD_total",
                        "SD_epistemic",
                        "SD_aleatoric",
                    ]
                ],
                on="pdbpath",
                how="left",
            )

            for column in _CACHE_VALUE_COLUMNS:
                inferred_df[column] = pd.to_numeric(
                    inferred_df[column], errors="coerce"
                ).fillna(0.0)

            cache_updates = _inferred_predictions_to_cache_updates(
                inferred_df=inferred_df,
                checkpoint_dir=str(checkpoint_dir),
                use_gpu=bool(use_gpu),
            )
            if cache_updates:
                cache.set_many(cache_updates)

        if inferred_df.empty:
            combined_df = cached_df.copy()
        elif cached_df.empty:
            combined_df = inferred_df.copy()
        else:
            combined_df = pd.concat([cached_df, inferred_df], axis=0, ignore_index=True)
    finally:
        cache.close()

    if not pending_df.empty:
        pending_input_path = results_path / "kcat_pending_input.csv"
        if pending_input_path.exists():
            pending_input_path.unlink(missing_ok=True)

    merged_df = prepared_input_df.merge(
        combined_df[
            [
                "SMILES",
                "sequence",
                "pdbpath",
                "Prediction_log10",
                "SD_total",
                "SD_epistemic",
                "SD_aleatoric",
            ]
        ],
        on=["SMILES", "sequence", "pdbpath"],
        how="left",
    )

    for column in _CACHE_VALUE_COLUMNS:
        merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce").fillna(0.0)

    predictions_df = merged_df[
        [
            "SMILES",
            "sequence",
            "pdbpath",
            "Prediction_log10",
            "SD_total",
            "SD_epistemic",
            "SD_aleatoric",
        ]
    ].copy()

    return KcatInferenceResult(
        predictions_df=predictions_df,
        cache_file=cache.path,
        newly_inferred_count=len(inferred_df),
        cached_count=len(cached_df),
    )
