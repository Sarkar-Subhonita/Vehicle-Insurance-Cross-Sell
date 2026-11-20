"""Run the vehicle insurance monitoring and retraining practical from the repo root."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from src.entity.estimator import MyModel
from src.utils.main_utils import load_object, read_yaml_file, save_object


SEED = 101
ID_COLUMN = "id"
CATEGORIES = {
    "Gender": ["Female", "Male"],
    "Vehicle_Age": ["1-2 Year", "< 1 Year", "> 2 Years"],
    "Vehicle_Damage": ["No", "Yes"],
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def validate_batch(frame, required_columns):
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    frame = frame.loc[:, required_columns]
    summary = {
        "rows": len(frame),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_records": int(frame.duplicated().sum()),
        "duplicate_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "target_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    if frame.empty or any(summary[key] for key in (
        "missing_values", "duplicate_records", "duplicate_ids"
    )):
        raise ValueError("Use a non-empty batch with no missing values or duplicates.")
    if set(frame[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Each input dataset needs both Response labels, 0 and 1.")
    for column, categories in CATEGORIES.items():
        if not frame[column].isin(categories).all():
            raise ValueError(f"Unexpected category in {column}. Expected {categories}.")
    return frame, summary


def encode_features(frame, transformation):
    # Fix categorical levels so even a small batch produces all model columns.
    features = frame.drop(columns=[TARGET_COLUMN]).copy()
    for column in ("Vehicle_Age", "Vehicle_Damage"):
        features[column] = pd.Categorical(features[column], categories=CATEGORIES[column])
    features = transformation._map_gender_column(features)
    features = transformation._create_dummy_columns(features)
    return transformation._rename_columns(features)


def fit_insurance_model(features, labels, transformation):
    preprocessor = transformation.get_data_transformer_object()
    transformed = preprocessor.fit_transform(features.drop(columns=[ID_COLUMN]))
    if labels.value_counts().min() < 6:
        raise ValueError("At least six examples per class are needed in training for SMOTEENN.")
    sampled_features, sampled_labels = SMOTEENN(
        sampling_strategy="minority", random_state=SEED
    ).fit_resample(transformed, labels)

    config = ModelTrainerConfig()
    model = RandomForestClassifier(
        n_estimators=config._n_estimators,
        min_samples_split=config._min_samples_split,
        min_samples_leaf=config._min_samples_leaf,
        max_depth=config._max_depth,
        criterion=config._criterion,
        random_state=config._random_state,
        n_jobs=2,
    )
    model.fit(sampled_features, sampled_labels)
    return MyModel(preprocessing_object=preprocessor, trained_model_object=model)


def evaluate(model, features, labels):
    # The original project's saved preprocessor may also include id.
    columns = list(model.preprocessing_object.feature_names_in_)
    predicted = model.predict(features.loc[:, columns])
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--new-batch", type=Path)
    parser.add_argument("--champion", type=Path, help="Previously saved, trusted project model.pkl")
    parser.add_argument("--sample-size", type=int, default=12000, help="Demo mode row count")
    args = parser.parse_args()
    paths = (args.historical, args.new_batch, args.champion)
    if any(paths) and not all(paths):
        parser.error("For a real batch, provide --historical, --new-batch and --champion.")

    schema = read_yaml_file(SCHEMA_FILE_PATH)
    required_columns = [next(iter(entry)) for entry in schema["columns"]]
    demo = args.champion is None

    print("Task A - Validate and prepare the incoming batch", flush=True)
    if demo:
        original = pd.read_csv("notebook/data.csv")
        if not 200 <= args.sample_size <= len(original):
            parser.error(f"--sample-size must be between 200 and {len(original)}.")
        sample = original.sample(n=args.sample_size, random_state=SEED)
        historical, incoming = train_test_split(
            sample, test_size=0.30, random_state=SEED, stratify=sample[TARGET_COLUMN]
        )
        print("Mode: simulated incoming batch from notebook/data.csv.", flush=True)
    else:
        historical = pd.read_csv(args.historical)
        incoming = pd.read_csv(args.new_batch)
        print("Mode: supplied incoming batch and existing champion.", flush=True)

    incoming, incoming_summary = validate_batch(incoming, required_columns)
    print("Historical data validation:", flush=True)
    historical, _ = validate_batch(historical, required_columns)
    if incoming[ID_COLUMN].isin(historical[ID_COLUMN]).any():
        raise ValueError("Incoming and historical IDs overlap. Use unseen incoming rows.")

    update_rows, evaluation_rows = train_test_split(
        incoming, test_size=0.50, random_state=SEED, stratify=incoming[TARGET_COLUMN]
    )
    output = Path("assessment_output") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    # Reuse the repo's data transformation and model wrapper, while keeping
    # resampling out of evaluation.
    transformation = DataTransformation(
        data_ingestion_artifact=None,
        data_transformation_config=DataTransformationConfig(),
        data_validation_artifact=None,
    )
    evaluation_x = encode_features(evaluation_rows, transformation)
    evaluation_y = evaluation_rows[TARGET_COLUMN]

    print("Task B - Score the existing model", flush=True)
    if demo:
        champion_path = output / "demo_champion.pkl"
        demo_champion = fit_insurance_model(
            encode_features(historical, transformation), historical[TARGET_COLUMN], transformation
        )
        save_object(str(champion_path), demo_champion)
    else:
        champion_path = args.champion

    champion_metrics = evaluate(load_object(str(champion_path)), evaluation_x, evaluation_y)

    print("Task C - Train and save a separate candidate", flush=True)
    combined = pd.concat([historical, update_rows], ignore_index=True)
    candidate_path = output / "candidate_model.pkl"
    candidate = fit_insurance_model(
        encode_features(combined, transformation), combined[TARGET_COLUMN], transformation
    )
    save_object(str(candidate_path), candidate)

    print("Task D - Compare models on the same untouched evaluation rows", flush=True)
    candidate_metrics = evaluate(load_object(str(candidate_path)), evaluation_x, evaluation_y)
    report = pd.DataFrame([
        {"model": "champion", **champion_metrics},
        {"model": "candidate", **candidate_metrics},
    ])
    report.to_csv(output / "evaluation_report.csv", index=False)

    eligible = candidate_metrics["f1"] > champion_metrics["f1"]
    decision = "Eligible for promotion" if eligible else "Retain champion"
    write_json(output / "quality_gate.json", {
        "mode": "simulated incoming batch" if demo else "supplied incoming batch",
        "data_validation": incoming_summary,
        "historical_rows": len(historical),
        "new_training_rows": len(update_rows),
        "evaluation_rows": len(evaluation_rows),
        "metric": "f1",
        "metric_reason": "F1 balances precision and recall for the minority interested class.",
        "rule": "Candidate F1 must be strictly higher than champion F1 on the same rows.",
        "champion_f1": champion_metrics["f1"],
        "candidate_f1": candidate_metrics["f1"],
        "decision": decision,
        "champion_overwritten": False,
    })
    print(report.to_string(index=False))
    print(f"Decision: {decision}")
    print(f"Task E - Demo files saved in: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Assessment could not finish: {error}")
