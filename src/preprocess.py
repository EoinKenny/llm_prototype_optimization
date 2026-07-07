"""Prepare the six datasets exactly as described in Appendix D.

Raw files are expected under datasets/<dataset_name>/. Outputs are written to
`datasets/preprocess/<dataset_name>/{train,test}.csv` with columns `text,label`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from src.config import DATA_DIR, DBPEDIA_TARGET_CLASSES, PREPROCESS_DIR, ensure_directories


def _find_file(directory: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return path
    lowered = {path.name.lower(): path for path in directory.glob("*") if path.is_file()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise FileNotFoundError(f"None of {candidates} found under {directory}")


def _text_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    by_lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(f"Could not identify a text column. Columns: {frame.columns.tolist()}")


def _label_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    by_lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(f"Could not identify a label column. Columns: {frame.columns.tolist()}")


def _write(dataset: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    output = PREPROCESS_DIR / dataset
    output.mkdir(parents=True, exist_ok=True)
    train = train[["text", "label"]].copy()
    test = test[["text", "label"]].copy()
    for frame in (train, test):
        frame["text"] = frame["text"].fillna("").astype(str).str.strip()
        frame.drop(frame[frame["text"] == ""].index, inplace=True)
        frame["label"] = frame["label"].astype(int)
        frame.reset_index(drop=True, inplace=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    print(f"{dataset}: wrote {len(train):,} train and {len(test):,} test examples")


def preprocess_trec() -> None:
    source = DATA_DIR / "trec"
    train = pd.read_csv(_find_file(source, ["train.csv"]))
    test = pd.read_csv(_find_file(source, ["test.csv"]))
    text_col = _text_column(train, ["text", "question"])
    label_col = _label_column(train, ["label-fine", "label_fine", "label"])
    train = train.rename(columns={text_col: "text", label_col: "label"})
    test_text_col = _text_column(test, ["text", "question"])
    test_label_col = _label_column(test, ["label-fine", "label_fine", "label"])
    test = test.rename(columns={test_text_col: "text", test_label_col: "label"})

    # Preserve integer labels where provided; otherwise create one mapping from train.
    if not pd.api.types.is_numeric_dtype(train["label"]):
        classes = sorted(train["label"].astype(str).unique())
        mapping = {label: index for index, label in enumerate(classes)}
        train["label"] = train["label"].astype(str).map(mapping)
        test["label"] = test["label"].astype(str).map(mapping)
    _write("trec", train, test)


def preprocess_dbpedia() -> None:
    source = DATA_DIR / "dbpedia"
    paths = {
        "train": _find_file(source, ["DBPEDIA_train.csv", "dbpedia_train.csv", "train.csv"]),
        "validation": _find_file(source, ["DBPEDIA_val.csv", "dbpedia_val.csv", "val.csv", "validation.csv"]),
        "test": _find_file(source, ["DBPEDIA_test.csv", "dbpedia_test.csv", "test.csv"]),
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    for name, frame in frames.items():
        if "l3" not in frame.columns:
            raise ValueError(f"DBpedia {name} file must contain an l3 column")

    # Reproduce pandas categorical codes consistently across every official split.
    all_l3 = pd.concat([frame["l3"].astype(str) for frame in frames.values()], ignore_index=True)
    categories = sorted(all_l3.unique().tolist())
    selected_names = [categories[index] for index in DBPEDIA_TARGET_CLASSES]
    final_mapping = {name: index for index, name in enumerate(sorted(selected_names))}

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        text_col = _text_column(frame, ["text", "content", "abstract"])
        output = frame[frame["l3"].astype(str).isin(selected_names)].copy()
        output["label"] = output["l3"].astype(str).map(final_mapping)
        output = output.rename(columns={text_col: "text"})
        return output[["text", "label"]]

    # The official test split remains untouched for evaluation. The supplied
    # training and validation splits are combined for model training.
    train = pd.concat([transform(frames["train"]), transform(frames["validation"])], ignore_index=True)
    test = transform(frames["test"])
    _write("dbpedia", train, test)


def preprocess_amazon_reviews() -> None:
    source = DATA_DIR / "amazon_reviews"
    json_path = _find_file(
        source,
        ["Cell_Phones_and_Accessories_5.json", "Cell_Phones_and_Accessories_5.json.gz"],
    )
    frame = pd.read_json(json_path, lines=True, compression="infer")
    review_col = _text_column(frame, ["reviewText", "review_text", "text"])
    rating_col = _label_column(frame, ["overall", "rating"])
    frame = frame[[review_col, rating_col]].rename(columns={review_col: "text", rating_col: "rating"})
    frame = frame[frame["text"].notna()].copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"] != ""]

    frame["label"] = 0
    frame.loc[frame["rating"] == 3, "label"] = 1
    frame.loc[frame["rating"] >= 4, "label"] = 2
    if len(frame) < 100_000:
        raise ValueError(f"Amazon source has only {len(frame):,} valid reviews; 100,000 are required")
    sampled = frame.sample(n=100_000, random_state=0).reset_index(drop=True)
    train = sampled.iloc[:90_000].copy()
    test = sampled.iloc[90_000:].copy()
    _write("amazon_reviews", train, test)


def preprocess_20newsgroups() -> None:
    train_raw = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
    test_raw = fetch_20newsgroups(subset="test", shuffle=True, random_state=42)
    train = pd.DataFrame({"text": train_raw.data, "label": train_raw.target})
    test = pd.DataFrame({"text": test_raw.data, "label": test_raw.target})
    _write("20newsgroups", train, test)


def preprocess_imdb() -> None:
    source = DATA_DIR / "imdb"
    train = pd.read_csv(_find_file(source, ["train.csv"]))
    test = pd.read_csv(_find_file(source, ["test.csv"]))

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        text_col = _text_column(frame, ["review", "text"])
        label_col = _label_column(frame, ["sentiment", "label"])
        output = frame.rename(columns={text_col: "text", label_col: "label"}).copy()
        if not pd.api.types.is_numeric_dtype(output["label"]):
            normalized = output["label"].astype(str).str.lower().str.strip()
            mapping = {"negative": 0, "positive": 1, "neg": 0, "pos": 1}
            output["label"] = normalized.map(mapping)
            if output["label"].isna().any():
                classes = sorted(normalized.unique().tolist())
                output["label"] = normalized.map({label: index for index, label in enumerate(classes)})
        return output

    _write("imdb", transform(train), transform(test))


def preprocess_agnews() -> None:
    source = DATA_DIR / "agnews"
    train = pd.read_csv(_find_file(source, ["train.csv"]))
    test = pd.read_csv(_find_file(source, ["test.csv"]))

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        text_col = _text_column(frame, ["Description", "text"])
        label_col = _label_column(frame, ["Class Index", "label"])
        output = frame.rename(columns={text_col: "text", label_col: "label"}).copy()
        output["label"] = output["label"].astype(int)
        if output["label"].min() == 1:
            output["label"] -= 1
        return output

    _write("agnews", transform(train), transform(test))


PROCESSORS = {
    "trec": preprocess_trec,
    "dbpedia": preprocess_dbpedia,
    "amazon_reviews": preprocess_amazon_reviews,
    "20newsgroups": preprocess_20newsgroups,
    "imdb": preprocess_imdb,
    "agnews": preprocess_agnews,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(PROCESSORS), choices=sorted(PROCESSORS))
    args = parser.parse_args()
    ensure_directories()
    for dataset in args.datasets:
        PROCESSORS[dataset]()


if __name__ == "__main__":
    main()
