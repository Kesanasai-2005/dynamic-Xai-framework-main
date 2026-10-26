"""Dataset loading and preprocessing for dynamic tabular data."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Exact id/ID/Id, or token form like loan_id / id_num (not "grid", "solid", …).
_ID_COL = re.compile(r"(^|_)id($|_)", re.IGNORECASE)


def _id_columns(columns, *, keep: str | None = None) -> list[str]:
    return [c for c in columns if c != keep and _ID_COL.search(str(c).strip())]


def drop_id_columns(
    df: pd.DataFrame, *, keep: str | None = None, verbose: bool = True
) -> pd.DataFrame:
    drop = _id_columns(df.columns, keep=keep)
    if not drop:
        return df
    if verbose:
        print(f"  dropped id columns: {', '.join(drop)}")
    return df.drop(columns=drop)


def load_dataset(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def analyze_dataset(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")
    df = drop_id_columns(df, keep=target, verbose=False)
    n_rows, n_cols = df.shape
    missing = df.isnull().sum().sum()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return {
        "rows": n_rows,
        "columns": n_cols,
        "missing_values": int(missing),
        "numeric_features": [c for c in num_cols if c != target],
        "categorical_features": [c for c in cat_cols if c != target],
        "target": target,
        "target_classes": df[target].nunique(),
        "class_balance": df[target].value_counts(normalize=True).to_dict(),
    }


def _split(X, y, test_size, random_state):
    """Stratified split that still works when test_size * n < n_classes.

    ponytail: bumps the test count up to one row per class. Needs >= 2 rows
    per class; otherwise the split is unstratified. Upgrade path is a
    user-facing warning when the configured fraction is ignored.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)
    if n < 2:
        raise ValueError(f"Need at least 2 rows to train and test; got {n}.")

    classes, counts = np.unique(y_arr, return_counts=True)
    n_classes = len(classes)
    if isinstance(test_size, (int, np.integer)):
        requested = int(test_size)
    else:
        requested = int(np.ceil(float(test_size) * n))
    requested = min(max(requested, 1), n - 1)

    can_stratify = int(counts.min()) >= 2 and n >= 2 * n_classes
    if can_stratify:
        n_test = min(max(requested, n_classes), n - n_classes)
        stratify = y
    else:
        n_test = requested
        stratify = None

    if n_test != requested:
        print(f"  test rows raised from {requested} to {n_test} so each class is in the test set")
    elif stratify is None:
        print(f"  unstratified split ({n} rows, {n_classes} classes)")

    return train_test_split(
        X, y, test_size=n_test, random_state=random_state, stratify=stratify
    )


def validate_test(train: pd.DataFrame, test: pd.DataFrame, target: str) -> list[str]:
    """Problems that stop `test` being scored by models trained on `train`. Empty = OK.

    Every train column must appear in test, except the user-selected `target`
    (test may be feature-only). Test must not introduce columns absent from train.
    """
    if test.empty:
        return ["Test dataset has no rows."]
    errors = []
    # Match all headers; only the selected target may be missing from test.
    missing = [c for c in train.columns if c != target and c not in test.columns]
    extra = [c for c in test.columns if c not in train.columns]
    if missing:
        errors.append(f"Test dataset is missing columns: {', '.join(missing)}")
    if extra:
        errors.append(f"Test dataset has columns not in train: {', '.join(extra)}")
    for c in train.columns.intersection(test.columns):
        if c == target:
            continue
        tr_num = pd.api.types.is_numeric_dtype(train[c])
        te_num = pd.api.types.is_numeric_dtype(test[c])
        if tr_num and not te_num and test[c].notna().any():
            errors.append(f"Column '{c}' is numeric in train but contains text in test.")
    if target in test.columns and target in train.columns:
        unseen = set(test[target].dropna().astype(str)) - set(train[target].dropna().astype(str))
        if unseen:
            errors.append(f"Test target has labels never seen in train: {', '.join(sorted(unseen))}")
    return errors


def preprocess(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    handle_missing: str = "median",
    test_df: pd.DataFrame | None = None,
):
    """Impute, encode, and scale. Uses `test_df` as the test set if given, else splits `df`."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")
    df = drop_id_columns(df, keep=target)
    has_test_target = False
    if test_df is not None:
        test_df = drop_id_columns(test_df, keep=target)
        errors = validate_test(df, test_df, target)
        if errors:
            raise ValueError(" ".join(errors))
        has_test_target = target in test_df.columns
        features = [c for c in df.columns if c != target]
        test_df = test_df[df.columns if has_test_target else features]

    X = df.drop(columns=[target])
    y = df[target]

    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))
    else:
        y = np.asarray(y)

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy=handle_missing)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    if test_df is None:
        X_train, X_test, y_train, y_test = _split(X, y, test_size, random_state)
    elif has_test_target:
        test_df = test_df.dropna(subset=[target])
        X_train, y_train = X, y
        X_test = test_df.drop(columns=[target])
        y_test = (
            label_encoder.transform(test_df[target].astype(str))
            if label_encoder is not None
            else np.asarray(test_df[target])
        )
    else:
        X_train, y_train = X, y
        X_test = test_df
        y_test = None

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())

    return {
        "X_train": X_train_t,
        "X_test": X_test_t,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
    }


if __name__ == "__main__":
    tiny = pd.DataFrame({"a": [1, 2, 3, 4, 5], "y": ["R", "A", "A", "R", "A"]})
    out = preprocess(tiny, "y", test_size=0.2)
    assert len(out["y_test"]) >= 2 and len(out["y_train"]) >= 2
    big = pd.DataFrame({"a": range(20), "y": [0, 1] * 10})
    out = preprocess(big, "y", test_size=0.2)
    assert len(out["y_test"]) == 4
    test = pd.DataFrame({"y": ["A", "R"], "a": [9, 1]})
    out = preprocess(tiny, "y", test_df=test)
    assert len(out["y_train"]) == 5 and list(out["y_test"]) == [0, 1]
    assert validate_test(tiny, test.rename(columns={"a": "b"}), "y")
    assert validate_test(tiny, pd.DataFrame({"a": [1], "y": ["Z"]}), "y")
    assert not validate_test(tiny, pd.DataFrame({"a": [1]}), "y")
    out = preprocess(tiny, "y", test_df=pd.DataFrame({"a": [9, 1]}))
    assert out["y_test"] is None and len(out["X_test"]) == 2
    with_id = pd.DataFrame(
        {"ID": [1, 2, 3, 4], "loan_id": [10, 11, 12, 13], "a": [1, 2, 3, 4], "y": [0, 1, 0, 1]}
    )
    out = preprocess(with_id, "y", test_size=0.5)
    assert "ID" not in out["X_train_raw"].columns
    assert "loan_id" not in out["X_train_raw"].columns
    assert "a" in out["X_train_raw"].columns
    print("ok")
