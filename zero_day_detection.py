import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM

#column names of the NSL-KDD data set
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]
#broader classes
ATTACK_GROUPS = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "mailbomb": "dos", "apache2": "dos", "processtable": "dos",
    "udpstorm": "dos", "worm": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

#dataset loading and giving names for columns.
def load_nsl_kdd(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=NSL_KDD_COLUMNS)

    #clean the labeled numbers
    df["label"] = df["label"].astype(str).str.strip()

     # Add new column to the grouped attack categories.
    df["attack_group"] = df["label"].map(lambda x: ATTACK_GROUPS.get(x, "unknown"))
    return df

    #splitting
def build_zero_day_split(train_df, test_df, held_out_attack="probe", val_frac=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    train_normal = train_df[train_df["attack_group"] == "normal"].copy()
    
    # Validation
    known_attacks_for_validation = train_df[
        (train_df["attack_group"] != "normal") & (train_df["attack_group"] != held_out_attack) & (train_df["attack_group"] != "unknown")].copy()
    
    #samples to validate.
    n_val = max(1, int(len(train_normal) * val_frac))
    val_idx = rng.choice(train_normal.index.to_numpy(), size=n_val, replace=False)
    val_normal = train_normal.loc[val_idx].copy()

    #training normal values.
    train_normal_final = train_normal.drop(index=val_idx).copy()

    # Balance validation 
    if len(known_attacks_for_validation) > len(val_normal):
        known_attacks_for_validation = known_attacks_for_validation.sample(
            n=len(val_normal), random_state=random_state
        )
    # Final validation set
    val_df = pd.concat([val_normal, known_attacks_for_validation], axis=0).sample(
        frac=1.0, random_state=random_state
    )

    test_normal = test_df[test_df["attack_group"] == "normal"].copy()
    test_zero_day = test_df[test_df["attack_group"] == held_out_attack].copy()

    test_final = pd.concat([test_normal, test_zero_day], axis=0).sample(
        frac=1.0, random_state=random_state
    )

    return train_normal_final, val_df, test_final


def get_xy(df):
    # Attack = 1, normal = 0
    y = (df["attack_group"] != "normal").astype(int).to_numpy()

    # Removing the non feature columns
    X = df.drop(columns=["label", "difficulty", "attack_group"]).copy()
    return X, y


def preprocess(train_X, val_X, test_X):
    # Categorical columns need encoding
    categorical_cols = ["protocol_type", "service", "flag"]

    numeric_cols = [c for c in train_X.columns if c not in categorical_cols]

    # Scaling 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train = preprocessor.fit_transform(train_X)
    X_val = preprocessor.transform(val_X)
    X_test = preprocessor.transform(test_X)
    return X_train, X_val, X_test


def find_best_threshold(y_true, scores):
    # Obtain precision,recall based thresholds.
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # Fallback.
    if len(thresholds) == 0:
        thr = float(np.median(scores))
        preds = (scores >= thr).astype(int)
        return thr, f1_score(y_true, preds, zero_division=0)

    #Select threshold with optimal F1 score.
    best_thr = thresholds[0]
    best_f1 = -1.0
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr), float(best_f1)


def evaluate(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    # Returning function
    return {
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    # Read cmd args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--held_out_attack", default="probe", choices=["dos", "probe", "r2l", "u2r"])
    parser.add_argument("--output_dir", default="outputs_zero_day")
    args = parser.parse_args()

    # Creating the output folder
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = load_nsl_kdd(args.train)
    test_df = load_nsl_kdd(args.test)

    print(f"Creating zero-day setup with held-out attack = {args.held_out_attack}")
    train_normal, val_df, test_final = build_zero_day_split(
        train_df, test_df, held_out_attack=args.held_out_attack
    )

    print(f"Train normal: {len(train_normal)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_final)}")

    # Prepare features and labels
    train_X, train_y = get_xy(train_normal)
    val_X, val_y = get_xy(val_df)
    test_X, test_y = get_xy(test_final)

    # Apply preprocessing
    X_train, X_val, X_test = preprocess(train_X, val_X, test_X)

    results = []

    print("Training Isolation Forest !!!")
    # Train Isolation Forest model
    iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    iforest.fit(X_train)

    # Computing the anomaly scores
    val_scores_if = -iforest.score_samples(X_val)
    test_scores_if = -iforest.score_samples(X_test)

    thr_if, _ = find_best_threshold(val_y, val_scores_if)
    metrics_if = evaluate(test_y, test_scores_if, thr_if)
    metrics_if["model"] = "Isolation Forest"
    results.append(metrics_if)

    print("Training One-Class SVM !!!")
    # Training One Class SVM model
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_train)

    #anomaly scores
    val_scores_svm = -ocsvm.score_samples(X_val)
    test_scores_svm = -ocsvm.score_samples(X_test)
    thr_svm, _ = find_best_threshold(val_y, val_scores_svm)
    metrics_svm = evaluate(test_y, test_scores_svm, thr_svm)
    metrics_svm["model"] = "One-Class SVM"
    results.append(metrics_svm)

    #results table
    results_df = pd.DataFrame(results)
    results_df = results_df[[
        "model", "precision", "recall", "f1_score", "pr_auc",
        "false_positive_rate", "threshold", "tn", "fp", "fn", "tp"
    ]]

    # Save CSV file
    results_df.to_csv(out_dir / "metrics.csv", index=False)

    # Save JSON file
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save summary information
    summary = {
        "held_out_attack": args.held_out_attack,
        "train_normal_samples": int(len(train_normal)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_final)),
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nRUN COMPLETE")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()