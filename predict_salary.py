# Course: CS 513 - Data Analytics & Machine Learning
# Purpose: Predict Premier League player salary using Random Forest Regression

import pandas as pd
import numpy as np
import os
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Load datasets
# -----------------------------

# Update these paths after downloading from Kaggle
salary_path = "data/player_salaries.csv"
data_dir = "data"

players_df = pd.read_csv(salary_path)

print("Players dataset columns:")
print(players_df.columns)


def _strip_accents(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _norm_key(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .map(_strip_accents)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _standardize_stats_df(df_in: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Standardize a stats dataframe to common join keys and prefix feature columns
    with the source_name to avoid collisions across files.
    """
    df = df_in.copy()

    # Map common key columns to a shared schema.
    rename_map = {}
    if "Player" in df.columns:
        rename_map["Player"] = "player_name"
    if "player" in df.columns:
        rename_map["player"] = "player_name"
    if "name" in df.columns:
        rename_map["name"] = "player_name"

    if "Squad" in df.columns:
        rename_map["Squad"] = "team"
    if "Team" in df.columns:
        rename_map["Team"] = "team"
    if "team" in df.columns:
        rename_map["team"] = "team"

    if "Pos" in df.columns:
        rename_map["Pos"] = "position"
    if "Position" in df.columns:
        rename_map["Position"] = "position"
    if "position" in df.columns:
        rename_map["position"] = "position"

    if "Age" in df.columns:
        rename_map["Age"] = "age"
    if "age" in df.columns:
        rename_map["age"] = "age"

    df = df.rename(columns=rename_map)

    # Require at least player + team to merge reliably.
    missing = [c for c in ["player_name", "team"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stats file '{source_name}' missing required merge columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["player_name"] = _norm_key(df["player_name"])
    df["team"] = _norm_key(df["team"])
    if "position" in df.columns:
        df["position"] = _norm_key(df["position"])

    # Drop obvious non-feature columns that cause noise or duplicates.
    for col in ["Rk", "Matches"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Deduplicate: if any file has multiple rows per player/team, keep the first.
    key_cols = ["player_name", "team"]
    if "position" in df.columns:
        key_cols.append("position")
    if "age" in df.columns:
        key_cols.append("age")
    df = df.drop_duplicates(subset=key_cols, keep="first")

    # Prefix all non-key columns with the source name to prevent collisions,
    # except for the key columns themselves.
    key_set = set(["player_name", "team", "position", "age"])
    prefixed_cols = {}
    for col in df.columns:
        if col in key_set:
            continue
        prefixed_cols[col] = f"{source_name}__{col}"
    df = df.rename(columns=prefixed_cols)

    return df


def load_and_merge_all_stats(data_directory: str) -> pd.DataFrame:
    """
    Load every CSV in data_directory except the salaries file, standardize columns,
    and merge into one wide stats table keyed by player_name + team (+ position/age when present).
    """
    csv_files = [
        f for f in os.listdir(data_directory)
        if f.lower().endswith(".csv") and f.lower() != "player_salaries.csv"
    ]
    if not csv_files:
        raise ValueError(f"No stats CSV files found in '{data_directory}'.")

    merged = None
    for fname in sorted(csv_files):
        path = os.path.join(data_directory, fname)
        source = os.path.splitext(fname)[0]
        df_raw = pd.read_csv(path)
        df_std = _standardize_stats_df(df_raw, source)

        if merged is None:
            merged = df_std
        else:
            # Merge on the intersection of join keys available in both.
            join_keys = [c for c in ["player_name", "team", "position", "age"] if c in merged.columns and c in df_std.columns]
            if "player_name" not in join_keys or "team" not in join_keys:
                join_keys = ["player_name", "team"]
            merged = merged.merge(df_std, on=join_keys, how="outer")

        print(f"\nLoaded stats file: {fname}")
        print(f"  rows={len(df_std):,} cols={len(df_std.columns):,}")

    return merged


# -----------------------------
# 2. Basic cleaning helpers
# -----------------------------

def clean_money_column(series):
    """
    Converts salary strings like '£100,000', '$2,500,000', or '100,000'
    into numeric values.
    """
    return (
        series.astype(str)
        .str.replace(r"[£$,]", "", regex=True)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
        .astype(float)
    )


def clean_percent_column(series):
    """
    Converts percentage strings like '72.5%' into 72.5.
    """
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
        .astype(float)
    )


# -----------------------------
# 3. Choose target column
# -----------------------------

# Prefer Annual salary when available, otherwise fall back.
target_col = "Annual" if "Annual" in players_df.columns else ("Weekly" if "Weekly" in players_df.columns else "Salary")

if target_col not in players_df.columns:
    raise ValueError(f"Could not find target column '{target_col}'. Check printed columns above.")

if not np.issubdtype(players_df[target_col].dtype, np.number):
    players_df[target_col] = clean_money_column(players_df[target_col])


# -----------------------------
# 4. Optional merge step
# -----------------------------
# Merge salaries with all available stats files in data/.

df = players_df.copy()
df = df.rename(columns={"Player": "player_name", "Team": "team", "Position": "position", "Age": "age"})

for col in ["player_name", "team", "position"]:
    if col in df.columns:
        df[col] = _norm_key(df[col])

stats_merged_df = load_and_merge_all_stats(data_dir)

# Merge on the richest common key set available to reduce duplicate columns.
# IMPORTANT: don't merge on `position` because position labels differ across sources
# (e.g., salaries may have CB/LB/RW while stats may use DF/MF/FW). Merging on position
# would drop most stats for those players.
merge_keys = ["player_name", "team"]
if "age" in df.columns and "age" in stats_merged_df.columns:
    merge_keys.append("age")

df = df.merge(stats_merged_df, on=merge_keys, how="left")

# If both sides had `position`, pandas will suffix them; keep the salary-file position.
if "position" not in df.columns and "position_x" in df.columns:
    df["position"] = df["position_x"]
    drop_cols = [c for c in ["position_x", "position_y"] if c in df.columns]
    df = df.drop(columns=drop_cols)


# -----------------------------
# 5. Drop rows without salary
# -----------------------------

df = df.dropna(subset=[target_col])

# Remove goalkeepers: their stats/role differ heavily and are sparse in our features.
if "position" in df.columns:
    df = df[~df["position"].astype(str).str.contains(r"\bGK\b", regex=True, na=False)].copy()


# -----------------------------
# 6. Select features
# -----------------------------
def _coalesce_first(df_in: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    """Create/overwrite out_col as first non-null across candidates (in order)."""
    available = [c for c in candidates if c in df_in.columns]
    if not available:
        return
    s = df_in[available[0]]
    for c in available[1:]:
        s = s.combine_first(df_in[c])
    df_in[out_col] = s


# ---- Feature rules (project requirements) ----
# - Predict Annual salary using performance stats
# - Weekly MUST NOT be a feature
# - Only categorical feature allowed: Position
# - Remove born columns (we already have age)
# - Remove team/nation from features
# - Remove duplicate stats across files; keep one unified column

# Drop disallowed/non-feature columns early.
drop_if_present = [
    "Weekly",  # MUST NOT be a feature
    "team",  # not relevant
    "Nation",  # not relevant
]

# Any born columns from merged stats sources should be excluded.
drop_if_present += [c for c in df.columns if c.lower().endswith("__born") or c.lower() == "born"]

for c in drop_if_present:
    if c in df.columns:
        df = df.drop(columns=[c])

# Build a deduplicated, readable performance feature set.
# Preference order: use `player_stats.csv` columns when available, otherwise fallback to the other sources.
_coalesce_first(df, "minutes", ["player_stats__minutes", "Squad_PlayerStats__stats_standard__Playing Time_Min"])
_coalesce_first(df, "starts", ["player_stats__starts", "Squad_PlayerStats__stats_standard__Playing Time_Starts"])
_coalesce_first(df, "played", ["player_stats__played", "Squad_PlayerStats__stats_standard__Playing Time_MP"])
_coalesce_first(df, "goals", ["player_stats__goals", "Squad_PlayerStats__stats_standard__Performance_Gls"])
_coalesce_first(df, "assists", ["player_stats__assists", "Squad_PlayerStats__stats_standard__Performance_Ast"])
_coalesce_first(df, "yellow", ["player_stats__yellow", "Squad_PlayerStats__stats_standard__Performance_CrdY"])
_coalesce_first(df, "red", ["player_stats__red", "Squad_PlayerStats__stats_standard__Performance_CrdR"])
_coalesce_first(df, "expected_goals", ["player_stats__expected_goals"])
_coalesce_first(df, "penalty_kicks", ["player_stats__penalty_kicks", "Squad_PlayerStats__stats_standard__Performance_PK"])
_coalesce_first(df, "penalty_kick_attempts", ["player_stats__penalty_kick_attempts", "Squad_PlayerStats__stats_standard__Performance_PKatt"])
_coalesce_first(df, "progressive_carries", ["player_stats__progressive_carries"])
_coalesce_first(df, "progressive_passes", ["player_stats__progressive_passes"])
_coalesce_first(df, "received_progressive_passes", ["player_stats__received_progressive_passes"])

# Keep possession metrics (not duplicates of the above in this dataset).
possession_keep = [
    "player_possession_stats__90s",
    "player_possession_stats__touches",
    "player_possession_stats__deffensive_touches",
    "player_possession_stats__middle_touches",
    "player_possession_stats__attacking_touches",
    "player_possession_stats__attempted_take_ons",
    "player_possession_stats__successful_take_ons",
    "player_possession_stats__takeons_tackled",
    "player_possession_stats__carries",
    "player_possession_stats__total_distance_carried",
    "player_possession_stats__received",
]
possession_keep = [c for c in possession_keep if c in df.columns]

# Position is the only categorical feature we want.
if "position" not in df.columns:
    raise ValueError("Expected a 'position' column after merging, but none was found.")

# Create a clean, readable X with no file prefixes in column names.
base_numeric = [
    "age",
    "minutes",
    "starts",
    "played",
    "goals",
    "assists",
    "yellow",
    "red",
    "expected_goals",
    "penalty_kicks",
    "penalty_kick_attempts",
    "progressive_carries",
    "progressive_passes",
    "received_progressive_passes",
]
numeric_cols = [c for c in base_numeric if c in df.columns]

rename_map = {c: c.split("__", 1)[-1] for c in possession_keep}
X = df[["position"] + numeric_cols + possession_keep].rename(columns=rename_map)
y = df[target_col]

print("\nUsing features:")
print("  Categorical:", ["position"])
print("  Numeric:", [c for c in X.columns if c != "position"])


# -----------------------------
# 7. Identify numeric/categorical columns
# -----------------------------

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = ["position"]

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# -----------------------------
# 8. Preprocessing
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ]
)


# -----------------------------
# 9. Build Random Forest pipeline
# -----------------------------

rf_regressor = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_regressor)
    ]
)


# -----------------------------
# 10. Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 11. Train model
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# 12. Evaluate model
# -----------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Regressor Results")
print("--------------------------------")
print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")

# MAE by position (global model)
if "position" in X_test.columns:
    print("\nMAE by position (global model):")
    for pos in sorted(X_test["position"].dropna().unique()):
        mask = X_test["position"] == pos
        if mask.sum() < 3:
            continue
        pos_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        print(f"  {pos}: {pos_mae:,.2f} (n={int(mask.sum())})")


# -----------------------------
# 13. Compare actual vs predicted
# -----------------------------

results_df = pd.DataFrame({
    "Player": df.loc[y_test.index, "player_name"].values if "player_name" in df.columns else y_test.index,
    "Position": X_test["position"].values if "position" in X_test.columns else None,
    "Actual Salary": y_test,
    "Predicted Salary": y_pred,
    "Error": y_test - y_pred
})

print("\nSample predictions:")
print(results_df.head(10).to_string(index=False))


# -----------------------------
# 14. Feature importance
# -----------------------------

trained_preprocessor = model.named_steps["preprocessor"]
trained_rf = model.named_steps["regressor"]

feature_names = trained_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": trained_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 20 Feature Importances:")
print(importance_df.head(20))


# -----------------------------
# 15. Train / evaluate models by position
# -----------------------------
def train_and_report_by_position(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    positions: list[str],
    random_state: int = 42,
    test_size: float = 0.2,
) -> None:
    """
    Fit one model per position using only numeric stats (position is constant so it's excluded),
    then print MAE + top importances per position.
    """
    if "position" not in X_all.columns:
        return

    numeric_cols_local = [c for c in X_all.columns if c != "position"]
    for pos in positions:
        Xp = X_all[X_all["position"] == pos].copy()
        yp = y_all.loc[Xp.index]

        if len(Xp) < 30:
            continue

        Xp = Xp[numeric_cols_local]
        # Drop columns that are entirely missing within this position slice.
        Xp = Xp.dropna(axis=1, how="all")

        num_feats = Xp.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(num_feats) == 0:
            continue
        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    num_feats,
                ),
            ]
        )

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

        m = Pipeline(steps=[("preprocessor", pre), ("regressor", rf)])

        Xtr, Xte, ytr, yte = train_test_split(
            Xp, yp, test_size=test_size, random_state=random_state
        )
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        mae_pos = mean_absolute_error(yte, pred)

        # Feature importances for this position-specific model (numeric only).
        pre_trained = m.named_steps["preprocessor"]
        rf_trained = m.named_steps["regressor"]
        feat_names = pre_trained.get_feature_names_out()
        imp = (
            pd.DataFrame({"Feature": feat_names, "Importance": rf_trained.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(10)
        )

        print(f"\nPosition model: {pos}")
        print(f"  MAE: {mae_pos:,.2f} (n={len(Xp):,})")
        print("  Top 10 importances:")
        print(imp.to_string(index=False))


pos_list = sorted([p for p in X["position"].dropna().unique() if p != "GK"])
print("\nTraining separate models by position (DF/MF/FW groupings as-is in data):")
train_and_report_by_position(X, y, pos_list)