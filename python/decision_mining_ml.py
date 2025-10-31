import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree as sktree
import matplotlib.pyplot as plt
from sklearn.tree import export_text, export_graphviz, plot_tree
import graphviz
import yaml
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import scipy.sparse

class DecisionMiningML:
    """Train decision trees per decision point from CSVs.

    Inputs/outputs:
    - Input: path to a CSV with columns including 'decision_point', 'activity' and other features.
    - Output: per-decision-point trained models and evaluation metrics.

    Error modes: raises FileNotFoundError for missing files, ValueError for missing required columns.
    """

    def __init__(self, csv_path: str, model_dir: str = "models", config_path: str = "config.yaml"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self.csv_path = csv_path
        # try to auto-detect delimiter; fall back to default comma
        try:
            self.df = pd.read_csv(csv_path, sep=None, engine='python')
        except Exception:
            self.df = pd.read_csv(csv_path)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        tree_config = self.config.get("decision_tree", {})

        # Extract pruning-related params
        self.prune_depth = tree_config.pop('prune_depth', 3)
        self.prune_min_samples_leaf = tree_config.pop('prune_min_samples_leaf', 10)

        self.tree_params = tree_config

        self.log_path = self.config.get('log_path')
        if not self.log_path:
            raise ValueError("log_path not found in config.yaml")

        # prepare model_dir based on log_path
        log_basename = os.path.basename(self.log_path)        # "BPI Challenge 2017.xes"
        log_stem = os.path.splitext(log_basename)[0]          # "BPI Challenge 2017"
        safe_stem = log_stem.replace(' ', '_')                # "BPI_Challenge_2017"
        self.model_dir = os.path.join('models', safe_stem)
        os.makedirs(self.model_dir, exist_ok=True)

        # drop unnamed index columns that often appear when reading CSVs
        unnamed_cols = [c for c in self.df.columns if str(c).lower().startswith('unnamed') or str(c) == 'Unnamed: 0']
        if unnamed_cols:
            self.df = self.df.drop(columns=unnamed_cols)


    def _validate(self):
        if 'decision_point' not in self.df.columns or 'activity' not in self.df.columns:
            raise ValueError("CSV must contain 'decision_point' and 'activity' columns")

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Prepare X, y from the decision-slice DataFrame.

        Strategy:
        - Drop identifier-like columns (decision_point, activity) from X.
        - For remaining non-numeric columns, do one-hot encoding (pd.get_dummies).
        - Label y is the chosen activity (string) encoded as integer labels returned via mapping.
        Returns: X, y, meta where meta contains label mapping and feature names.
        """
        df = df.copy()
        y_raw = df['activity'].astype(str)
        # drop columns that shouldn't be features
        drop_cols = ['decision_point', 'activity']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # --- Timestamp handling: extract hour, weekday, and elapsed time per case ---
        # find timestamp columns by name
        timestamp_cols = [c for c in X.columns if 'timestamp' in str(c).lower() or 'time' in str(c).lower()]
        # try parsing and extracting features
        # detect case id column for elapsed computation
        case_cols = [c for c in X.columns if c.lower() in ('case_id', 'case:concept:name', 'concept:name')]
        case_col = case_cols[0] if case_cols else None
        for tcol in timestamp_cols:
            try:
                parsed = pd.to_datetime(X[tcol], errors='coerce')
                if parsed.notna().any():
                    X[f"{tcol}_hour"] = parsed.dt.hour.fillna(-1).astype(int)
                    X[f"{tcol}_weekday"] = parsed.dt.weekday.fillna(-1).astype(int)
                    # elapsed seconds since case start when case id exists
                    if case_col:
                        # compute case-level min timestamp from the whole dataframe (not just group)
                        # we use df (the group) to compute relative time inside this slice
                        case_min = parsed.groupby(X[case_col]).transform('min')
                        elapsed = (parsed - case_min).dt.total_seconds().fillna(0)
                        X[f"{tcol}_elapsed_seconds"] = elapsed.astype(float)
                # drop original timestamp column
                X = X.drop(columns=[tcol])
            except Exception:
                # leave as-is if parsing fails
                pass

        # remove identifier-like columns from features (don't one-hot high-cardinality IDs)
        id_like = []
        for c in X.columns:
            if c.lower() in ('case_id', 'case:concept:name', 'concept:name'):
                id_like.append(c)
        if id_like:
            # keep case_id for elapsed computation but drop from features
            X = X.drop(columns=id_like)

        # If no other columns exist, create synthetic count-based features if present activity columns exist
        if X.shape[1] == 0:
            # try selecting columns that look like activity counts (other columns with numeric values)
            # fallback: use a single constant feature
            X = pd.DataFrame({'const': np.ones(len(df))})

        # Convert object columns to dummies, but drop very high-cardinality columns
        # identify object columns
        obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for c in obj_cols:
            nunique = X[c].nunique(dropna=False)
            if nunique > max(20, 0.5 * len(X)):
                # too many unique values -> drop to avoid overfitting (e.g., raw timestamps)
                X = X.drop(columns=[c])

        X = pd.get_dummies(X, sparse=True, drop_first=True, dtype=np.float32)
        X_sparse = scipy.sparse.csr_matrix(X.values)

        # Align types
        X_vals = X_sparse
        # Encode y
        labels = sorted(y_raw.unique())
        label_map = {lab: i for i, lab in enumerate(labels)}
        y = y_raw.map(label_map).values

        meta = {'label_map': label_map, 'feature_names': list(X.columns)}
        return X_vals, y, meta

    def train_for_all(self, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """Train two trees per decision_point: a default tree and an aggressively pruned tree.

        Returns a dict keyed by decision_point with training/eval info and model paths.
        """
        self._validate()

        results = {}

        def _train_single_decision_point(dp, group):
            entry = {}
            warnings = []
            try:
                X, y, meta = self._prepare_features(group)
            except Exception as e:
                warnings.append(f"feature_prep_error: {e}")
                return dp, {'error': str(e)}

            if len(np.unique(y)) < 2:
                return dp, {'error': 'Only one activity present, cannot train classifier'}

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )

            clf = DecisionTreeClassifier(**self.tree_params)
            clf.fit(X_train, y_train)

            pruned_params = self.tree_params.copy()
            pruned_params.update({
                "max_depth": self.prune_depth,
                "min_samples_leaf": self.prune_min_samples_leaf,
            })
            clf_pruned = DecisionTreeClassifier(**pruned_params)
            clf_pruned.fit(X_train, y_train)

            def eval_model(m, Xs, ys):
                ypred = m.predict(Xs)
                return {
                    'accuracy': float(accuracy_score(ys, ypred)),
                    'precision_macro': float(precision_score(ys, ypred, average='macro', zero_division=0)),
                    'recall_macro': float(recall_score(ys, ypred, average='macro', zero_division=0)),
                    'f1_macro': float(f1_score(ys, ypred, average='macro', zero_division=0)),
                    'n_classes': int(len(np.unique(ys)))
                }

            eval_default = eval_model(clf, X_test, y_test)
            eval_pruned = eval_model(clf_pruned, X_test, y_test)

            safe_dp = str(dp).replace(' ', '_').replace('/', '_')
            entry = {
                'n_samples': int(len(group)),
                'meta': meta,
                'eval_default': eval_default,
                'eval_pruned': eval_pruned,
            }

            # === Save textual rules and visuals (same as your code) ===
            try:
                inv_label_map = {v: k for k, v in meta['label_map'].items()}
                class_names = [inv_label_map[int(c)] for c in clf.classes_]
                class_names_pruned = [inv_label_map[int(c)] for c in clf_pruned.classes_]

                rules_default = export_text(clf, feature_names=meta['feature_names'],
                                            show_weights=False, class_names=class_names)
                rules_pruned = export_text(clf_pruned, feature_names=meta['feature_names'],
                                        show_weights=False, class_names=class_names_pruned)
                
                rules_default_path = os.path.join(self.model_dir, f"dt_default_{safe_dp}.txt")
                rules_pruned_path = os.path.join(self.model_dir, f"dt_pruned_{safe_dp}.txt")

                with open(rules_default_path, 'w', encoding='utf-8') as f:
                    f.write(rules_default)
                with open(rules_pruned_path, 'w', encoding='utf-8') as f:
                    f.write(rules_pruned)

                entry.update({'rules_default': rules_default_path, 'rules_pruned': rules_pruned_path})
            except Exception as e:
                print("Exception Print rules:", e)
                warnings.append(f"rules_error: {e}")
                entry.setdefault('warnings', []).append(f"rules_error: {e}")

            # === Save evaluation metrics ===
            try:
                eval_path = os.path.join(self.model_dir, f"eval_{safe_dp}.txt")
                with open(eval_path, 'w', encoding='utf-8') as f:
                    f.write(f"Decision point: {dp}\nSamples: {entry['n_samples']}\n\n")
                    f.write("=== Default tree ===\n")
                    for k, v in eval_default.items():
                        f.write(f"{k}: {v}\n")
                    f.write("\n=== Pruned tree ===\n")
                    for k, v in eval_pruned.items():
                        f.write(f"{k}: {v}\n")
                entry['eval_path'] = eval_path
            except Exception as e:
                warnings.append(f"eval_save_error: {e}")
                entry.setdefault('warnings', []).append(f"eval_save_error: {e}")

            entry['warnings'] = warnings
            entry['model_objects'] = {'default': clf, 'pruned': clf_pruned}
            return dp, entry

        # === Parallel execution ===
        groups = list(self.df.groupby('decision_point'))
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

        with tqdm_joblib(tqdm(desc="Training decision trees", total=len(groups), unit="dp")):
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_train_single_decision_point)(dp, group)
                for dp, group in groups
            )

        # === Combine results back into dict ===
        results = dict(parallel_results)
        for dp, entry in tqdm(results.items(), desc="Generating tree visuals", unit="dp"):
            try:
                clf = entry['model_objects']['default']
                clf_pruned = entry['model_objects']['pruned']
                meta = entry['meta']
                safe_dp = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(dp))

                vis_default = os.path.join(self.model_dir, f"dt_default_{safe_dp}.png")
                plt.figure(figsize=(12, 8))
                sktree.plot_tree(clf, feature_names=meta['feature_names'],
                                class_names=[str(x) for x in meta['label_map'].keys()],
                                filled=True, fontsize=8)
                plt.tight_layout()
                plt.savefig(vis_default)
                plt.close()

                vis_pruned = os.path.join(self.model_dir, f"dt_pruned_{safe_dp}.png")
                plt.figure(figsize=(12, 8))
                sktree.plot_tree(clf_pruned, feature_names=meta['feature_names'],
                                class_names=[str(x) for x in meta['label_map'].keys()],
                                filled=True, fontsize=8)
                plt.tight_layout()
                plt.savefig(vis_pruned)
                plt.close()

                entry.update({'viz_default': vis_default, 'viz_pruned': vis_pruned})
            except Exception as e:
                entry.setdefault('warnings', []).append(f"viz_error: {e}")

        return results


    def load_model(self, model_path: str):
        raise RuntimeError("Models are not persisted to disk by default. Load from returned 'model_objects' in train_for_all results.")
