import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree as sktree
import matplotlib.pyplot as plt
from sklearn.tree import export_text, export_graphviz
import graphviz


class DecisionMiningML:
    """Train decision trees per decision point from CSVs.

    Inputs/outputs:
    - Input: path to a CSV with columns including 'decision_point', 'activity' and other features.
    - Output: per-decision-point trained models and evaluation metrics.

    Error modes: raises FileNotFoundError for missing files, ValueError for missing required columns.
    """

    def __init__(self, csv_path: str, model_dir: str = "models"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self.csv_path = csv_path
        # try to auto-detect delimiter; fall back to default comma
        try:
            self.df = pd.read_csv(csv_path, sep=None, engine='python')
        except Exception:
            self.df = pd.read_csv(csv_path)

        # drop unnamed index columns that often appear when reading CSVs
        unnamed_cols = [c for c in self.df.columns if str(c).lower().startswith('unnamed') or str(c) == 'Unnamed: 0']
        if unnamed_cols:
            self.df = self.df.drop(columns=unnamed_cols)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

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

        X = pd.get_dummies(X, drop_first=True)

        # Align types
        X_vals = X.values
        # Encode y
        labels = sorted(y_raw.unique())
        label_map = {lab: i for i, lab in enumerate(labels)}
        y = y_raw.map(label_map).values

        meta = {'label_map': label_map, 'feature_names': list(X.columns)}
        return X_vals, y, meta

    def train_for_all(self, test_size: float = 0.3, random_state: int = 42,
                      prune_depth: int = 3, prune_min_samples_leaf: int = 10) -> Dict[str, Any]:
        """Train two trees per decision_point: a default tree and an aggressively pruned tree.

        Returns a dict keyed by decision_point with training/eval info and model paths.
        """
        self._validate()
        results = {}

        for dp, group in self.df.groupby('decision_point'):
            # For each decision point, we have rows where 'activity' is the observed choice
            try:
                X, y, meta = self._prepare_features(group)
            except Exception as e:
                results[dp] = {'error': str(e)}
                continue

            # split
            if len(np.unique(y)) < 2:
                results[dp] = {'error': 'Only one activity present, cannot train classifier'}
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
            )

            # default tree
            clf = DecisionTreeClassifier(random_state=random_state)
            clf.fit(X_train, y_train)

            # pruned tree (aggressive)
            clf_pruned = DecisionTreeClassifier(max_depth=prune_depth,
                                                min_samples_leaf=prune_min_samples_leaf,
                                                random_state=random_state)
            clf_pruned.fit(X_train, y_train)

            # evaluate
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

            # NOTE: previously models were saved to disk with joblib, but this behavior
            # was removed. We still keep the trained model objects in memory and include
            # any exported artifacts (rules, visualizations) below.
            safe_dp = str(dp).replace(' ', '_').replace('/', '_')
            default_path = None
            pruned_path = None

            # prepare result entry
            entry = {
                'n_samples': int(len(group)),
                'meta': meta,
                # models are kept in-memory only
                'default_model': None,
                'pruned_model': None,
                'eval_default': eval_default,
                'eval_pruned': eval_pruned,
            }

            # save textual rules
            try:
                rules_default = export_text(clf, feature_names=meta['feature_names'])
                rules_pruned = export_text(clf_pruned, feature_names=meta['feature_names'])
                rules_default_path = os.path.join(self.model_dir, f"dt_default_{safe_dp}.txt")
                rules_pruned_path = os.path.join(self.model_dir, f"dt_pruned_{safe_dp}.txt")
                with open(rules_default_path, 'w', encoding='utf-8') as f:
                    f.write(rules_default)
                with open(rules_pruned_path, 'w', encoding='utf-8') as f:
                    f.write(rules_pruned)
                entry.update({'rules_default': rules_default_path, 'rules_pruned': rules_pruned_path})
            except Exception as e:
                entry.setdefault('warnings', []).append(f"rules_error: {e}")

            # try Graphviz export (DOT -> PNG). Fallback to matplotlib plotting if graphviz not available.
            try:
                dot_default = export_graphviz(clf, out_file=None, feature_names=meta['feature_names'], class_names=[str(x) for x in meta['label_map'].keys()], filled=True, rounded=True)
                g_default = graphviz.Source(dot_default)
                vis_default = os.path.join(self.model_dir, f"dt_default_{safe_dp}.png")
                g_default.format = 'png'
                g_default.render(filename=os.path.splitext(vis_default)[0], cleanup=True)

                dot_pruned = export_graphviz(clf_pruned, out_file=None, feature_names=meta['feature_names'], class_names=[str(x) for x in meta['label_map'].keys()], filled=True, rounded=True)
                g_pruned = graphviz.Source(dot_pruned)
                vis_pruned = os.path.join(self.model_dir, f"dt_pruned_{safe_dp}.png")
                g_pruned.format = 'png'
                g_pruned.render(filename=os.path.splitext(vis_pruned)[0], cleanup=True)

                entry.update({'viz_default': vis_default, 'viz_pruned': vis_pruned})
            except Exception:
                # fallback to matplotlib plotting
                try:
                    vis_default = os.path.join(self.model_dir, f"dt_default_{safe_dp}.png")
                    plt.figure(figsize=(12, 8))
                    sktree.plot_tree(clf, feature_names=meta['feature_names'], class_names=[str(x) for x in meta['label_map'].keys()], filled=True, fontsize=8)
                    plt.title(f"Decision Tree (default) - {dp}")
                    plt.tight_layout()
                    plt.savefig(vis_default)
                    plt.close()

                    vis_pruned = os.path.join(self.model_dir, f"dt_pruned_{safe_dp}.png")
                    plt.figure(figsize=(12, 8))
                    sktree.plot_tree(clf_pruned, feature_names=meta['feature_names'], class_names=[str(x) for x in meta['label_map'].keys()], filled=True, fontsize=8)
                    plt.title(f"Decision Tree (pruned) - {dp}")
                    plt.tight_layout()
                    plt.savefig(vis_pruned)
                    plt.close()

                    entry.update({'viz_default': vis_default, 'viz_pruned': vis_pruned})
                except Exception as e:
                    entry.setdefault('warnings', []).append(f"viz_error: {e}")

            # attach the trained model objects in-memory under a separate key
            entry['model_objects'] = {'default': clf, 'pruned': clf_pruned}
            results[dp] = entry

        return results

    def load_model(self, model_path: str):
        raise RuntimeError("Models are not persisted to disk by default. Load from returned 'model_objects' in train_for_all results.")

