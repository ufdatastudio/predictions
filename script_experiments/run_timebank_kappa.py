import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SON_FILE = os.path.join(SCRIPT_DIR, "../data/timebank_1_2/annotators/joined_files/timebank-son-binary-v1.csv")
MAYA_FILE = os.path.join(SCRIPT_DIR, "../data/timebank_1_2/annotators/joined_files/timebank-maya-binary-v1.csv")
OUT_FILE = os.path.join(SCRIPT_DIR, "timebank_disagreements.csv")


def cohen_kappa_score(a, b):
    """Cohen's kappa for two raters (no scikit-learn needed)."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    p_observed = (a == b).mean()
    labels = sorted(set(a) | set(b))
    p_chance = sum((a == k).mean() * (b == k).mean() for k in labels)
    return (p_observed - p_chance) / (1 - p_chance)


def confusion_matrix(a, b, labels):
    """Counts: rows = first rater, columns = second rater."""
    return pd.crosstab(pd.Categorical(a, categories=labels),
                       pd.Categorical(b, categories=labels), dropna=False).values


def read_csv_any(path):
    for enc in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


son = read_csv_any(SON_FILE)
maya = read_csv_any(MAYA_FILE)
print(f"Son's file : {len(son)} rows")
print(f"Maya's file: {len(maya)} rows")

if len(son) != len(maya):
    raise ValueError("The two files have different numbers of rows. Rows are matched by position, "
                     "so both files must keep all rows in the same order.")

df = pd.DataFrame({
    "Base Sentence": son["Base Sentence"],
    "Son": pd.to_numeric(son["Human Annotation"], errors="coerce"),
    "Maya": pd.to_numeric(maya["Human Annotation"], errors="coerce"),
    "Son Reasoning": son.get("Human Reasoning"),
    "Maya Reasoning": maya.get("Human Reasoning"),
})

both = df.dropna(subset=["Son", "Maya"])          # skip rows that either annotator left empty
son_lab, maya_lab = both["Son"].astype(int), both["Maya"].astype(int)

kappa = cohen_kappa_score(son_lab, maya_lab)
cm = confusion_matrix(son_lab, maya_lab, labels=[0, 1])
agree = int((son_lab == maya_lab).sum())
disagree = len(both) - agree

print("\n" + "=" * 45)
print(f"Sentences labeled by both : {len(both)}")
print(f"Skipped (empty in a file) : {len(df) - len(both)}")
print(f"Agreements                : {agree} ({agree / len(both):.1%})")
print(f"Disagreements             : {disagree} ({disagree / len(both):.1%})")
print(f"Cohen's kappa             : {kappa:.3f}")
print("=" * 45)
print("\nAgreement table (rows = Son, columns = Maya):")
print(pd.DataFrame(cm, index=["Son 0", "Son 1"], columns=["Maya 0", "Maya 1"]))

both[son_lab != maya_lab].to_csv(OUT_FILE, index=False)
print(f"\nDisagreements saved to {OUT_FILE}")
