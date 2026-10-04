import os
import pandas as pd


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


# Paths are relative to this script's folder (script_experiments/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAWNICK_FILE = os.path.join(SCRIPT_DIR, "../data/chronicle2050/annotators/chronicle2050-shawnick-binary-v7.csv")
ADRIANO_FILE = os.path.join(SCRIPT_DIR, "../data/chronicle2050/annotators/chronicle2050-adriano-binary-v1.xlsx")
OUT_FILE = os.path.join(SCRIPT_DIR, "chronicle2050_disagreements.csv")

# Text label -> number
LABEL_MAP = {
    "prediction": 1, "1": 1, "1.0": 1,
    "not-prediction": 0, "non-prediction": 0, "not prediction": 0, "0": 0, "0.0": 0,
}


def read_any(path):
    """Read .csv or .xlsx."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


def to_binary(labels):
    """Convert text or numeric labels to 1 / 0 (empty stays empty)."""
    return labels.astype(str).str.strip().str.lower().map(LABEL_MAP)


shawnick = read_any(SHAWNICK_FILE).dropna(how="all").reset_index(drop=True)
adriano = read_any(ADRIANO_FILE).dropna(how="all").reset_index(drop=True)
print(f"Shawnick's file: {len(shawnick)} rows")
print(f"Adriano's file : {len(adriano)} rows")

if len(shawnick) != len(adriano):
    raise ValueError("The two files have different numbers of rows. Rows are matched by position, "
                     "so both files must keep all rows in the same order.")

df = pd.DataFrame({
    "Sentence": shawnick["sentence"],
    "Shawnick": to_binary(shawnick["Human Annotation"]),
    "Adriano": to_binary(adriano["Human Annotation"]),
    "Adriano Reasoning": adriano.get("Human Reasoning"),
})

both = df.dropna(subset=["Shawnick", "Adriano"])   # skip rows that either annotator left empty
sh_lab, ad_lab = both["Shawnick"].astype(int), both["Adriano"].astype(int)

kappa = cohen_kappa_score(sh_lab, ad_lab)
cm = confusion_matrix(sh_lab, ad_lab, labels=[0, 1])
agree = int((sh_lab == ad_lab).sum())
disagree = len(both) - agree

print("\n" + "=" * 45)
print(f"Sentences labeled by both : {len(both)}")
print(f"Skipped (empty in a file) : {len(df) - len(both)}")
print(f"Agreements                : {agree} ({agree / len(both):.1%})")
print(f"Disagreements             : {disagree} ({disagree / len(both):.1%})")
print(f"Cohen's kappa             : {kappa:.3f}")
print("=" * 45)
print("\nAgreement table (rows = Shawnick, columns = Adriano):")
print(pd.DataFrame(cm, index=["Shawnick 0", "Shawnick 1"], columns=["Adriano 0", "Adriano 1"]))

both[sh_lab != ad_lab].to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"\nDisagreements saved to {OUT_FILE}")
