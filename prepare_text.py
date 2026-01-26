import pandas as pd
import os
out = os.path.join(os.path.expanduser("~"), "C:", "ny311_ready_900.csv")


inp = r"C:\temp_nlp\output\ny311_ready_900.csv"
out = r"C:\temp_nlp\output\ny311_finished_900.csv"

df = pd.read_csv(inp, low_memory=False, dtype=str)

text_cols = [c for c in ["Complaint Type", "Descriptor", "Resolution Description"] if c in df.columns]

df["text"] = (
    df[text_cols].fillna("").agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df[df["text"].str.len() > 0]
df.to_csv(out, index=False, encoding="utf-8")

print("DONE:", out)
print("TEXT COLS USED:", text_cols)
print("ROWS:", len(df))


