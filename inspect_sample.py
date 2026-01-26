import pandas as pd

path = r"C:\temp_nlp\output\ny311_ready_900.csv"
df = pd.read_csv(path, low_memory=False)

print("ROWS:", len(df))
print("COLUMNS:", list(df.columns))
print(df.head(3).to_string(index=False))
