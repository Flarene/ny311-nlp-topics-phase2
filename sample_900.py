import csv
import os
import uuid

inp = r"C:\temp_nlp\raw\311-service-requests-from-2010-to-present.csv"
out = r"C:\temp_nlp\output\ny311_ready_900.csv"
n_rows = 900

# Step A: test input read
try:
    fin = open(inp, "r", encoding="utf-8", errors="ignore", newline="")
    fin.close()
except Exception as e:
    print("INPUT_FAIL:", inp)
    raise

# Step B: test output write
try:
    fout = open(out, "w", encoding="utf-8", newline="")
    fout.close()
except Exception as e:
    print("OUTPUT_FAIL:", out)
    raise

# Step C: create the sample
with open(inp, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
     open(out, "w", encoding="utf-8", newline="") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    writer.writerow(next(reader))
    for i, row in enumerate(reader, start=1):
        writer.writerow(row)
        if i >= n_rows:
            break

print("DONE:", out)