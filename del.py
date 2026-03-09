import os
import csv

facts_dir = "data/companyfacts"
input_csv = "data/company_tickers.csv"
output_csv = "data/company_tickers_filtered.csv"

# Step 1: collect CIKs that have fact files
fact_ciks = set()

with os.scandir(facts_dir) as entries:
    for entry in entries:
        if entry.name.startswith("CIK") and entry.name.endswith(".json"):
            cik = entry.name[3:-5]
            fact_ciks.add(cik)

# Step 2: filter CSV
kept = 0
removed = 0

with (
    open(input_csv, newline="", encoding="utf-8") as infile,
    open(output_csv, "w", newline="", encoding="utf-8") as outfile,
):
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        cik = row["cik_str"].zfill(10)

        if cik in fact_ciks:
            writer.writerow(row)
            kept += 1
        else:
            removed += 1

print("Companies kept:", kept)
print("Companies removed:", removed)
