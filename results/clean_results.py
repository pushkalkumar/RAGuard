import csv

input_file = "summary_all_raw.csv"
output_file = "summary_all_final.csv"

with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    seen = set()
    for row in reader:
        if not row or "Dataset" in row[0] or "Retriever" in row[0]:
            continue
        if len(row) >= 4:
            # keep only the first four columns
            clean_row = row[:4]
            if tuple(clean_row) not in seen:
                writer.writerow(clean_row)
                seen.add(tuple(clean_row))

print("✅ Cleaned CSV written to summary_all_final.csv")

