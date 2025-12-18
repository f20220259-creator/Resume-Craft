
import pandas as pd
import re
import argparse
import os

parser = argparse.ArgumentParser(description="Preprocess Job Descriptions CSV into preprocessed_jds.csv")
parser.add_argument("--jd_csv", type=str, default="job_title_des.csv", help="Path to the Job Descriptions CSV")
args = parser.parse_args()

jd_path = args.jd_csv
if not os.path.exists(jd_path):
    raise FileNotFoundError(f"Job descriptions CSV not found at: {jd_path}")

print(f"📄 Reading job descriptions from: {jd_path}")
df = pd.read_csv(jd_path)

def clean_text(text):
    """Basic cleaning for job descriptions"""
    text = re.sub(r"\s+", " ", str(text))  # remove extra spaces
    text = text.replace("\n", " ")
    return text.strip()

results = []
for i, row in df.iterrows():
    job_title = clean_text(row.get("Job Title", ""))
    job_desc = clean_text(row.get("Job Description", ""))

    if not job_desc:  # skip empty rows
        continue

    results.append({
        "id": i,
        "category": job_title,         # job title as category
        "job_description": job_desc
    })

output = pd.DataFrame(results)
output.to_csv("preprocessed_jds.csv", index=False)

print("✅ Preprocessing complete! Saved to preprocessed_jds.csv")
