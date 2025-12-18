import pandas as pd
import json
import re

# Load Resume-Atlas dataset
df = pd.read_csv("resumes_dataset.csv")

def clean_text(text):
    """Basic text cleaning for resumes"""
    text = re.sub(r"\s+", " ", str(text))  # remove extra spaces
    text = text.replace("\n", " ")
    return text.strip()

def preprocess_resume(resume_text):
    """Convert raw resume text into JSON sections (simple heuristic)"""
    resume_json = {
        "Education": [],
        "Experience": [],
        "Skills": [],
        "Projects": []
    }

    text = clean_text(resume_text)

    # Very naive splitting
    if "education" in text.lower():
        resume_json["Education"].append(text)
    if "experience" in text.lower():
        resume_json["Experience"].append(text)
    if "skills" in text.lower():
        resume_json["Skills"].append(text)
    if "project" in text.lower():
        resume_json["Projects"].append(text)

    return resume_json

results = []
for i, row in df.iterrows():
    resume_text = clean_text(row["Text"])
    resume_json = preprocess_resume(resume_text)

    results.append({
        "id": i,
        "category": row["Category"],
        "resume_json": json.dumps(resume_json)
    })

output = pd.DataFrame(results)
output.to_csv("preprocessed_dataset.csv", index=False)

print("✅ Preprocessing complete! Saved to preprocessed_dataset.csv")
