from datasets import load_dataset
import pandas as pd
import re
import json

print("⏳ Downloading full ResumeAtlas dataset...")

# Load dataset from Hugging Face
ds = load_dataset("ahmedheakl/resume-atlas", split="train")

# Convert to pandas DataFrame
df = pd.DataFrame(ds)

# Auto-detect text and optional category columns
text_candidates = [
    "resume_text", "text", "Text", "resume", "content", "Resume", "ResumeText",
]
category_candidates = [
    "category", "label", "industry", "job_title", "title", "Category",
]

text_col = next((c for c in text_candidates if c in df.columns), None)
cat_col = next((c for c in category_candidates if c in df.columns), None)

if not text_col:
    # Provide a clear error with available columns
    raise KeyError(
        f"Could not find a resume text column. Available columns: {list(df.columns)}. "
        f"Looked for any of: {text_candidates}"
    )

# Drop rows missing text
df = df.dropna(subset=[text_col])

print("✅ Download complete!")
print("Total resumes:", len(df))

# Save full dataset as CSV for reference
df.to_csv("ResumeAtlas_full.csv", index=False)
print("✅ Saved full dump as ResumeAtlas_full.csv")

# Build id, category, resume_json
def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.replace("\n", " ").strip()

def preprocess_resume(resume_text):
    resume_json = {
        "Education": [],
        "Experience": [],
        "Skills": [],
        "Projects": []
    }
    t = clean_text(resume_text)
    low = t.lower()
    if "education" in low:
        resume_json["Education"].append(t)
    if "experience" in low:
        resume_json["Experience"].append(t)
    if "skills" in low:
        resume_json["Skills"].append(t)
    if "project" in low:
        resume_json["Projects"].append(t)
    # Always include full text for downstream flexibility
    resume_json["FullText"] = t
    return resume_json

results = []
for i, row in df.iterrows():
    t = row[text_col]
    cat = row[cat_col] if cat_col else ""
    rjson = preprocess_resume(t)
    results.append({
        "id": i,
        "category": cat,
        "resume_json": json.dumps(rjson)
    })

out = pd.DataFrame(results)
out.to_csv("resumes_dataset.csv", index=False)
print("✅ Saved pipeline CSV as resumes_dataset.csv with columns:", list(out.columns))
