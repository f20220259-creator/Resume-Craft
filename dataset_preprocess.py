import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split

# Load pipeline CSV produced by dataset.py (schema: id, category, resume_json)
df = pd.read_csv("resumes_dataset.csv")
print("Before cleaning:", len(df))

# Parse resume_json and extract full text
def extract_text(rjson_str):
    try:
        data = json.loads(rjson_str)
        # Prefer FullText if present; otherwise join any section texts
        if isinstance(data, dict):
            if "FullText" in data and isinstance(data["FullText"], str):
                return data["FullText"]
            # Fallback: concatenate known sections
            parts = []
            for k in ["Education", "Experience", "Skills", "Projects"]:
                v = data.get(k)
                if isinstance(v, list):
                    parts.extend([str(x) for x in v])
            return " ".join(parts).strip() or None
    except Exception:
        return None
    return None

df["resume_text"] = df["resume_json"].apply(extract_text)

# Drop rows with missing resume text
df = df.dropna(subset=["resume_text"]).reset_index(drop=True)

# Cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?$%()'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['resume_text'] = df['resume_text'].apply(clean_text)

# Remove very short resumes
df = df[df['resume_text'].str.split().str.len() > 50]
print("After cleaning:", len(df))

# Keep consistent output columns
out_cols = ["id", "category", "resume_text", "resume_json"]
present_cols = [c for c in out_cols if c in df.columns]
df = df[present_cols]

# Split into 50/30/20
train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.375, random_state=42)

# Save
train.to_csv("train.csv", index=False)
val.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)
print("✅ Saved cleaned & split datasets.")

print(f"Train: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")
