import pandas as pd
import json
import os

# Load the two datasets
resumes = pd.read_csv("resumes_dataset.csv") 
jobs_path = "job_title_des.csv"
if not os.path.exists(jobs_path):
    raise FileNotFoundError(f"JD CSV not found at: {jobs_path}. Place it in the project root or update the path.")
jobs = pd.read_csv(jobs_path)


print("Resumes:", len(resumes))
print("Job Descriptions:", len(jobs))

# Ensure both have a category column for matching
resumes["category"] = resumes["category"].astype(str).str.lower().str.strip()
# Support either preprocessed (category) or raw JD (Job Title)
if "category" in jobs.columns:
    jobs["category"] = jobs["category"].astype(str).str.lower().str.strip()
elif "Job Title" in jobs.columns:
    jobs["category"] = jobs["Job Title"].astype(str).str.lower().str.strip()
else:
    raise KeyError("JD CSV must include either 'category' or 'Job Title' column")

# Ensure resume_text exists in resumes by extracting from resume_json if needed
if "resume_text" not in resumes.columns:
    if "resume_json" not in resumes.columns:
        raise KeyError("resumes_dataset.csv must have either 'resume_text' or 'resume_json'")
    def extract_text(rjson_str):
        try:
            data = json.loads(rjson_str)
            if isinstance(data, dict):
                if isinstance(data.get("FullText"), str):
                    return data["FullText"]
                parts = []
                for k in ["Education", "Experience", "Skills", "Projects"]:
                    v = data.get(k)
                    if isinstance(v, list):
                        parts.extend([str(x) for x in v])
                return " ".join(parts).strip() or None
        except Exception:
            return None
        return None
    resumes["resume_text"] = resumes["resume_json"].apply(extract_text)
    resumes = resumes.dropna(subset=["resume_text"]).reset_index(drop=True)

# Merge by category / job title
merged = pd.merge(resumes, jobs, on="category", how="inner")

print("✅ Paired dataset created successfully!")
print("Total pairs:", len(merged))

# Keep only useful columns
jd_col = "job_description" if "job_description" in merged.columns else ("Job Description" if "Job Description" in merged.columns else None)
cols = ["category", "resume_text"] + ([jd_col] if jd_col else [])
paired = merged[cols]
if jd_col and jd_col != "job_description":
    paired = paired.rename(columns={jd_col: "job_description"})

# Save paired dataset
paired.to_csv("resume_jd_pairs.csv", index=False)
print("💾 Saved paired dataset → resume_jd_pairs.csv")
