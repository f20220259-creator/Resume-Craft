import pandas as pd
from baseline_module import baseline_resume_generator

# Load your dataset (CSV you created earlier)
df = pd.read_csv("resumes_dataset.csv")

# For demo, let’s just take the first 10 rows
#df = df.head(10)

df = df.sample(10, random_state=42)


# Job description
job_desc = "Seeking an HR Manager with skills in recruitment, employee relations, and compliance"

results = []
for i, row in df.iterrows():
    resume_text = row["Text"]  # column name from Resume-Atlas
    category = row["Category"]

    tailored_resume, scores = baseline_resume_generator(resume_text, job_desc)

    results.append({
        "id": i,
        "category": category,
        "original_resume": resume_text[:300],  # show first 300 chars only
        "job_description": job_desc,
        "baseline_resume": tailored_resume,
        "jaccard": scores["jaccard"],
        "rapidfuzz": scores["rapidfuzz"],
        "tfidf": float(scores["tfidf"])
    })

# Save output
output = pd.DataFrame(results)
output.to_csv("baseline_results.csv", index=False)

print("✅ Baseline results saved to baseline_results.csv")
