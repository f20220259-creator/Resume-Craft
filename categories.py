import pandas as pd
import subprocess
import time
import os

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
MODEL = "gemma:2b"               # Model in Ollama (make sure it's pulled)
TIMEOUT = 600                  # Timeout per resume (10 min max)
BATCH_SIZE = 10                # Save progress every 10
SAMPLE_SIZE = 50               # Generate for first 50 resumes

INPUT_FILE = "train_jdresume.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1️⃣ Load Dataset
# ----------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
print(f"🔹 Loaded {len(df)} total rows from {INPUT_FILE}")
df = df.head(SAMPLE_SIZE)
print(f"🎯 Processing only first {len(df)} rows for testing")

# Auto-detect correct column names
resume_col = next(c for c in df.columns if "resume" in c.lower())
jd_col = next(c for c in df.columns if "job" in c.lower() or "description" in c.lower())
output_col = "tailored_resume"

print(f"🧭 Using columns: resume = '{resume_col}', JD = '{jd_col}'")

# ----------------------------------------------------------
# 2️⃣ Generate Tailored Resumes
# ----------------------------------------------------------
outputs = []

for i, row in df.iterrows():
    resume = str(row[resume_col])
    jd = str(row[jd_col])

    prompt = f"""
You are an expert resume writer.
Rewrite the following resume to best match the job description while keeping all information truthful.
Emphasize relevant skills, achievements, and experiences.
Maintain a clean professional format with: Summary, Skills, Experience, Projects, Education.

Resume:
{resume}

Job Description:
{jd}

Return only the rewritten resume. Do not include explanations.
"""

    try:
        proc = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=TIMEOUT
        )
        output = proc.stdout.strip()
    except Exception as e:
        print(f"⚠️ Error at row {i}: {e}")
        output = ""

    outputs.append(output)

    if (i + 1) % BATCH_SIZE == 0:
        print(f"✅ Processed {i + 1}/{len(df)} resumes...")
        temp_df = df.iloc[:i + 1].copy()
        temp_df[output_col] = outputs
        temp_df.to_csv(f"{OUTPUT_DIR}/train_tailored_partial.csv", index=False)
        time.sleep(1)

# ----------------------------------------------------------
# 3️⃣ Save Final Output
# ----------------------------------------------------------
df[output_col] = outputs
df.to_csv(f"{OUTPUT_DIR}/train_tailored_50.csv", index=False)
print("\n🎯 Generation complete!")
print(f"💾 Saved → {OUTPUT_DIR}/train_tailored_50.csv")
