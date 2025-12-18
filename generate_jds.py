
import csv
import requests
import json
import time
from tqdm import tqdm
import os
import random

# Configuration
INPUT_FILE = "preprocessed_dataset.csv"  # Only reading extracted resumes
OUTPUT_FILE = "synthetic_training_dataset.csv"
MODEL_NAME = "gemma:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_SAMPLES = 50  # Start with a small batch

def generate_jd(resume_text):
    """
    Uses Llama3 to hallucinate a matching Job Description for a given resume.
    """
    system_prompt = (
        "You are an expert HR Recruiter. "
        "Read the following resume snippet and generate a realistic Job Description (JD) "
        "that this candidate would be perfectly qualified for. "
        "The JD should include: Job Title, Responsibilities, and Required Skills. "
        "Keep it concise but detailed enough for semantic matching."
    )
    
    prompt = f"Resume Content:\n{resume_text[:2000]}\n\nGenerate a matching Job Description:"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7 
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error generating JD: {e}")
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Loading Resumes...")
    
    # Read all rows first
    rows = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure we sort/shuffle or pick a subset
    if len(rows) > MAX_SAMPLES:
        random.seed(42)
        rows = random.sample(rows, MAX_SAMPLES)
    
    print(f"Generating JDs for {len(rows)} resumes using {MODEL_NAME}...")
    print("This requires a running GPU and may take time.")

    # List to store results
    synthetic_data = []

    for index, row in tqdm(enumerate(rows), total=len(rows)):
        resume_json_str = row.get('resume_json', '')
        if not resume_json_str or len(resume_json_str) < 10:
            continue
        
        resume_text = ""
        try:
            resume_data = json.loads(resume_json_str)
            # Flatten the resume content for the LLM
            education = " ".join(resume_data.get("Education", [])) if isinstance(resume_data.get("Education"), list) else ""
            experience = " ".join(resume_data.get("Experience", [])) if isinstance(resume_data.get("Experience"), list) else ""
            skills = " ".join(resume_data.get("Skills", [])) if isinstance(resume_data.get("Skills"), list) else ""
            
            resume_text = f"Education: {education}\nExperience: {experience}\nSkills: {skills}"
        except Exception as e:
            # print(f"Error parsing JSON for row {index}: {e}")
            continue

        if len(resume_text) < 50:
            continue
            
        jd = generate_jd(resume_text)
        
    # Initialize file with header if not exists
    file_exists = os.path.isfile(OUTPUT_FILE)
    if not file_exists:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Resume", "Job_Description", "Category"])
            writer.writeheader()

    for index, row in tqdm(enumerate(rows), total=len(rows)):
        resume_json_str = row.get('resume_json', '')
        if not resume_json_str or len(resume_json_str) < 10:
            continue
        
        resume_text = ""
        try:
            resume_data = json.loads(resume_json_str)
            # Flatten the resume content for the LLM
            education = " ".join(resume_data.get("Education", [])) if isinstance(resume_data.get("Education"), list) else ""
            experience = " ".join(resume_data.get("Experience", [])) if isinstance(resume_data.get("Experience"), list) else ""
            skills = " ".join(resume_data.get("Skills", [])) if isinstance(resume_data.get("Skills"), list) else ""
            
            resume_text = f"Education: {education}\nExperience: {experience}\nSkills: {skills}"
        except Exception as e:
            continue

        if len(resume_text) < 50:
            continue
            
        jd = generate_jd(resume_text)
        
        if jd:
            new_row = {
                "Resume": resume_text,
                "Job_Description": jd,
                "Category": row.get('Category', 'Unknown')
            }
            # Append immediately
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["Resume", "Job_Description", "Category"])
                writer.writerow(new_row)
            
    print(f"Done! Saved content to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
