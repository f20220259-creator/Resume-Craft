"""Quick test of all 4 scenarios with Gemma 2B on GPU."""
import sys
import requests

RESUME = """Ayush Kumar - Fintech Researcher
Skills: Python, TensorFlow, Power BI, Financial Modeling, R, AWS Cloud SQL, Tableau, Excel, GitHub
Experience: Data Analyst at Emerson FZE Dubai (built AI document processing, dashboards, automation)
Research Collaborator at Tecnologico de Monterrey (ESG analysis, predictive models, risk assessment)"""

SCENARIOS = {
    "1. AI Engineer (Perfect Match)": "AI Engineer - Build Generative AI apps with Python, PyTorch, LLMs, RAG, vector DBs. Requirements: Python, PyTorch, CUDA, HuggingFace, LangChain.",
    "2. Full Stack Developer (Pivot)": "Senior Full Stack Developer - Build web apps with React.js, Node.js, RESTful APIs, PostgreSQL. Requirements: HTML, CSS, JavaScript, React, AWS/Azure, CI/CD.",
    "3. Product Manager (Management)": "Technical Product Manager - Define product roadmap, collaborate with engineering, translate business goals to specs, manage sprints. Requirements: Technical background, stakeholder communication, project leadership.",
    "4. Head Chef (Stress Test)": "Head Chef - Lead kitchen with AI-driven inventory, create menus, manage staff, ensure food safety."
}

PROMPT = """You are a Career Consultant. Analyze this resume against the job description.

RESUME: {resume}

JOB: {jd}

Provide SPECIFIC advice using this EXACT format:

## Skill Translation Table
| Resume Skill | JD Requirement | How to Frame It |
|:---|:---|:---|
| [skill] | [requirement] | [framing suggestion] |

## 3 Missing Keywords
1. **[Keyword]**: How to add it
2. **[Keyword]**: How to add it  
3. **[Keyword]**: How to add it

## Pivot Pitch (2 sentences)
> [Your pitch here]
"""

print("=" * 60)
print("ResumeCraft - All 4 Test Scenarios")
print("=" * 60)
sys.stdout.flush()

for name, jd in SCENARIOS.items():
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {name}")
    print("=" * 60)
    sys.stdout.flush()
    
    try:
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma:2b",
            "prompt": PROMPT.format(resume=RESUME, jd=jd),
            "stream": False,
            "options": {"num_gpu": 35, "temperature": 0.3, "top_p": 0.9}
        }, timeout=120)
        
        print(resp.json().get("response", "Error"))
    except Exception as e:
        print(f"Error: {e}")
    sys.stdout.flush()

print("\n" + "=" * 60)
print("ALL SCENARIOS COMPLETED!")
print("=" * 60)
