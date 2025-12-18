"""Test simplified prompt with complete example."""
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

RESUME = """Ayush Kumar - Fintech Researcher
ayushcsdev3073@gmail.com | +91-7338976356
Skills: Python, TensorFlow, Power BI, Financial Modeling, SQL, Excel, GitHub

Experience:
- Data Analyst, Emerson FZE Dubai (Feb-Aug 2025): Built dashboards, automation tools
- Research Collaborator, Tecnologico de Monterrey: ESG analysis, predictive models"""

JD = """Head Chef - Lead kitchen with AI-driven inventory. Manage staff, create menus, ensure food safety."""

PROMPT = """Career Consultant: Help this person apply for a new job.

THEIR RESUME:
{resume}

TARGET JOB:
{jd}

Copy this EXACT format and fill in with their actual data:

## Career Transition Analysis
Name: [their name]
Target: [job title from JD]

## 1. Skill Translation
| Skill | Application to Target Job |
|:---|:---|
| Python | Automate inventory and ordering systems |
| Excel | Track costs and manage budgets |
| Data Analysis | Forecast demand and reduce waste |
| SQL | Manage operational databases |
| Automation | Streamline workflows |

(Replace with their actual skills and how each applies to THIS job)

## 2. Experience Reframing
Keep their original job title. Show how to describe it for the new role:
- Data Analyst at X Company: "Experience with data systems applies to operational management"

## 3. What to Add
- [Certification for target job]: Why needed
- [Skill to learn]: Why needed
- [Experience to gain]: How to get it

## 4. Keywords for ATS
- [word from JD]
- [word from JD]
- [word from JD]

## 5. Summary
[Name] is a [background]. [How skills transfer to new role]."""

print("="*60)
print("CHEF TEST - SIMPLIFIED PROMPT WITH EXAMPLE")
print("="*60)

resp = requests.post("http://localhost:11434/api/generate", json={
    "model": "gemma:2b",
    "prompt": PROMPT.format(resume=RESUME, jd=JD),
    "stream": False,
    "options": {"num_gpu": 35, "temperature": 0.15}
}, timeout=120)

print(resp.json().get("response", "Error"))
