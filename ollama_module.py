"""
Minimal Ollama client for generating job-specific resumes without importing the
full zlm package or langchain dependencies.
"""
import json
import requests


class LLMModel:
    def __init__(self, model_name="gemma:2b", base_url="http://localhost:11434"):
        """Initialize the LLM model with the specified model name.

        Args:
            model_name (str): Name of the Ollama model to use (default: "gemma:2b")
            base_url (str): Base URL for the local Ollama server
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.system_prompt = (
            "You are an AI assistant that helps optimize resumes for specific job descriptions. "
            "Modify the given resume to better match the provided job description. "
            "Focus on highlighting relevant skills and experiences that align with the job requirements. "
            "Return only the modified resume in JSON format."
        )

    def generate_critique(self, resume_text, jd_text):
        """
        Generates career advice for tailoring resume to JD.
        """
        prompt = f"""Career Consultant: Help this person apply for a new job.

THEIR RESUME:
{resume_text[:3000]}

TARGET JOB:
{jd_text[:1200]}

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
[Name] is a [background]. [How skills transfer to new role].
"""
        
        self.force_unload("mxbai-embed-large")
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        """Call the local Ollama HTTP API to generate a response."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": "gemma:2b", # Force Gemma 2B for generation (smarter than mxbai, lighter than llama3)
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "num_gpu": 35,  # Force all layers to GPU (GTX 1060 6GB)
                "num_ctx": 4096,  # Context window
                "temperature": 0.3,  # Lower = more focused/deterministic
                "top_p": 0.9,  # Nucleus sampling for coherence
                "top_k": 40,  # Limit token choices
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"Error generating response: {e}"

    def get_vector(self, text: str):
        """Generate embeddings for the given text using local Ollama.

        Args:
            text (str): Input text to embed.

        Returns:
            numpy.ndarray: The embedding vector.
            None: If extraction fails.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required. Please install it with `pip install numpy`.")

        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model_name,
            "input": text,
            "keep_alive": 0, # Force unload to save VRAM for potential Llama3 swap
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                return None
            return np.array(embeddings[0])
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Fallback to legacy /api/embeddings
                url = f"{self.base_url}/api/embeddings"
                payload = {
                    "model": self.model_name,
                    "prompt": text,
                    # Legacy endpoint often doesn't support keep_alive or handles it poorly
                }
                try:
                    resp = requests.post(url, json=payload, timeout=600)
                    resp.raise_for_status()
                    data = resp.json()
                    embedding = data.get("embedding", [])
                    if not embedding:
                        return None
                    return np.array(embedding)
                except Exception as ex:
                    print(f"Legacy Embedding Error: {ex}")
                    return None
            else:
                print(f"Ollama API Error: {e}")
                return None
        except Exception as e:
            print(f"Unexpected Error in get_vector: {e}")
            return None

    def force_unload(self, model_name):
        """Force a model to unload from VRAM to prevent OOM."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "keep_alive": 0
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except:
            pass


    def generate_resume(self, resume_json, job_description):
        """DEPRECATED: Use the MLP pipeline instead.
        This legacy method is kept for backward compatibility but warns about deprecation.
        """
        print("WARNING: 'generate_resume' is deprecated. Use the MLP pipeline with 'get_vector'.")
        # Legacy implementation wrapped in try-catch or preserved if needed, 
        # but for this refactor we acknowledge it's not the primary path anymore.
        pass

