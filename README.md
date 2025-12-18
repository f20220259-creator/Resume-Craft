# ResumeCraft: Project Handover & Implementation Guide

## Overview

Welcome to **ResumeCraft** — an open-source, end-to-end **Job-Specific Resume Generation System** powered by **open-weight Large Language Models (LLMs)** and a **Skill-Gap Analysis Neural Adapter**.

ResumeCraft is designed as a **fully local, reproducible, and privacy-preserving alternative** to closed-source tools such as GPT-4 or Gemini-based resume generators. The system preprocesses resumes and job descriptions, identifies skill gaps using a lightweight MLP adapter, and generates **tailored, ATS-aware resumes** using **LLaMA-3 executed locally via Ollama**.

This repository serves as a **handover and implementation guide** for installing, configuring, and running ResumeCraft on local hardware.


## System Architecture (High-Level)

**Pipeline Flow:**

1. **Input**
   - Candidate Resume (PDF)
   - Target Job Description (Text)

2. **Data Preprocessing**
   - Text cleaning
   - Section parsing (Education, Experience, Skills, Projects)
   - Domain identification
   - Resume–JD pairing
   - Structured JSON generation

3. **Skill-Gap Analysis (MLP Adapter)**
   - Resume & JD embeddings (frozen embedder)
   - Feature interaction (concat, abs-diff, element-wise product)
   - Lightweight MLP adapter
   - Identification of missing / underrepresented skills

4. **LLM-Based Resume Tailoring**
   - LLaMA-3 model executed locally via Ollama
   - Job-specific prompt injection
   - Skill-gap–aware rewriting

5. **Output**
   - Fully tailored, job-aligned resume



## 1. System Prerequisites

To ensure stable local inference and embedding computation:

| Component | Minimum Requirement | Recommended |
|--------|---------------------|------------|
| OS | Windows 10/11 (64-bit) | Windows 11 |
| CPU | Quad-core 2.5 GHz+ | Hexa-core or better |
| GPU | NVIDIA GTX 1060 (6 GB VRAM) | NVIDIA RTX 3060+ (8 GB+ VRAM) |
| RAM | 8 GB | 16 GB+ |
| Storage | 5 GB free | SSD preferred |
| Python | 3.11+ | 3.11 |

> ⚠️ GPU acceleration is optional but **strongly recommended** for faster LLM inference.



## 2. Core Inference Engine: Ollama

ResumeCraft uses **Ollama** for local execution of open-source LLMs.

### Installation

1. Download Ollama from the official website  
   👉 https://ollama.ai

2. Install and ensure the Ollama service is running.

3. Verify installation:
```bash
ollama --version
