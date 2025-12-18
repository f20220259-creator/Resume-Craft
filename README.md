# ResumeCraft: Project Handover & Implementation Guide

<p align="center">
  <img src="ARCHITECTURE FINAL.png" alt="ResumeCraft Architecture" width="900"/>
</p>

---

Welcome to the ResumeCraft ecosystem. This documentation serves as a comprehensive guide for the installation, configuration, and operation of the local AI Resume Consultant.  

ResumeCraft leverages the Gemma 2B model (orchestrated via Ollama) and a specialized MLP Adapter to provide high-fidelity resume tailoring and expert-level career consulting directly on local hardware.

---

## 1. System Prerequisites

To ensure optimal performance—particularly during inference—ensure the host machine meets the following specifications:

| Component | Minimum Requirement | Recommended |
|----------|---------------------|------------|
| Operating System | Windows 10/11 (64-bit) | Windows 11 |
| Processor | Quad-core 2.5GHz+ | Hexa-core or better |
| GPU | NVIDIA GTX 1060 (6GB VRAM) | NVIDIA RTX 3060+ (8GB+ VRAM) |
| Memory (RAM) | 8 GB | 16 GB+ |
| Storage | 5 GB available space | SSD preferred |
| Environment | Python 3.11+ | Python 3.11 |

---

## 2. Core Engine Setup: Ollama

ResumeCraft relies on Ollama as the backend inference engine.

### Installation

Download the installer from the official Ollama website.  
Execute the installer and ensure the Ollama service is active in your system tray.  

Verify the installation by executing the following in your terminal:

```bash
ollama --version
