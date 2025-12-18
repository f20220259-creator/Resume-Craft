"""
-----------------------------------------------------------------------
File: main.py
Creation Time: Nov 24th 2023 7:04 pm
Author: Saurabh Zinjad
Developer Email: zinjadsaurabh1997@gmail.com
Copyright (c) 2023 Saurabh Zinjad. All rights reserved | GitHub: Ztrimus
-----------------------------------------------------------------------
"""

import argparse
from baseline_module import baseline_resume_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run baseline instead of LLM")
    parser.add_argument("--resume_text", type=str, help="Resume text input")
    parser.add_argument("--job_description", type=str, help="Job description input")
    args = parser.parse_args()

    if args.baseline:
        tailored_resume, scores = baseline_resume_generator(args.resume_text, args.job_description)
        print("\nOriginal Resume:\n", args.resume_text)
        print("\nJob Description:\n", args.job_description)
        print("\nBaseline Job-Specific Resume:\n", tailored_resume)
        print("\nSimilarity Scores:\n", scores)
