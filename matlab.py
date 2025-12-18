import os
import matplotlib as mpl
if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Pipeline stages
stages = [
    "Baseline",
    "LLaMA-3 Tailored",
    "Skill-Gap Enhanced",
    "Final Tailored"
]

# Similarity metrics
cosine = [0.2800, 0.4758, 0.5210, 0.5685]
jaccard = [0.1250, 0.2024, 0.2360, 0.2718]
tfidf = [0.4560, 0.4758, 0.5325, 0.5894]
fuzzy = [0.4560, 0.6837, 0.7210, 0.7589]

x = np.arange(len(stages))
width = 0.2

plt.figure(figsize=(9, 5))
plt.bar(x - 1.5*width, cosine, width, label="Cosine Similarity")
plt.bar(x - 0.5*width, jaccard, width, label="Jaccard Index")
plt.bar(x + 0.5*width, tfidf, width, label="TF-IDF Cosine")
plt.bar(x + 1.5*width, fuzzy, width, label="Fuzzy Ratio")

plt.xlabel("Pipeline Stage")
plt.ylabel("Similarity Score")
plt.xticks(x, stages, rotation=20)
plt.title("Comparison of Resume–Job Alignment Metrics Across Pipeline Stages")
plt.legend()
plt.tight_layout()
plt.savefig("pipeline_metrics.png", dpi=200, bbox_inches="tight")
plt.close()
