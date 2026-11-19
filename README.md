# OT Lab Exam – Q4
### ML + Quantum + Parallel Computing: Bank Fraud Alert Selection
**Enrolment No:** 241070022

---

## Problem Statement
A bank has 12 flagged alerts today; reviewers can check only **5 of 12**.  
Use KNN fraud-probability scores, build a QUBO, solve with Simulated Quantum Annealing, and parallelize with `joblib`.

---

## Notebook: `241070022_OT_LAB_EXAM_Q4.ipynb`

| Part | Topic | Marks | CO |
|------|-------|-------|----|
| (a) | Machine Learning – KNN fraud classifier (600 rows, seed=22, 70/30 split) | 1 | CO1 |
| (b) | QUBO formulation – Q matrix, penalty, brute-force verification | 1 | CO2 |
| (c) | Quantum Computing – Simulated Annealing, compare with brute force | 1.5 | CO2 |
| (d) | Parallel Computing – `joblib.Parallel` grid of (γ, β), speedup & efficiency | 1 | CO4 |
| (e) | Conclusion (3 lines) | 0.5 | CO4 |

---

## Running on Google Colab
1. Upload `241070022_OT_LAB_EXAM_Q4.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run the first cell **"Install Dependencies"** (installs required packages)
3. Run all remaining cells top-to-bottom (`Runtime → Run all`)
4. Graphs/plots are saved to the `results/` folder automatically

---

## Results Folder
All diagrams and output plots are saved in `results/`:

| File | Description |
|------|-------------|
| `a1_feature_distributions.png` | Histograms of features by class |
| `a2_k_selection.png` | KNN k-value vs accuracy curve |
| `a3_confusion_matrix.png` | Confusion matrix on test set |
| `a4_alert_fraud_probabilities.png` | KNN fraud scores for 12 alerts |
| `b1_qubo_Q_matrix.png` | Heatmap of the 12×12 Q matrix |
| `b2_brute_force_energy_dist.png` | Energy distribution over all C(12,5) solutions |
| `c1_sa_convergence_distribution.png` | SA convergence + energy distribution (500 runs) |
| `d1_parallel_benchmark.png` | Wall time / speedup / efficiency for 1,2,4 workers |
| `d2_energy_landscape.png` | QUBO energy over (γ, β) grid |

---

## Dependencies
All standard — available on Colab without extra setup:
`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `tqdm`
