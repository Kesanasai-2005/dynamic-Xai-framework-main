# Dynamic Explainable AI Framework — Complete Study Notes

**UI note (only):** Streamlit powers the UI elements and web server. Nothing else in this guide depends on it. The real work is the CLI pipeline in `main.py`: load → analyze → preprocess → train → per-application HII (SHAP/LIME) → compare → report.

---

## 1. What this project is

A **tabular classification** framework that does two things for every model:

1. **Predict** — train a classifier and score it (Accuracy, Precision, Recall, F1).
2. **Explain & score the explanations** — use SHAP and LIME on test rows (“applications”), then combine four explanation-quality metrics into one score: **HII (Hybrid Interpretability Index)**.

Default demo data: `data/raw/heart.csv`, target column `disease`.

**Pipeline (conceptually):**

```
CSV → analyze → impute/encode/scale → train models
     → pick n_apps test rows
     → SHAP + LIME per row
     → Consistency, Stability, Fidelity, Feature Agreement
     → HII = weighted average
     → comparison table (performance + interpretability)
```

---

## 2. Big-picture concepts

### 2.1 Explainable AI (XAI)

1. **Definition**  
   Techniques that explain *why* a model predicted a class for a given input, usually by assigning importance (and often direction: push toward / away from a class) to each feature.

2. **Example**  
   Model says “disease = yes” for a patient. An explanation might say: high cholesterol pushed toward yes (+0.4), young age pushed away (−0.2).

3. **Why we care here**  
   Accuracy alone does not tell you if explanations are trustworthy, stable, or agree across methods. This project measures that.

4. **Simple terms**  
   Not just “what did the model answer?” but “which inputs drove that answer, and can we trust that story?”

---

### 2.2 Local vs global explanation

1. **Definition**  
   - **Local:** explanation for one row (one application / patient / loan).  
   - **Global:** average importance across many rows.

2. **Example**  
   Local: for patient #17, `thalach` mattered most. Global: across 20 patients, `|SHAP|` of `cp` is highest on average.

3. **Why we calculate both**  
   HII metrics are computed **per application**, then averaged. Global SHAP/LIME importances help compare which features dominate overall.

4. **Simple terms**  
   Local = one person’s story. Global = the usual story across many people.

---

### 2.3 SHAP (SHapley Additive exPlanations)

1. **Definition**  
   Attribution method from game theory: each feature gets a contribution so that contributions add up toward the model’s output (relative to a baseline). Positive SHAP → pushes toward the target class; negative → pushes away.

2. **Example**  
   Baseline probability of disease ≈ 0.30. Features contribute: age +0.15, cholesterol +0.10, resting BP −0.05 → roughly 0.30 + 0.15 + 0.10 − 0.05 = 0.50 predicted probability.

3. **Why we use it**  
   Solid, widely used attributions. Used for **Consistency**, **Stability**, **Feature Agreement**, and global importance. Tree models use `TreeExplainer`; logistic regression uses **Permutation SHAP** on `predict_proba` for the chosen class.

4. **Simple terms**  
   Fair credit-sharing: “how much did each feature add or subtract from this prediction?”

**How this repo picks SHAP:**

| Model type | SHAP engine |
|---|---|
| Decision Tree / Random Forest | `shap.TreeExplainer` |
| Logistic Regression (and other non-trees) | Permutation `shap.Explainer` on class probability |

---

### 2.4 LIME (Local Interpretable Model-agnostic Explanations)

1. **Definition**  
   Around one instance, LIME samples nearby points, gets the black-box predictions, and fits a **simple local linear model**. Coefficients of that local model are the LIME feature weights. `exp.score` is how well that local model fits (weighted R²).

2. **Example**  
   For loan applicant X, LIME samples neighbors, fits a local linear model, and says: income +0.4, debt −0.3 toward “Approved”. If those neighbors’ predictions are well explained by that line, R² (score) is high (e.g. 0.85).

3. **Why we use it**  
   Independent second explainer. Compared to SHAP for **Consistency** and **Feature Agreement**. Its local R² becomes **Fidelity**.

4. **Simple terms**  
   Zoom into one prediction, pretend the model is a straight line nearby, and read the slopes — then check if that line was a good fit.

Config knobs: `lime_num_samples` (default 1000), `discretize_continuous=True`.

---

### 2.5 Target class for explanations

1. **Definition**  
   SHAP/LIME are computed for one class index (e.g. “disease present” or “Approved”).

2. **Example**  
   Binary heart disease: class index 1. Loan data with labels `Approved` / `Rejected`: prefer `Approved` if present.

3. **Why**  
   “Importance toward disease” is clearer than mixing both classes.

4. **Simple terms**  
   We always ask: “what pushed the model toward *this* outcome?”

Resolution order in code: `positive_class` config → label `"Approved"` if present → else class index `1` (or `0` if only one class).

---

### 2.6 Top-K nonzero features

1. **Definition**  
   From a SHAP (or LIME) vector, take features with |weight| > ~0, sort by absolute magnitude, keep at most `top_k` (default 10).

2. **Example**  
   SHAP = `[0.5, -0.4, 0.01, 0.0]`, top_k=2 → indices of 0.5 and −0.4.

3. **Why**  
   Consistency and Feature Agreement focus on features that actually matter, not noise/zeros.

4. **Simple terms**  
   Only judge the “headline” features, not every tiny coefficient.

---

## 3. Preprocessing concepts

### 3.1 Train / test split (or uploaded test set)

1. **Definition**  
   Hold out data the model never trains on, so scores reflect generalization. Optionally supply a separate test CSV with the same columns.

2. **Example**  
   1000 rows, `test_size=0.2` → ~800 train / ~200 test (stratified when possible).

3. **Why**  
   Explaining only training rows would overstate how good explanations look in the real world.

4. **Simple terms**  
   Study on homework, grade on a new quiz.

---

### 3.2 Missing value imputation

1. **Definition**  
   Fill empty cells: numeric with median (configurable); categorical with most frequent value.

2. **Example**  
   Cholesterol missing → replace with median cholesterol of the training set.

3. **Why**  
   Most sklearn models cannot train on NaNs.

4. **Simple terms**  
   Guess a sensible fill instead of dropping every incomplete row.

---

### 3.3 StandardScaler (numeric)

1. **Definition**  
   Transform each numeric feature to mean ≈ 0, std ≈ 1 using **train** statistics.

2. **Example**  
   Age 60 with train mean 50, std 10 → (60−50)/10 = 1.0.

3. **Why**  
   Helps logistic regression; keeps numeric scales comparable. Feature names become `num__age`, etc.

4. **Simple terms**  
   Put every number on a common “how many standard deviations from average” scale.

---

### 3.4 One-hot encoding (categorical)

1. **Definition**  
   Turn a category into 0/1 columns, one per level (`handle_unknown="ignore"`).

2. **Example**  
   `cp` ∈ {typical, atypical, other} → `cat__cp_typical`, `cat__cp_atypical`, `cat__cp_other`.

3. **Why**  
   Classifiers need numbers, not strings.

4. **Simple terms**  
   Each category gets its own yes/no switch.

---

### 3.5 Label encoding (target)

1. **Definition**  
   If the target is text, map classes to integers with `LabelEncoder`.

2. **Example**  
   `Approved` → 0, `Rejected` → 1 (order depends on sorted unique labels).

3. **Why**  
   sklearn classifiers expect numeric `y`.

4. **Simple terms**  
   Give each class a number ID.

---

### 3.6 Stratified split (when possible)

1. **Definition**  
   Keep similar class proportions in train and test; ensure each class can appear in the test set when data is tiny.

2. **Example**  
   90% healthy / 10% disease in full data → roughly the same ratio in test.

3. **Why**  
   Avoids a test set that accidentally has only one class.

4. **Simple terms**  
   Don’t put all the rare cases only in training (or only in test).

---

## 4. Models

All three are **classifiers**. Each exposes `train` + `evaluate` (Accuracy, Precision, Recall, F1 — weighted averages for multi-class).

---

### 4.1 Logistic Regression

1. **Definition**  
   Linear model: weighted sum of features → logistic (sigmoid / softmax) → class probabilities. Here: `LogisticRegression(max_iter=1000)`.

2. **Example**  
   `score = 0.8·chol_scaled + (−0.3)·age_scaled + …` → P(disease) = 1 / (1 + e^(−score)). If P > 0.5 (binary default), predict disease.

3. **Why we include it**  
   Strong, simple baseline; usually easier to interpret coefficients. Good contrast vs trees for HII (different SHAP path: Permutation SHAP).

4. **Simple terms**  
   Draw a soft straight boundary in feature space and read probabilities from which side you’re on.

---

### 4.2 Decision Tree

1. **Definition**  
   Hierarchical if/else rules on features. Depth limited: `max_depth=5`.

2. **Example**  
   `if chol > 240 → if age > 55 → disease else no disease else …`

3. **Why we include it**  
   Naturally “rule-like”; TreeExplainer applies. Can overfit if deep — depth cap keeps it smaller.

4. **Simple terms**  
   A flowchart of questions until you reach a leaf prediction.

---

### 4.3 Random Forest

1. **Definition**  
   Ensemble of many decision trees (`n_estimators=100`, `max_depth=10`); final class by majority / averaged probabilities.

2. **Example**  
   100 trees vote; 72 say disease → predict disease (with probability ≈ 0.72).

3. **Why we include it**  
   Usually stronger accuracy than a single tree; still tree-SHAP friendly. Useful accuracy-vs-HII trade-off comparisons.

4. **Simple terms**  
   Ask many slightly different trees and go with the crowd.

---

## 5. Prediction performance metrics

Computed on the **test set** via sklearn. Multi-class uses `average="weighted"`.

---

### 5.1 Accuracy

1. **Definition**  
   Fraction of predictions that match the true label:  
   `(correct predictions) / (total predictions)`.

2. **Example**  
   180 correct out of 200 → Accuracy = 0.90.

3. **Why we calculated it**  
   Quick overall “how often is the model right?” for model comparison.

4. **Simple terms**  
   Percentage of test rows the model got right.

**Caveat:** With imbalanced classes, high accuracy can hide bad minority-class performance — hence Precision/Recall/F1.

---

### 5.2 Precision (weighted)

1. **Definition**  
   Among rows predicted as class C, what fraction really are C. Weighted = average across classes weighted by support.

2. **Example**  
   Predicted disease for 50 people; 40 truly have disease → Precision_disease = 40/50 = 0.80.

3. **Why we calculated it**  
   Measures **false alarm** cost: “when we say yes, how often are we right?”

4. **Simple terms**  
   Of all the “disease” alarms, how many were real?

---

### 5.3 Recall (weighted)

1. **Definition**  
   Among rows that truly are class C, what fraction did we catch. Weighted across classes.

2. **Example**  
   60 real disease cases; model finds 40 → Recall_disease = 40/60 ≈ 0.67.

3. **Why we calculated it**  
   Measures **missed cases**: “when truth is yes, how often do we catch it?”

4. **Simple terms**  
   Of all sick patients, what share did we detect?

---

### 5.4 F1 (weighted)

1. **Definition**  
   Harmonic mean of Precision and Recall:  
   `F1 = 2 · (Precision · Recall) / (Precision + Recall)`.

2. **Example**  
   P=0.80, R=0.67 → F1 ≈ 0.73.

3. **Why we calculated it**  
   Single number balancing “don’t cry wolf” and “don’t miss cases.”

4. **Simple terms**  
   A fair middle ground between Precision and Recall — neither can be terrible without pulling F1 down.

---

## 6. Explainability metrics (per application, then averaged)

For each sampled test row (`n_apps`, default 20):

| Symbol | Name | Sources |
|--------|------|---------|
| **C** | Consistency | SHAP vs LIME **signs** on top-K |
| **S** | Stability | SHAP vs SHAP after tiny noise |
| **F** | Fidelity | LIME local R² clipped to [0,1] |
| **A** | Feature Agreement | Jaccard of top-K SHAP vs LIME sets |
| **HII** | Hybrid Interpretability Index | Weighted average of C,S,F,A |

Default weights: **0.25 each** (`config.yaml` → `hii.weights`).

---

### 6.1 Consistency (C)

1. **Definition**  
   Fraction of the top-K **nonzero SHAP** features whose **sign** matches LIME’s sign for that feature (LIME must also be nonzero).  
   `C = (# matching signs) / (# top-K SHAP features)`.

2. **Example**  
   Top-2 SHAP features: age +0.5, chol −0.4.  
   LIME: age +0.3, chol −0.2 → both signs match → **C = 1.0**.  
   If LIME has chol **+0.2** → one mismatch → **C = 0.5**.

3. **Why we calculated it**  
   If two independent explainers disagree on *direction* (helps vs hurts the class), the explanation story is unreliable.

4. **Simple terms**  
   “Do SHAP and LIME agree on whether each important feature pushes *toward* or *away from* the class?”

Code: `metrics/consistency.py`.

---

### 6.2 Stability (S)

1. **Definition**  
   Mean **cosine similarity** between the row’s SHAP vector and SHAP vectors of slightly perturbed copies of that row. Result clipped to [0, 1].  
   Perturbation: Gaussian noise with `perturb_scale` (default 0.05) on `num__*` features (`n_perturbations` times, default 2).

2. **Example**  
   SHAP_base ≈ SHAP_perturbed → cosine ≈ 0.99 → **S ≈ 0.99**.  
   If noise flips attributions wildly → cosine low → **S** low.

3. **Why we calculated it**  
   A trustworthy explanation should not completely change when the input jitters by a tiny amount (sensor noise, rounding, etc.).

4. **Simple terms**  
   Nudge the inputs a little — does the explanation story stay roughly the same?

Cosine similarity reminder: 1 = same direction, 0 = orthogonal, negative = opposite (clipped so S stays in [0,1]).

Code: `metrics/stability.py`.

---

### 6.3 Fidelity (F)

1. **Definition**  
   `F = clip(LIME.score, 0, 1)` where `LIME.score` is the local weighted **R²** of LIME’s surrogate linear model around that instance.

2. **Example**  
   LIME R² = 0.85 → **F = 0.85**.  
   R² = 1.5 (numerical quirk) → **F = 1.0**.  
   Bad local fit R² = −0.2 → **F = 0.0** after clip.

3. **Why we calculated it**  
   LIME’s coefficients only mean something if the local linear model actually fits the black box nearby. Low F → don’t trust that local story.

4. **Simple terms**  
   “Was LIME’s simple local model a good imitation of the real model around this point?”

Code: `metrics/fidelity.py`.

---

### 6.4 Feature Agreement (A)

1. **Definition**  
   **Jaccard similarity** between the sets of top-K nonzero SHAP features and top-K nonzero LIME features:  
   `A = |SHAP_set ∩ LIME_set| / |SHAP_set ∪ LIME_set|`.

2. **Example**  
   SHAP top-3 indices: {0, 2, 5}. LIME top-3: {0, 2, 7}.  
   Intersection = {0,2} (size 2), union = {0,2,5,7} (size 4) → **A = 2/4 = 0.5**.

3. **Why we calculated it**  
   Even if signs agree on overlap, the *which features matter* lists can differ. A measures overlap of those “important feature” shortlists.

4. **Simple terms**  
   “Do SHAP and LIME point at mostly the same important features?” (ignores sign; that’s Consistency’s job).

Code: `metrics/feature_agreement.py`.

---

### 6.5 Hybrid Interpretability Index (HII)

1. **Definition**  
   Weighted average of the four metrics:  
   ```
   HII = (w_C·C + w_S·S + w_F·F + w_A·A) / (w_C + w_S + w_F + w_A)
   ```  
   Default weights all 0.25 → simple mean of C, S, F, A. Rounded to 4 decimals.

2. **Example**  
   C=0.80, S=0.90, F=0.70, A=0.60 →  
   HII = (0.80+0.90+0.70+0.60)/4 = **0.75**.

3. **Why we calculated it**  
   One comparable score so models can be ranked on **explanation quality**, next to accuracy. Supports trade-off language: best accuracy vs best HII.

4. **Simple terms**  
   A report-card average for “how good are the explanations?” — agreement across methods, stability under noise, and local-model fit.

**Aggregation:**

- Per application: compute C,S,F,A → HII for that row.  
- Per model: mean of C,S,F,A across `n_apps`, then recompute HII from those means.  
- Comparison table sorted by **HII descending**.

Code: `hii/hybrid_index.py`, orchestration in `hii/evaluate.py`.

**Rough reading guide used in summaries:**

| HII | Level |
|-----|--------|
| ≥ 0.75 | high |
| ≥ 0.50 | moderate |
| < 0.50 | low |

---

## 7. Worked mini-walkthrough (one application)

Suppose one test patient, `top_k=2`, one perturbation for simplicity.

**SHAP vector** (features age, chol, thalach): `[+0.40, −0.30, +0.05]`  
**LIME vector:** `[+0.35, −0.25, +0.10]`  
**LIME.score (R²):** `0.82`  
**SHAP after tiny noise:** `[+0.39, −0.31, +0.04]`

**Consistency (C)**  
Top-2 SHAP by |value|: age (+), chol (−). LIME same signs → **C = 1.0**.

**Feature Agreement (A)**  
SHAP set {age, chol}, LIME set {age, chol} (top-2) → Jaccard = 1.0 → **A = 1.0**.

**Fidelity (F)**  
clip(0.82, 0, 1) → **F = 0.82**.

**Stability (S)**  
Cosine(SHAP, perturbed SHAP) ≈ 0.999 → **S ≈ 1.0**.

**HII**  
(1.0 + 1.0 + 0.82 + 1.0) / 4 = **0.955**.

Interpretation: for this row, explainers agree on who matters and in which direction, the local LIME fit is solid, and small noise barely changes SHAP — excellent local interpretability.

---

## 8. How the four metrics differ (intuition)

| Question you ask | Metric |
|------------------|--------|
| Same *direction* of effect? | Consistency (C) |
| Same *which features* shortlist? | Feature Agreement (A) |
| Explanation stable if input wiggles? | Stability (S) |
| Is LIME’s local line a good fit? | Fidelity (F) |
| Overall explanation quality? | HII |

You can have high **A** but low **C** (same features, opposite signs — bad).  
You can have high **C** and **A** but low **F** (agree, but LIME’s surrogate is a poor fit — still cautious).  
You can have high C/A/F but low **S** (pretty story that falls apart under noise).

---

## 9. Config knobs that change results

From `config/config.yaml`:

| Key | Role |
|-----|------|
| `preprocessing.test_size` | Fraction held out if no separate test file |
| `preprocessing.handle_missing` | Numeric imputer strategy (e.g. median) |
| `models.*` | Which of LR / RF / DT to run |
| `explainability.n_apps` | How many test rows get full HII |
| `explainability.top_k` | Size of top feature sets for C and A |
| `explainability.n_perturbations` | Perturbed copies per row for S |
| `explainability.perturb_scale` | Noise std for stability |
| `explainability.shap_evals` | Max evals for Permutation SHAP (auto ≥ 2·n_features+1) |
| `explainability.lime_num_samples` | Neighborhood size for LIME |
| `explainability.background_size` | Background rows for SHAP masker |
| `explainability.positive_class` | Preferred class name for attributions |
| `hii.weights` | Relative importance of C, S, F, A |

---

## 10. Outputs of a run (CLI)

- `data/processed/report.csv` — model comparison (Accuracy…F1 + C,S,F,A,HII).  
- `data/processed/report.png` — comparison figure.  
- `*_per_application.csv` — row-level C,S,F,A,HII, prediction, probability.  
- Console summary: best accuracy, best HII, trade-offs, weakest component per model, shared top SHAP drivers.

---

## 11. Module map (for studying the code)

| Path | Responsibility |
|------|----------------|
| `main.py` | Pipeline orchestration |
| `preprocessing/preprocess.py` | Load, analyze, impute, encode, scale, split |
| `models/*.py` | Train + prediction metrics |
| `hii/evaluate.py` | Per-app SHAP/LIME + C/S/F/A/HII |
| `hii/hybrid_index.py` | Weighted HII |
| `metrics/*.py` | Individual metric formulas |
| `explainability/*.py` | Standalone SHAP/LIME helpers (importance-style) |
| `evaluation/model_comparison.py` | Table + text summary |
| `config/config.yaml` | Dataset, models, explainability, HII weights |

---

## 12. One-page cheat sheet

- **Goal:** Compare classifiers on **accuracy** and **explanation quality (HII)**.  
- **Explainers:** SHAP (primary attributions) + LIME (second opinion + local R²).  
- **C** = sign agreement on top features.  
- **A** = overlap of top feature sets (Jaccard).  
- **S** = SHAP stays similar after tiny noise (cosine).  
- **F** = LIME local R² quality.  
- **HII** = average (or weighted average) of C,S,F,A.  
- **Models:** Logistic Regression, Decision Tree, Random Forest.  
- **UI:** Streamlit only for dashboard / web server.

---

*These notes match the implementation in this repository (metrics formulas, HII weights, SHAP/LIME wiring in `hii/evaluate.py`). If config weights or `top_k` change, numerical examples change but definitions stay the same.*
