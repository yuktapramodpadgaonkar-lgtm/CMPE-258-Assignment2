# CMPE-258 — Assignment 2 (Option 1: Training Optimization)

**Face mask detection with Ultralytics YOLOv8**

| | |
|---|---|
| **Course** | CMPE-258 |
| **Option** | Option 1 — Training optimization for 2D object detection |
| **Author** | *Yukta Pramod Padgaonkar* |
| **ID** | *018464615* |

---

## 1. Objective

This work optimizes **training** for a **2D object detector** on a face-mask dataset. The assignment requires:

- A **baseline** model and training setup  
- One or more **improved** approaches (architecture and/or training strategy)  
- A **clear description** of what changed, why, and how performance was affected  
- **COCO-style evaluation** (precision, recall, mAP@0.5, mAP@0.5:0.95) **before and after** changes  

We use **Ultralytics YOLOv8** on **Google Colab** (GPU), with metrics produced by `model.val()` on a **fixed validation split** and the same `data.yaml` for all runs.

---

## 2. Dataset and preprocessing

**Source:** [Face Mask Detection (Kaggle — andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) — images with **Pascal VOC XML** bounding boxes.

**Classes (3):**

| ID | Class name |
|----|------------|
| 0 | `with_mask` |
| 1 | `without_mask` |
| 2 | `mask_weared_incorrect` |

**Preprocessing:** XML annotations were converted to **YOLO format** (`class cx cy w h`, normalized). The data were split into **train** and **validation** sets; all reported metrics use the **same validation set** (128 images, 494 instances in the evaluation run summarized below) defined in `mask_yolo/data.yaml`.

**Rationale:** A single frozen val split ensures that differences across experiments reflect **model and training choices**, not a change in test data.

---

## 3. Baseline setup

**Model:** `yolov8n.pt` (YOLOv8 **nano** — small, fast baseline).

**Training (summary):**

- Image size **640**, batch **16**, **60** epochs, early stopping **patience 15**  
- Default-oriented augmentations (e.g. flips, HSV, mosaic as in Ultralytics defaults)  
- Optimizer left at **auto** (typical SGD-style schedule in Ultralytics)  
- **No** cosine LR, **no** extra geometric jitter (`degrees=0`), **no** mixup in the baseline recipe  

**Goal:** Establish a reproducible **lower bound** with a compact model and standard training, before changing schedule, augmentations, optimizer, or architecture.

---

## 4. Improved approaches

We group improvements into **(A) training strategy on YOLOv8n**, **(B) architecture scale**, and **(C) targeted ablations**.

### 4.1 Training strategy only — `improved_n_strategy` (YOLOv8n)

**Intent:** Improve generalization and calibration without increasing model size.

**Key changes (vs baseline):**

| Knob | Rationale |
|------|-----------|
| **AdamW** optimizer | Often stable for detection with default weight decay behavior. |
| **Cosine LR** (`cos_lr=True`) | Smooth decay can help late-epoch refinement. |
| **Warmup** (e.g. 3 epochs) | Reduces early instability after augmentation-heavy batches. |
| **Stronger geometry / color aug** | `degrees`, `translate`, `scale`; tuned HSV; **mixup** at modest rate to expose harder blends. |
| **Mosaic + `close_mosaic`** | Standard YOLO recipe; taper mosaic near end of training. |

**Observed trade-off:** Mean **precision** dropped vs baseline while **recall** and **mAP@0.5** rose — consistent with a detector that **fires more boxes** and catches more objects, including the difficult class (Section 6).

### 4.2 Architecture change — `improved_s_strategy` (YOLOv8s)

**Intent:** Satisfy the rubric’s “architecture via configuration” by moving from **nano** to **small** (`yolov8s.pt`), keeping the **same strategic** augmentations and optimizer style as the improved **n** run.

**Expectation:** More parameters (~11M vs ~3M) improve representational capacity; inference cost increases.

### 4.3 Longer training — `exp_6A_long_ep120_pat40`

**Intent:** The improved recipe may need **more time** to converge on val mAP, especially for the **rare** class.

**Changes:** **120** epochs, **patience 40** (same YOLOv8n + same augmentation/optimizer style as `improved_n_strategy`).

### 4.4 Input resolution — `exp_6B_imgsz704`, `exp_6C_imgsz768`

**Intent:** Test whether **higher resolution** helps small or thin face/mask structures.

**Settings:** Same recipe as improved strategy where possible; **batch size** reduced (e.g. 12 @ 704, 8 @ 768) to fit GPU memory.

**Cost:** Higher **latency** per image at inference (observed in Colab logs).

### 4.5 Optimizer comparison — `exp_6D_optimizer_sgd`, `exp_6E_optimizer_adamw_matched`

**Intent:** Compare **SGD** vs **AdamW** under a matched training budget.

**Note:** `exp_6E_optimizer_adamw_matched` produced **identical** validation metrics to `improved_n_strategy` in our run — indicating the same effective hyperparameter recipe (or duplicate experiment). We treat it as **confirmation** of the AdamW result rather than an independent second trial.

### 4.6 Oversampling rare-class examples — `exp_6F_oversample_rare_3x`

**Intent:** The class **`mask_weared_incorrect`** has **few** training and validation instances; oversampling **duplicates** rare-class examples in the **training** set (training `data.yaml` pointing at an oversampled layout) can encourage the network to pay more attention to that head.

**Evaluation:** All models were still validated with the **original** validation `data.yaml` so metrics remain **comparable**.

---

## 5. Evaluation protocol (COCO-style)

For **each** trained checkpoint (`runs/<run_name>/weights/best.pt`):

```text
metrics = YOLO(path).val(data=mask_yolo/data.yaml, split="val", verbose=True)
```

**Reported quantities (Ultralytics / COCO-style detection):**

- **P** — mean precision (box)  
- **R** — mean recall (box)  
- **mAP50** — mAP at IoU 0.50  
- **mAP50-95** — mAP averaged over IoU 0.50–0.95  

Per-class columns use the same definitions restricted to each class.

**Environment (logged):** Ultralytics **8.4.38**, PyTorch **2.10** + CUDA **12.8**, **Tesla T4** GPU.

---

## 6. Results

### 6.1 Overall metrics (validation)

All values from a single `val()` pass per run; numbers rounded for display.

| Run | P | R | mAP50 | mAP50-95 |
|-----|------|------|--------|----------|
| **baseline_n** | 0.901 | 0.717 | 0.784 | 0.546 |
| improved_n_strategy | 0.810 | 0.778 | 0.827 | 0.560 |
| **exp_6A_long_ep120_pat40** | **0.907** | **0.798** | **0.858** | **0.584** |
| exp_6B_imgsz704 | 0.854 | 0.726 | 0.806 | 0.537 |
| exp_6C_imgsz768 | 0.819 | 0.756 | 0.789 | 0.532 |
| exp_6D_optimizer_sgd | 0.817 | 0.792 | 0.832 | 0.569 |
| exp_6E_optimizer_adamw_matched | 0.810 | 0.778 | 0.827 | 0.560 |
| exp_6F_oversample_rare_3x | 0.816 | **0.812** | 0.796 | 0.541 |
| improved_s_strategy | 0.903 | 0.778 | 0.815 | 0.550 |

**Best overall:** **`exp_6A_long_ep120_pat40`** achieves the highest **mAP50** and **mAP50-95**, and the strongest **precision–recall balance** in the table.

### 6.2 Per-class mAP50 (validation)

| Run | with_mask | without_mask | mask_weared_incorrect |
|-----|-----------|--------------|------------------------|
| baseline_n | 0.975 | 0.860 | 0.516 |
| improved_n_strategy | 0.972 | 0.827 | 0.682 |
| **exp_6A_long_ep120_pat40** | 0.973 | 0.838 | **0.762** |
| exp_6B_imgsz704 | 0.974 | 0.865 | 0.579 |
| exp_6C_imgsz768 | 0.971 | 0.832 | 0.565 |
| exp_6D_optimizer_sgd | 0.975 | 0.854 | 0.666 |
| exp_6F_oversample_rare_3x | 0.966 | 0.837 | 0.586 |
| improved_s_strategy | 0.977 | 0.869 | 0.598 |

**Interpretation:**

- **`with_mask`** is near saturation for all runs (small relative spread).  
- **`mask_weared_incorrect`** is the **bottleneck** for baseline mAP; the largest **absolute** gains appear under **long training (6A)** and **strategy-focused** runs (e.g. improved **n**, **SGD**).  
- **`exp_6F`** achieves the **highest recall (0.812)** overall, consistent with oversampling encouraging sensitivity, while mean mAP50 is not the top entry — a useful **precision–recall trade-off** discussion.

### 6.3 Primary “before vs after” comparison (vs baseline)

Using **`baseline_n`** as “before”:

| Experiment | Δ mAP50 | Δ mAP50-95 | Comment |
|------------|---------|------------|---------|
| improved_n_strategy | +0.043 | +0.014 | Stronger recall; better rare class; lower mean P. |
| **exp_6A_long_ep120_pat40** | **+0.074** | **+0.038** | **Largest gain**; best rare-class mAP50. |
| improved_s_strategy | +0.031 | +0.004 | Architecture scale helps; 6A still leads on mAP. |

*(Exact deltas are in the notebook printout and in `coco_val_comparison.csv`.)*

---

## 7. Discussion

1. **Why baseline precision was high but recall lower**  
   The baseline favors **confident** predictions on dominant classes; it **under-detects** the rare **`mask_weared_incorrect`** cases, which drags mean recall and mAP.

2. **Why longer training (6A) wins**  
   With the same **n**-scale model, **more epochs** and **higher patience** allow optimization to continue while validation mAP improves — especially valuable for **hard, low-frequency** patterns.

3. **Why larger image size (6B/6C) did not beat 6A here**  
   Higher `imgsz` increases compute and can change effective batch statistics; on this split, **schedule length** mattered more than resolution alone.

4. **Duplicate metrics (6E vs improved_n_strategy)**  
   Report honestly: metrics are **bit-for-bit identical**; cite as **reproducibility** of the AdamW configuration or remove one row from the “main” table to avoid implying independent replication.

5. **Limitations**  
   Validation **per-class** estimates for **`mask_weared_incorrect`** are **high variance** (few instances). Reporting **overall mAP** alongside per-class tables is appropriate; claims should emphasize **consistent trends** across runs, not only point estimates.

---

## 8. Conclusion

We presented a **baseline YOLOv8n** detector on the Kaggle face-mask dataset and several **training-strategy** and **architecture** improvements evaluated with **identical COCO-style validation**. The strongest configuration in our experiments was **`exp_6A_long_ep120_pat40`**, improving **mAP50** from **0.784** to **0.858** and **mAP50-95** from **0.546** to **0.584** vs baseline, with the largest improvement on the **minority class** `mask_weared_incorrect`. **`improved_s_strategy`** demonstrates that **scaling width/depth** (YOLOv8s) also helps but did not exceed the **long-trained nano** model on this benchmark.

---

## 9. Artifacts submitted

| Artifact | Description |
|----------|-------------|
| Jupyter notebook | Full pipeline: data prep, training, `val()`, comparison tables, optional qualitative plots. |
| `coco_val_comparison.csv` | Machine-readable summary of all runs (same metrics as Section 6). |
| This report | Strategy description and interpretation of results. |
|zipped archive of the training `runs/` directory | (Ultralytics outputs: per-experiment folders with `weights/best.pt`, `results.csv`, plots, etc.). Use it to reproduce or inspect checkpoints and training logs alongside the notebook and report.

---

## 10. References

1. Ultralytics YOLOv8 — [https://docs.ultralytics.com](https://docs.ultralytics.com)  
2. Face Mask Detection dataset — Andrew Mvd, Kaggle (`andrewmvd/face-mask-detection`).  
3. Lin et al., *Microsoft COCO: Common Objects in Context* — COCO detection metrics (mAP).  

---

*End of report.*
