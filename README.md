# Towards Distribution-Agnostic Generalized Category Discovery.

**Forked from:** [NeurIPS 2023] [BaCon](https://arxiv.org/abs/2310.01376) — official implementation. Repo gốc: [JianhongBai/BaCon](https://github.com/JianhongBai/BaCon).

## Introduction

Data imbalance and open-ended distribution are two intrinsic characteristics of the real visual world. Though encouraging progress has been made in tackling each challenge separately, few works dedicated to combining them towards real-world scenarios. While several previous works have focused on classifying close-set samples and detecting open-set samples during testing, it's still essential to be able to classify unknown subjects as human beings. In this paper, we formally define a more realistic task as distribution-agnostic generalized category discovery (DA-GCD): generating fine-grained predictions for both close- and open-set classes in a long-tailed open-world setting. To tackle the challenging problem, we propose a Self-**Ba**lanced **Co**-Advice co**n**trastive framework (BaCon), which consists of a contrastive-learning branch and a pseudo-labeling branch, working collaboratively to provide interactive supervision to resolve the DA-GCD task. In particular, the contrastive-learning branch provides reliable distribution estimation to regularize the predictions of the pseudo-labeling branch, which in turn guides contrastive learning through self-balanced knowledge transfer and a proposed novel contrastive loss. We compare BaCon with state-of-the-art methods from two closely related fields: imbalanced semi-supervised learning and generalized category discovery. The effectiveness of BaCon is demonstrated with superior performance over all baselines and comprehensive analysis across various datasets.

## Method

<div align=center>
<img src="pipeline.png" width="800" >
</div>

Overview of the self-balanced co-advice contrastive framework (BaCon).

## Environment

Requirements:

```
loguru
numpy
pandas
scikit_learn
scipy
torch==1.10.0
torchvision==0.11.1
tqdm
```

## Data

We provide the specific train split of CIFAR-10 and CIFAR-100 with different imbalance ratios, please refer to ``data_uq_idxs`` for details. We also provide the source code in ``data/imagenet.py`` for splitting data to the proposed DA-GCD setting.

## Pretrained models downloading

[CIFAR-10](https://drive.google.com/file/d/1OFcLeDK1HUD6N9pQuRHp_TJ0CaTybWxJ/view?usp=sharing)

[CIFAR-100](https://drive.google.com/file/d/1pbZukDXHqwUtvfP24lqlvKT5V2UHfOGk/view?usp=sharing)

[ImageNet-100](https://drive.google.com/file/d/1L_sL4B7WhyPeqna5hYSpP2k52FKbDxQl/view?usp=sharing)

## Training

### CIFAR-10

```
bash run_cifar10.sh
```

### CIFAR-100

```
bash run_cifar100.sh
```

### ImageNet-100

```
bash run_imagenet100.sh
```

## Acknowledgments

---

## Run Plan — Ablation imb10 (BaCon track, 7 lệnh)

**Trạng thái:** cấu hình Full (dual + teacher + prototype + tail-aware) đã chạy xong trên cả 3 dataset với exp-name `*_Teacher_C3_imb10` (CUB / Cars / Cifar100) — dùng làm mốc so sánh. Plan này là phần ablation còn lại.

**Mapping cấu hình → flag** (đối chiếu log run `*Teacher_C3_imb10`):

| # | Cấu hình | Bật | Tắt | So sánh |
| --- | --- | --- | --- | --- |
| 1 | Full(wo Tail) | dual D1+D2, teacher EMA, prototype (slots M=3, fused CE, gate-reg, EMA proto) | Lcap — tail-aware capacity (eq 6-8) | Full → #1 = đóng góp riêng của Lcap |
| 2 | Full(wo Tail, wo Dual) | teacher EMA, prototype | Lcap + dual (D1+D2) | #1 → #2 = đóng góp dual |
| 3 | Full(wo Dual) | teacher EMA, prototype, Lcap | dual (D1+D2) | Full → #3 = đóng góp dual (giữ Lcap) |

Tail-aware (Lcap) nằm trong nhánh prototype nên không tách được — không tồn tại cấu hình "dual + tail-aware mà không prototype". Cấu hình 1/2 chạy trên cả 3 dataset; cấu hình 3 chỉ chạy trên Cifar100. So sánh phụ: #2 vs #3 = đóng góp Lcap trong môi trường không dual.

Chạy server thường từ thư mục gốc repo (`--labeled-classes`: cub200=100, stanford_cars=98, cifar100=80; đổi server thì set `CUB_ROOT`/`CARS_ROOT` đúng):

```bash
# --- CUB-200 (100 known / 100 novel): cấu hình 1 — Full(wo Tail) ---
python -m model.bacon --dataset-name cub200 --labeled-classes 100 --imb-ratio 10 --epochs 200 --stop-epoch 200 --batch-size 256 --num-workers 8 --backbone dinov2_vitb14 --est-freq 10 --ce-warmup 1 --alpha 0 --beta 0.5 --warmup-teacher-temp-epochs 50 --vis-freq 10 --early-stop-patience 50 --exp-name cub200_Teacher_C3_imb10_noTailcap --enable-pseudo-labeling --pseudo-mode 1 --pseudo-warmup-epoch 30 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --use-momentum-teacher --teacher-m0 0.996 --ablate-adaptive-capacity

# --- CUB-200: cấu hình 2 — Full(wo Tail, wo Dual) ---
python -m model.bacon --dataset-name cub200 --labeled-classes 100 --imb-ratio 10 --epochs 200 --stop-epoch 200 --batch-size 256 --num-workers 8 --backbone dinov2_vitb14 --est-freq 10 --ce-warmup 1 --alpha 0 --beta 0.5 --warmup-teacher-temp-epochs 50 --vis-freq 10 --early-stop-patience 50 --exp-name cub200_ProtoTeacher_noTailcap --use-parts --num-slots 3 --use-momentum-teacher --teacher-m0 0.996 --ablate-adaptive-capacity

# --- Stanford Cars (98 known / 98 novel): cấu hình 1 — Full(wo Tail) ---
python -m model.bacon --dataset-name stanford_cars --labeled-classes 98 --imb-ratio 10 --epochs 200 --stop-epoch 200 --batch-size 256 --num-workers 8 --backbone dinov2_vitb14 --est-freq 10 --ce-warmup 1 --alpha 0 --beta 0.5 --warmup-teacher-temp-epochs 50 --vis-freq 10 --early-stop-patience 50 --exp-name cars_Teacher_C3_imb10_noTailcap --enable-pseudo-labeling --pseudo-mode 1 --pseudo-warmup-epoch 30 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --use-momentum-teacher --teacher-m0 0.996 --ablate-adaptive-capacity

# --- Stanford Cars: cấu hình 2 — Full(wo Tail, wo Dual) ---
python -m model.bacon --dataset-name stanford_cars --labeled-classes 98 --imb-ratio 10 --epochs 200 --stop-epoch 200 --batch-size 256 --num-workers 8 --backbone dinov2_vitb14 --est-freq 10 --ce-warmup 1 --alpha 0 --beta 0.5 --warmup-teacher-temp-epochs 50 --vis-freq 10 --early-stop-patience 50 --exp-name cars_ProtoTeacher_noTailcap --use-parts --num-slots 3 --use-momentum-teacher --teacher-m0 0.996 --ablate-adaptive-capacity

# --- Cifar100 (80 known / 20 novel): cấu hình 1 — Full(wo Tail) ---
python -m model.bacon --dataset-name cifar100 --labeled-classes 80 --imb-ratio 10 --epochs 200 --stop-epoch 200 --batch-size 256 --num-workers 8 --backbone dinov2_vitb14 --est-freq 10 --ce-warmup 1 --alpha 0 --beta 0.5 --warmup-teacher-temp-epochs 50 --vis-freq 10 --early-stop-patience 50 --exp-name cifar100_Teacher_C3_imb10_noTailcap --enable-pseudo-labeling --pseudo-mode 1 --pseudo-warmup-epoch 30 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --use-momentum-teacher --teacher-m0 0.996 --ablate-adaptive-capacity

# --- Cifar100: cấu hình 2 — Full(wo Tail, wo Dual) ---
python -m model.bacon --dataset-name cifar100 --labeled-classes 80 --imb-ratio 10 --epochs 200 --stop-epoch 200 --batch-size 256 --num-workers 8 --backbone dinov2_vitb14 --est-freq 10 --ce-warmup 1 --alpha 0 --beta 0.5 --warmup-teacher-temp-epochs 50 --vis-freq 10 --early-stop-patience 50 --exp-name cifar100_ProtoTeacher_noTailcap --use-parts --num-slots 3 --use-momentum-teacher --teacher-m0 0.996 --ablate-adaptive-capacity

# --- Cifar100: cấu hình 3 — Full(wo Dual), giữ Lcap ---
python -m model.bacon --dataset-name cifar100 --labeled-classes 80 --imb-ratio 10 --epochs 200 --stop-epoch 200 --batch-size 256 --num-workers 8 --backbone dinov2_vitb14 --est-freq 10 --ce-warmup 1 --alpha 0 --beta 0.5 --warmup-teacher-temp-epochs 50 --vis-freq 10 --early-stop-patience 50 --exp-name cifar100_Teacher_C3_imb10_noDual --enable-pseudo-labeling --pseudo-mode 1 --pseudo-warmup-epoch 30 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --use-momentum-teacher --teacher-m0 0.996
```

Kiểm tra khởi chạy: cả 3 cấu hình đều giữ `--use-parts` nên log đầu phải in dòng `[AdaPart] Ablations: ...` — cấu hình 1/2 phải hiện `adaptive_cap=False`, cấu hình 3 hiện `adaptive_cap=True`.

---

## Quick Start — SimGCD Rerun (4 cấu hình × 3 datasets, 12 lệnh Modal + 12 lệnh server)

> Nguồn flag đối chiếu: `gcd-hosts/SimGCD/train.py` (flag **gạch dưới** `_`) + `gcd-hosts/modal_train_simgcd.py` (CLI Modal **gạch ngang** `-`, boolean chỉ gọi tên, không kèm `true`).
> Tên dataset SimGCD: `cub` (CUB-200), `scars` (Stanford Cars — chữ `car` chỉ là tên gọi tắt), `aircraft` (FGVC-Aircraft). Khác với track BaCon (`cub200` / `stanford-cars` / `fgvc-aircraft`).

**Mapping 4 cấu hình → flag** (backbone chung `dinov2_vitb14`, `epochs=200`, `seed=0`):

| # | Cấu hình                                                                          | Flag thêm vào                                                            |
| - | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| A | SimGCD baseline                                                                     | (không thêm gì)                                                         |
| B | Dual pseudo-only (known Mode-1 + novel,**không** parts/teacher)              | `--enable_pseudo_labeling --pseudo_mode 1 ... --enable_novel_pseudo ...` |
| C | Multi-prototype + teacher-only (3 slots/lớp + teacher EMA,**không** pseudo) | `--use_parts --num_slots 3 --use_momentum_teacher`                       |
| D | Dual + multi-prototype + teacher (full ⭐)                                          | cả hai cụm trên                                                         |

**Giả định chung (đã gõ rõ trong lệnh để không phụ thuộc default code):** `batch_size=128`, `grad_from_block=11`, `sup_weight=0.35`, `transform=imagenet`, `eval=v2`, `warmup_teacher_temp=0.07/0.04/30`, `use_ssb_splits`; `memax_weight`: CUB=`2`, Cars/Aircraft=`1` (theo `scripts/run_*.sh`); pseudo-known Mode-1 (warmup 30, freq 10, iters 20, top-ratio 0.8, cap 500); novel warmup 50 / freq 10 / **max-iters 6 / cap 200** (khác default code là 2×100 nên phải gõ rõ); parts `num_slots=3, part_lambda=0.5, tau_c=0.1`; teacher `m0=0.996`. Exp-name Modal tự sinh `simgcd-<data>-<variant>-dinov2b14-seed0-<stamp>-<suffix>`.

### A. 12 lệnh Modal (chạy từ thư mục `gcd-hosts/SimGCD/` — file `modal_train_simgcd.py` đã có bản copy trong đó — detached với `-d`)

```bash
# Lần đầu chạy aircraft trên Modal: nạp data 1 lần (idempotent, scars đã có sẵn)
modal run modal_train_simgcd.py::fetch_aircraft

# --- CUB-200 (100 known / 100 novel) ---
# A1. Baseline
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimBase
# B1. Dual-only (known Mode-1 + novel)
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimDual --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10
# C1. Multi-prototype (3 slots) + teacher-only, không pseudo
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimProtoT --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
# D1. Full: dual + multi-prototype + teacher
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimFull --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996

# --- Stanford Cars = scars (98 known / 98 novel) ---
# A2. Baseline
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimBase
# B2. Dual-only (known Mode-1 + novel)
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimDual --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10
# C2. Multi-prototype (3 slots) + teacher-only, không pseudo
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimProtoT --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
# D2. Full: dual + multi-prototype + teacher
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimFull --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996

# --- FGVC-Aircraft (50 known / 50 novel theo SSB split SimGCD) ---
# A3. Baseline
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimBase
# B3. Dual-only (known Mode-1 + novel)
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimDual --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10
# C3. Multi-prototype (3 slots) + teacher-only, không pseudo
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimProtoT --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
# D3. Full: dual + multi-prototype + teacher
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimFull --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
```

Theo dõi log: `modal app logs simgcd-train` (launcher có preflight check flag miễn phí trước khi submit).

### B. 12 lệnh server thường (chạy từ thư mục `SimGCD/`, flag gạch dưới `_`)

```bash
# Sửa 3 đường data cho đúng server trước khi chạy (CUB = PARENT của CUB_200_2011;
# scars = thư mục chứa devkit/cars_train...; aircraft = thư mục fgvc-aircraft-2013b/ chứa data/images):
export SIMGCD_CUB_ROOT=/data/cub
export SIMGCD_SCARS_ROOT=/data/cars
export SIMGCD_AIRCRAFT_ROOT=/data/aircraft/fgvc-aircraft-2013b

# --- CUB-200 ---
# A1. Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimBase
# B1. Dual-only (known Mode-1 + novel)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimDual --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10
# C1. Multi-prototype (3 slots) + teacher-only, không pseudo
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimProtoT --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
# D1. Full: dual + multi-prototype + teacher
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimFull --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10 --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996

# --- Stanford Cars (dataset_name=scars) ---
# A2. Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimBase
# B2. Dual-only (known Mode-1 + novel)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimDual --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10
# C2. Multi-prototype (3 slots) + teacher-only, không pseudo
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimProtoT --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
# D2. Full: dual + multi-prototype + teacher
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimFull --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10 --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996

# --- FGVC-Aircraft (dataset_name=aircraft) ---
# A3. Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimBase
# B3. Dual-only (known Mode-1 + novel)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimDual --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10
# C3. Multi-prototype (3 slots) + teacher-only, không pseudo
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimProtoT --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
# D3. Full: dual + multi-prototype + teacher
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimFull --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10 --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
```

Lưu ý server Windows (PowerShell): thay `export X=...` bằng `$env:X="..."` và bỏ prefix `CUDA_VISIBLE_DEVICES=0` (chọn GPU bằng `--` hoặc biến môi trường tương đương).
