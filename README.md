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

## Quick Start — Run Commands (Modal)

**Lưu ý quan trọng về dấu gạch ngang (`-`) trên Modal:**

- Modal tự động chuyển đổi các tham số gạch dưới (`_`) trong file Python thành dấu gạch ngang (`-`) trên command line. Do đó, TẤT CẢ các cờ khi gõ lệnh đều phải dùng dấu `-` (ví dụ: `--dataset-name`, `--enable-pseudo-labeling`).
- Đối với các tham số kiểu Boolean (Đúng/Sai), Modal KHÔNG nhận giá trị `true` hay `false` đi kèm. Để bật cờ, bạn chỉ cần gọi tên cờ đó: `--enable-pseudo-labeling` (tuyệt đối không viết thêm chữ `true` ở sau).

Defaults đã đặt sẵn trong `modal_train.py` (không cần gõ lại): `epochs=200`, `batch_size=1024`, `imb_ratio=100`, `vis_freq=10`; pseudo: `pseudo_mode=2`, `confidence_threshold=0.9`, `pseudo_top_ratio=0.8`, `max_samples_per_class=500`, `pseudo_update_freq=10`, `max_pseudo_iterations=20`.

### Chạy detached (khỏi sợ tắt máy)

`modal run` thường stream log từ máy local — tắt laptop là run bị cancel. Dùng entrypoint `launch` để spawn detached:

```bash
# --- CIFAR-100 ---
# Baseline (Tắt pseudo - chạy giống hệt bản gốc)
modal run -d modal_train.py::launch --dataset-name cifar100
# Pseudo labeling mode 1 ý tưởng thầy
modal run -d modal_train.py::launch --dataset-name cifar100 --enable-pseudo-labeling
# Pseudo labeling mode 2 + threshold riêng
modal run -d modal_train.py::launch --dataset-name cifar100 --enable-pseudo-labeling --pseudo-mode 2 --confidence-threshold 0.95

# --- CIFAR-10 ---
# Baseline (Tắt pseudo - chạy giống hệt bản gốc)
modal run -d modal_train.py::launch --dataset-name cifar10
# Pseudo labeling mode 1 ý tưởng thầy
modal run -d modal_train.py::launch --dataset-name cifar10 --enable-pseudo-labeling
# Pseudo labeling mode 2
modal run -d modal_train.py::launch --dataset-name cifar10 --enable-pseudo-labeling --pseudo-mode 2 --confidence-threshold 0.75

# --- CUB-200 ---
# Baseline (Tắt pseudo - chạy giống hệt bản gốc)
modal run -d modal_train.py::launch --dataset-name cub200
# Pseudo labeling mode 1 ý tưởng thầy
modal run -d modal_train.py::launch --dataset-name cub200 --enable-pseudo-labeling
# Pseudo labeling mode 2
modal run -d modal_train.py::launch --dataset-name cub200 --enable-pseudo-labeling --pseudo-mode 2 --confidence-threshold 0.75

# Theo dõi / quản lý sau khi spawn
modal app logs bacon-train      # attach xem log, Ctrl+C thoát KHÔNG kill run
modal app list                  # trạng thái apps

# Lấy kết quả (chart, final_report) về máy khi xong
modal volume get bacon-storage "outputs/experiments/<tên_experiment>" dev_outputs/
```

### Chạy kèm download tự động (ngồi chờ trên máy)

```bash
modal run modal_train.py --dataset-name cifar10
modal run modal_train.py --dataset-name cifar100 --enable-pseudo-labeling
modal run modal_train.py --dataset-name cub200 --enable-pseudo-labeling
modal run modal_train.py --dataset-name imagenet100
```

Xong tự tải chart/final report vào `dev_outputs/<experiment>/`.

### Ghi đè thông số khác

```bash
modal run -d modal_train.py::launch --dataset-name cifar10 --epochs 100 --batch-size 512 \
    --lr 0.01 --exp-name-suffix test1
# Thông số ngoài danh sách -> truyền qua extra_args (dấu gạch ngang):
modal run -d modal_train.py::launch --dataset-name cifar100 \
    --extra_args "--sup-weight 0.35 --vis-freq 5"
```

---

## TensorBoard

```bash
# View training progress
tensorboard --logdir=/bacon-storage/outputs/experiments
```

## Pseudo Labeling Parameters

Đặt trong `modal_train.py` — hàm `train()` / `launch()` (sửa default ngay đầu hàm). Giá trị được chuyển thành flag gạch-ngang cho `model/bacon.py`.

| Modal param (`_`)          | Flag tương ứng (`-`)    | Mô tả                            | Default |
| ---------------------------- | ---------------------------- | ---------------------------------- | ------- |
| `--enable-pseudo-labeling` | `--enable-pseudo-labeling` | Bật pseudo labeling               | False   |
| `--pseudo-mode`            | `--pseudo-mode`            | 1 = best class, 2 = high conf      | 2       |
| `--confidence-threshold`   | `--confidence-threshold`   | Ngưỡng confidence (mode 2)       | 0.9     |
| `--pseudo-top-ratio`       | `--pseudo-top-ratio`       | Mode 1: giữ top-t% conf cao nhất | 0.8     |
| `--max-samples-per-class`  | `--max-samples-per-class`  | Cap mẫu mỗi class mỗi iteration | 500     |
| `--pseudo-update-freq`     | `--pseudo-update-freq`     | Chu kỳ update (epochs)            | 10      |
| `--max-pseudo-iterations`  | `--max-pseudo-iterations`  | Số lần inject tối đa           | 20      |

Bias-evidence outputs (mặc định bật khi có pseudo labeling): `[PSEUDO AUDIT]` trong log (% novel bị nuốt, % sai class cũ) + chart `visualizations/per_class_acc_epoch{N}.png` (mỗi 10 epoch) và `visualizations/bias_evidence_iter{N}.png` (mỗi iteration).

## Training locally (không Modal)

```bash
bash run_cifar10.sh     # flag gạch-ngang, xem file để biết đầy đủ thông số
```
