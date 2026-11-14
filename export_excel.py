"""=============================================================================
BACON BENCHMARK LOG PARSER & EXCEL / GOOGLE SHEETS EXPORTER
=============================================================================
- Trích xuất kết quả đánh giá song song trên:
    1. Tập UNLABELED TRAIN (Hỗ trợ Best Unlabeled Train & Final Epoch 200)
    2. Tập DISJOINT TEST   (Hỗ trợ Best Disjoint Test & Final Epoch 200)
- Tự động quét và cập nhật các log mới nhất từ D:\\data\\log bacon
- Tự động thay thế bản cũ khi có log mới hơn mà không cần phải xóa file thủ công
- Xử lý mượt mà khi file Excel đang mở (tự lưu file dự phòng tránh lỗi Permission)
- Tự động highlight: IN ĐẬM (Max - Tốt nhất) và GẠCH CHÂN (2nd - Tốt nhì)
- Phân nhóm cấu hình (BaCon x Datasets x Imb x 6 Configs) thành các bảng trực quan
- Xuất file Excel (.xlsx), CSV (.csv) và TSV (.tsv) cho cả Unlabeled Train và Disjoint Test
=============================================================================
"""

import os
import re
import json
import glob
import csv
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Thư mục log mặc định và đường dẫn xuất file
DEFAULT_LOG_DIR = r"D:\data\log bacon"
DEFAULT_OUTPUT_EXCEL = r"D:\data\log bacon\Bang_Ket_Qua_Thuc_Nghiem_BaCon.xlsx"

DEFAULT_CSV_UNL = r"D:\data\log bacon\bacon_unlabeled_train_for_googlesheet.csv"
DEFAULT_TSV_UNL = r"D:\data\log bacon\bacon_unlabeled_train_paste_ready.tsv"

DEFAULT_CSV_TEST = r"D:\data\log bacon\bacon_disjoint_test_for_googlesheet.csv"
DEFAULT_TSV_TEST = r"D:\data\log bacon\bacon_disjoint_test_paste_ready.tsv"

MODELS = [
    ("bacon", "BaCon"),
    ("simgcd", "SimGCD"),
]

DATASETS = [
    ("cub", "CUB-200-2011", ["cub", "cub200"]),
    ("car", "Stanford Cars", ["car", "cars", "scars", "stanford_cars"]),
    ("air", "FGVC Aircraft", ["air", "aircraft", "fgvc_aircraft"]),
]

DATASET_MAP = {
    "cub": "cub", "cub200": "cub",
    "car": "car", "cars": "car", "scars": "car", "stanford_cars": "car",
    "air": "air", "aircraft": "air", "fgvc_aircraft": "air",
}

IMB_RATIOS = ["imb20", "imb50", "imb100"]

CONFIGS = [
    ("Base", "Base", "Baseline chuẩn"),
    ("Pseudo", "Pseudo", "Baseline + Pseudo-labeling"),
    ("Prototype", "Prototype", "Baseline + Prototype Parts (Ablated)"),
    ("Prototype_Tailaware", "Prototype_Tailaware", "Baseline + Prototype + Tail-Aware"),
    ("Full_wo_Tail", "Full_wo_Tail", "Full Model w/o Tail-Aware"),
    ("Full", "Full", "ProtoTail Full Model"),
]

METRIC_KEYS = ["all", "old", "new", "h_score", "k_many", "k_med", "k_few", "u_many", "u_med", "u_few"]


def extract_timestamp(fname, mtime):
    """Trích xuất mốc thời gian từ tên file log (dạng ISO) hoặc lấy mtime của hệ thống."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
        except Exception:
            pass
    return datetime.fromtimestamp(mtime)


def normalize_canonical_key(run_name, fname):
    """Chuẩn hoá Run Name thành key chuẩn mực: {model}_{dataset}_{config}_{imb}."""
    combined = (run_name + " " + fname).lower().replace("-", "_")
    
    # 1. Model
    m = "bacon" if "bacon" in combined else ("simgcd" if "simgcd" in combined else "bacon")
    
    # 2. Dataset
    d = ""
    for k, v in DATASET_MAP.items():
        if f"_{k}_" in combined or f"_{k}" in combined or f"{k}_" in combined or f"-{k}-" in combined:
            d = v
            break

    # 3. Imbalance Ratio
    imb = ""
    for i in ["imb20", "imb50", "imb100", "imb10"]:
        if i in combined:
            imb = i
            break

    # 4. Config
    cfg = ""
    if "prototype_tailaware" in combined:
        cfg = "prototype_tailaware"
    elif "prototype" in combined:
        cfg = "prototype"
    elif "full_wo_tail" in combined:
        cfg = "full_wo_tail"
    elif "full" in combined:
        cfg = "full"
    elif "pseudo" in combined:
        cfg = "pseudo"
    elif "base" in combined:
        cfg = "base"

    if m and d and cfg and imb:
        return f"{m}_{d}_{cfg}_{imb}"
    return combined.strip()


def parse_single_log(filepath, metric_mode="best"):
    """
    Phân tích 1 file log và trích xuất chỉ số UNLABELED TRAIN và DISJOINT TEST.
    metric_mode:
        - 'best': Trích xuất kết quả tại Epoch có All Acc cao nhất (Best Unlabeled-Train & Best Disjoint-Test).
        - 'final': Trích xuất kết quả tại Epoch cuối cùng (Epoch 200 Paper Protocol).
    """
    fname = os.path.basename(filepath)
    stat = os.stat(filepath)
    file_dt = extract_timestamp(fname, stat.st_mtime)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[!] Lỗi khi đọc file {fname}: {e}")
        return None

    content = "".join(lines)

    # 1. Nhận diện nơi chạy (Kaggle Account / Server / Modal)
    account = ""
    acc_m = re.search(r"Kaggle Account:\s*(\S+)", content)
    if acc_m:
        account = f"Kaggle ({acc_m.group(1)})"
    elif "modal run" in content.lower() or "modal app" in content.lower():
        account = "Modal Cloud"
    elif "cuda_visible_devices" in content.lower() or "nohup" in content.lower():
        account = "Server Linux"
    else:
        prefix_m = re.match(r"(tam\d+)_", fname)
        if prefix_m:
            account = f"Kaggle ({prefix_m.group(1)})"
        else:
            account = "Server / Local"

    # 2. Nhận diện Run Name & Chuẩn hoá key
    rn_m = re.search(r"Run Name:\s*(\S+)", content)
    run_name = rn_m.group(1) if rn_m else fname
    canon_key = normalize_canonical_key(run_name, fname)

    # 3. Trích xuất chỉ số UNLABELED TRAIN & DISJOINT TEST
    best_unl_item = None
    best_unl_all = -1.0
    final_unl_item = None

    best_test_item = None
    best_test_all = -1.0
    final_test_item = None

    max_epoch_seen = 0

    # Đọc từ JSON lines
    for line in lines:
        l_str = line.strip()
        if l_str.startswith("{") and l_str.endswith("}"):
            try:
                data = json.loads(l_str)
                ep = data.get("epoch", 0)
                if ep > max_epoch_seen:
                    max_epoch_seen = ep

                # --- UNLABELED TRAIN ---
                if "unlabeled_train" in data and isinstance(data["unlabeled_train"], dict):
                    u = data["unlabeled_train"]
                    all_v = float(u.get("all", u.get("all_acc", 0.0)))
                    u_dict = {
                        "epoch": ep,
                        "all": all_v,
                        "old": float(u.get("old", u.get("known_acc", 0.0))),
                        "new": float(u.get("new", u.get("novel_acc", 0.0))),
                        "h_score": float(u.get("h_score", 0.0)),
                        "k_many": float(u.get("k_many", u.get("known_many_acc", -1.0))),
                        "k_med": float(u.get("k_med", u.get("known_med_acc", -1.0))),
                        "k_few": float(u.get("k_few", u.get("known_few_acc", -1.0))),
                        "u_many": float(u.get("u_many", u.get("novel_many_acc", -1.0))),
                        "u_med": float(u.get("u_med", u.get("novel_med_acc", -1.0))),
                        "u_few": float(u.get("u_few", u.get("novel_few_acc", -1.0))),
                    }
                    if ep == 200:
                        final_unl_item = u_dict
                    if all_v > best_unl_all:
                        best_unl_all = all_v
                        best_unl_item = u_dict

                # --- DISJOINT TEST ---
                if "test" in data and isinstance(data["test"], dict):
                    t = data["test"]
                    t_all_v = float(t.get("all", t.get("all_acc", 0.0)))
                    t_dict = {
                        "epoch": ep,
                        "all": t_all_v,
                        "old": float(t.get("old", t.get("known_acc", 0.0))),
                        "new": float(t.get("new", t.get("novel_acc", 0.0))),
                        "h_score": float(t.get("h_score", 0.0)),
                        "k_many": float(t.get("k_many", t.get("known_many_acc", -1.0))),
                        "k_med": float(t.get("k_med", t.get("known_med_acc", -1.0))),
                        "k_few": float(t.get("k_few", t.get("known_few_acc", -1.0))),
                        "u_many": float(t.get("u_many", t.get("novel_many_acc", -1.0))),
                        "u_med": float(t.get("u_med", t.get("novel_med_acc", -1.0))),
                        "u_few": float(t.get("u_few", t.get("novel_few_acc", -1.0))),
                    }
                    if ep == 200:
                        final_test_item = t_dict
                    if t_all_v > best_test_all:
                        best_test_all = t_all_v
                        best_test_item = t_dict

            except Exception:
                pass

    # Regex fallback cho summary blocks
    # 1. Best Unlabeled Train
    if not best_unl_item:
        best_unl_m = re.search(
            r"\[BEST UNLABELED-TRAIN MODEL\]\s+\(Achieved at Epoch (\d+)\):\s+Unlabeled Train:\s+All:\s+([\d\.]+)%\s+\|\s+Known:\s+([\d\.]+)%\s+\|\s+Novel:\s+([\d\.]+)%\s+\|\s+H-Score:\s+([\d\.]+)%\s+Shot Breakdown:\s+Known \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]\s+Novel \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]",
            content,
        )
        if best_unl_m:
            best_unl_item = {
                "epoch": int(best_unl_m.group(1)),
                "all": float(best_unl_m.group(2)),
                "old": float(best_unl_m.group(3)),
                "new": float(best_unl_m.group(4)),
                "h_score": float(best_unl_m.group(5)),
                "k_many": float(best_unl_m.group(6)),
                "k_med": float(best_unl_m.group(7)),
                "k_few": float(best_unl_m.group(8)),
                "u_many": float(best_unl_m.group(9)),
                "u_med": float(best_unl_m.group(10)),
                "u_few": float(best_unl_m.group(11)),
            }

    # 2. Best Disjoint Test
    if not best_test_item:
        best_test_m = re.search(
            r"\[BEST DISJOINT-TEST MODEL\]\s+\(Achieved at Epoch (\d+)\):\s+Disjoint Test:\s+All:\s+([\d\.]+)%\s+\|\s+Known:\s+([\d\.]+)%\s+\|\s+Novel:\s+([\d\.]+)%\s+\|\s+H-Score:\s+([\d\.]+)%\s+Shot Breakdown:\s+Known \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]\s+Novel \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]",
            content,
        )
        if best_test_m:
            best_test_item = {
                "epoch": int(best_test_m.group(1)),
                "all": float(best_test_m.group(2)),
                "old": float(best_test_m.group(3)),
                "new": float(best_test_m.group(4)),
                "h_score": float(best_test_m.group(5)),
                "k_many": float(best_test_m.group(6)),
                "k_med": float(best_test_m.group(7)),
                "k_few": float(best_test_m.group(8)),
                "u_many": float(best_test_m.group(9)),
                "u_med": float(best_test_m.group(10)),
                "u_few": float(best_test_m.group(11)),
            }

    # 3. Final Model Unlabeled Train & Disjoint Test (Epoch 200)
    if not final_unl_item:
        final_unl_m = re.search(
            r"\[FINAL MODEL - PAPER PROTOCOL\]\s+\(Epoch 200\):.*?Unlabeled Train:\s+All:\s+([\d\.]+)%\s+\|\s+Known:\s+([\d\.]+)%\s+\|\s+Novel:\s+([\d\.]+)%\s+\|\s+H-Score:\s+([\d\.]+)%\s+Shot Breakdown:\s+Known \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]\s+Novel \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]",
            content,
            re.DOTALL,
        )
        if final_unl_m:
            final_unl_item = {
                "epoch": 200,
                "all": float(final_unl_m.group(1)),
                "old": float(final_unl_m.group(2)),
                "new": float(final_unl_m.group(3)),
                "h_score": float(final_unl_m.group(4)),
                "k_many": float(final_unl_m.group(5)),
                "k_med": float(final_unl_m.group(6)),
                "k_few": float(final_unl_m.group(7)),
                "u_many": float(final_unl_m.group(8)),
                "u_med": float(final_unl_m.group(9)),
                "u_few": float(final_unl_m.group(10)),
            }

    if not final_test_item:
        final_test_m = re.search(
            r"\[FINAL MODEL - PAPER PROTOCOL\]\s+\(Epoch 200\):.*?Disjoint Test:\s+All:\s+([\d\.]+)%\s+\|\s+Known:\s+([\d\.]+)%\s+\|\s+Novel:\s+([\d\.]+)%\s+\|\s+H-Score:\s+([\d\.]+)%\s+Shot Breakdown:\s+Known \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]\s+Novel \[Many:\s+([\d\.\-]+),\s+Med:\s+([\d\.\-]+),\s+Few:\s+([\d\.\-]+)\]",
            content,
            re.DOTALL,
        )
        if final_test_m:
            final_test_item = {
                "epoch": 200,
                "all": float(final_test_m.group(1)),
                "old": float(final_test_m.group(2)),
                "new": float(final_test_m.group(3)),
                "h_score": float(final_test_m.group(4)),
                "k_many": float(final_test_m.group(5)),
                "k_med": float(final_test_m.group(6)),
                "k_few": float(final_test_m.group(7)),
                "u_many": float(final_test_m.group(8)),
                "u_med": float(final_test_m.group(9)),
                "u_few": float(final_test_m.group(10)),
            }

    # Chọn metric theo mode
    chosen_unl = best_unl_item if metric_mode == "best" else final_unl_item
    if not chosen_unl and best_unl_item:
        chosen_unl = best_unl_item
    if not chosen_unl and final_unl_item:
        chosen_unl = final_unl_item

    chosen_test = best_test_item if metric_mode == "best" else final_test_item
    if not chosen_test and best_test_item:
        chosen_test = best_test_item
    if not chosen_test and final_test_item:
        chosen_test = final_test_item

    is_done = False
    if "Training finished successfully!" in content or (max_epoch_seen >= 200) or (final_unl_item is not None):
        is_done = True

    status = "Hoàn thành" if is_done else (f"Đang chạy (Ep {max_epoch_seen})" if max_epoch_seen > 0 else "Chưa có kết quả")

    return {
        "run_name": run_name,
        "canon_key": canon_key,
        "file_name": fname,
        "account": account,
        "dt": file_dt,
        "status": status,
        "is_done": is_done,
        # Unlabeled Train
        "unl_epoch": chosen_unl["epoch"] if chosen_unl else None,
        "unl_metrics": chosen_unl,
        "best_unl_item": best_unl_item,
        "final_unl_item": final_unl_item,
        # Disjoint Test
        "test_epoch": chosen_test["epoch"] if chosen_test else None,
        "test_metrics": chosen_test,
        "best_test_item": best_test_item,
        "final_test_item": final_test_item,
    }


def parse_all_logs(log_dir, metric_mode="best"):
    """Quét và trích xuất chỉ số từ toàn bộ log (tự động lấy file mới nhất)."""
    files = []
    for ext in ["*.log", "*.txt"]:
        files.extend(glob.glob(os.path.join(log_dir, ext)))
        files.extend(glob.glob(os.path.join(log_dir, "*", ext)))

    files = [f for f in sorted(list(set(files))) if "old log" not in f.lower()]

    candidates = {}

    for filepath in files:
        item = parse_single_log(filepath, metric_mode=metric_mode)
        if not item or not item["canon_key"]:
            continue

        canon_key = item["canon_key"]
        if canon_key not in candidates:
            candidates[canon_key] = item
        else:
            prev = candidates[canon_key]
            if item["is_done"] and not prev["is_done"]:
                candidates[canon_key] = item
            elif item["is_done"] and prev["is_done"]:
                if item["dt"] > prev["dt"]:
                    candidates[canon_key] = item
            elif not prev["is_done"] and item["dt"] > prev["dt"]:
                candidates[canon_key] = item

    return candidates


def find_log_item(logs_data, model_key, d_key, cfg_key, imb):
    """Tìm log chính xác theo cấu hình chuẩn."""
    expected_run = f"{model_key}_{d_key}_{cfg_key}_{imb}".lower()
    return logs_data.get(expected_run)


def generate_table_rows(logs_data, split_type="unlabeled_train", metric_mode="best"):
    """
    Tạo mảng 2 chiều chứa toàn bộ dữ liệu bảng theo chuẩn Google Sheets.
    split_type: 'unlabeled_train' hoặc 'disjoint_test'
    """
    split_title = "UNLABELED TRAIN" if split_type == "unlabeled_train" else "DISJOINT TEST"
    metrics_field = "unl_metrics" if split_type == "unlabeled_train" else "test_metrics"

    sub_headers = [
        "Ablation Setting",
        "All (%)", "Old (%)", "New (%)", "H-Score",
        "Many (%)", "Med (%)", "Few (%)",
        "Many (%)", "Med (%)", "Few (%)",
        "Ghi chú nơi chạy", "Tên File Log", "Trạng thái"
    ]

    rows = []
    rows.append([
        f"BẢNG TỔNG HỢP KẾT QUẢ {split_title} — MÔ HÌNH BACON (27 CẤU HÌNH x 3 DATASETS) "
        f"Quy ước trực quan ({split_title}): IN ĐẬM (nền xanh lá nhạt) = Tốt nhất (1st) | GẠCH CHÂN (nền xanh dương nhạt) = Tốt nhì (2nd) | Dấu '-' = Chưa chạy / N/A",
        f"CHỈ SỐ CHUNG ({split_title}) All (%)", "Old (%)", "New (%)", "H-Score",
        "KNOWN CLASSES (OLD) Many (%)", "Med (%)", "Few (%)",
        "NOVEL CLASSES (NEW) Many (%)", "Med (%)", "Few (%)",
        "THÔNG TIN THỰC NGHIỆM Ghi chú nơi chạy", "Tên File Log", "Trạng thái"
    ])

    for d_key, d_label, _ in DATASETS:
        rows.append([f"DATASET: {d_label.upper()}"] + [""] * 13)
        for imb in IMB_RATIOS:
            rows.append([f"BẢNG: BaCon — {d_label} (Tỷ lệ mất cân bằng: {imb.upper()})"] + [""] * 13)
            rows.append(["CẤU HÌNH MÔ HÌNH", f"CHỈ SỐ CHUNG ({split_title})", "", "", "", "KNOWN CLASSES (OLD)", "", "", "NOVEL CLASSES (NEW)", "", "", "THÔNG TIN THỰC NGHIỆM", "", ""])
            rows.append(sub_headers)

            for cfg_key, cfg_label, _ in CONFIGS:
                item = find_log_item(logs_data, "bacon", d_key, cfg_key, imb)
                metrics = item[metrics_field] if item else None
                note = item["account"] if (item and item["account"]) else "Chưa chạy"
                logfile = item["file_name"] if item else "-"
                status = item["status"] if item else "Chưa chạy"

                if metrics:
                    def f_val(k):
                        v = metrics.get(k)
                        if v is None or v < 0:
                            return "-"
                        return f"{v:.2f}"

                    row = [
                        cfg_label,
                        f_val("all"), f_val("old"), f_val("new"), f_val("h_score"),
                        f_val("k_many"), f_val("k_med"), f_val("k_few"),
                        f_val("u_many"), f_val("u_med"), f_val("u_few"),
                        note, logfile, status
                    ]
                else:
                    row = [cfg_label] + ["-"] * 10 + [note, logfile, status]

                rows.append(row)

    return rows


def export_to_google_sheet_csv_and_tsv(logs_data, csv_path, tsv_path, split_type="unlabeled_train", metric_mode="best"):
    """Xuất file CSV và TSV chuẩn hóa cho Google Sheets."""
    rows = generate_table_rows(logs_data, split_type=split_type, metric_mode=metric_mode)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[+] Đã xuất file CSV ({split_type}): {csv_path}")

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    print(f"[+] Đã xuất file TSV ({split_type} - Paste-ready): {tsv_path}")


def build_excel_report(logs_data, output_path, metric_mode="best"):
    """Tạo workbook Excel hoàn chỉnh với các sheet chuyên biệt: Dashboard, Unlabeled Train, Disjoint Test."""
    wb = Workbook()
    default_sheet = wb.active

    EMERALD_HEADER = "047857"  # BaCon
    BLUE_HEADER = "2563EB"     # SimGCD
    BLUE_LIGHT = "EFF6FF"
    AMBER_LIGHT = "FFFBEB"
    PURPLE_LIGHT = "FAF5FF"
    GRAY_HEADER = "475569"
    GRAY_LIGHT = "F8FAFC"
    BORDER_COLOR = "CBD5E1"
    
    FILL_MAX = "DCFCE7"        # Max: In đậm + xanh bạc hà nhạt
    FILL_2ND = "E0F2FE"        # 2nd: Gạch chân + xanh da trời nhạt
    FILL_ZEBRA = "F8FAFC"

    font_family = "Segoe UI"
    
    thin_border = Border(
        left=Side(style="thin", color=BORDER_COLOR),
        right=Side(style="thin", color=BORDER_COLOR),
        top=Side(style="thin", color=BORDER_COLOR),
        bottom=Side(style="thin", color=BORDER_COLOR),
    )
    
    thick_bottom_border = Border(
        left=Side(style="thin", color=BORDER_COLOR),
        right=Side(style="thin", color=BORDER_COLOR),
        top=Side(style="thin", color=BORDER_COLOR),
        bottom=Side(style="medium", color="64748B"),
    )

    def render_mini_table(ws, start_row, model_key, model_label, d_key, d_label, imb, split_type="unlabeled_train"):
        split_title = "UNLABELED TRAIN" if split_type == "unlabeled_train" else "DISJOINT TEST"
        metrics_field = "unl_metrics" if split_type == "unlabeled_train" else "test_metrics"
        epoch_field = "unl_epoch" if split_type == "unlabeled_train" else "test_epoch"

        title = f"BẢNG: {model_label} — {d_label} (Tỷ lệ mất cân bằng: {imb.upper()}) [{split_title}]"
        
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=15)
        title_cell = ws.cell(row=start_row, column=1, value=title)
        title_cell.font = Font(name=font_family, size=11, bold=True, color="FFFFFF")
        title_cell.fill = PatternFill(start_color=EMERALD_HEADER if model_key == "bacon" else BLUE_HEADER, fill_type="solid")
        title_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[start_row].height = 26

        sr = start_row + 1
        ws.row_dimensions[sr].height = 20

        c_cfg = ws.cell(row=sr, column=1, value="CẤU HÌNH MÔ HÌNH")
        c_cfg.font = Font(name=font_family, size=9, bold=True, color="FFFFFF")
        c_cfg.fill = PatternFill(start_color=GRAY_HEADER, fill_type="solid")
        c_cfg.alignment = Alignment(horizontal="center", vertical="center")
        c_cfg.border = thin_border

        ws.merge_cells(start_row=sr, start_column=2, end_row=sr, end_column=6)
        c_main = ws.cell(row=sr, column=2, value=f"CHỈ SỐ CHUNG ({metric_mode.upper()} {split_title})")
        c_main.font = Font(name=font_family, size=9, bold=True, color="1E3A8A")
        c_main.fill = PatternFill(start_color=BLUE_LIGHT, fill_type="solid")
        c_main.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(2, 7):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=BLUE_LIGHT, fill_type="solid")

        ws.merge_cells(start_row=sr, start_column=7, end_row=sr, end_column=9)
        c_old = ws.cell(row=sr, column=7, value="KNOWN CLASSES (OLD)")
        c_old.font = Font(name=font_family, size=9, bold=True, color="92400E")
        c_old.fill = PatternFill(start_color=AMBER_LIGHT, fill_type="solid")
        c_old.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(7, 10):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=AMBER_LIGHT, fill_type="solid")

        ws.merge_cells(start_row=sr, start_column=10, end_row=sr, end_column=12)
        c_new = ws.cell(row=sr, column=10, value="NOVEL CLASSES (NEW)")
        c_new.font = Font(name=font_family, size=9, bold=True, color="6B21A8")
        c_new.fill = PatternFill(start_color=PURPLE_LIGHT, fill_type="solid")
        c_new.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(10, 13):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=PURPLE_LIGHT, fill_type="solid")

        ws.merge_cells(start_row=sr, start_column=13, end_row=sr, end_column=15)
        c_info = ws.cell(row=sr, column=13, value="THÔNG TIN THỰC NGHIỆM")
        c_info.font = Font(name=font_family, size=9, bold=True, color="334155")
        c_info.fill = PatternFill(start_color=GRAY_LIGHT, fill_type="solid")
        c_info.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(13, 16):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=GRAY_LIGHT, fill_type="solid")

        sub_headers = [
            ("Ablation Setting", 1),
            ("Epoch", 2),
            ("All (%)", 3),
            ("Old (%)", 4),
            ("New (%)", 5),
            ("H-Score", 6),
            ("Many (%)", 7),
            ("Med (%)", 8),
            ("Few (%)", 9),
            ("Many (%)", 10),
            ("Med (%)", 11),
            ("Few (%)", 12),
            ("Ghi chú nơi chạy", 13),
            ("Tên File Log", 14),
            ("Trạng thái", 15),
        ]
        shr = start_row + 2
        ws.row_dimensions[shr].height = 22
        for text, col_idx in sub_headers:
            cell = ws.cell(row=shr, column=col_idx, value=text)
            cell.font = Font(name=font_family, size=9, bold=True, color="334155")
            cell.fill = PatternFill(start_color="F1F5F9", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thick_bottom_border

        rows_data = []
        column_values = {k: [] for k in METRIC_KEYS}

        for cfg_key, cfg_label, cfg_desc in CONFIGS:
            log_item = find_log_item(logs_data, model_key, d_key, cfg_key, imb)
            metrics = log_item[metrics_field] if log_item else None
            note = log_item["account"] if (log_item and log_item["account"]) else "Chưa chạy"
            logfile = log_item["file_name"] if log_item else "-"
            status = log_item["status"] if log_item else "Chưa chạy"
            ep_val = log_item[epoch_field] if log_item and log_item[epoch_field] is not None else "-"

            rows_data.append({
                "config_label": cfg_label,
                "ep_val": ep_val,
                "metrics": metrics,
                "note": note,
                "logfile": logfile,
                "status": status,
            })

            if metrics:
                for k in METRIC_KEYS:
                    val = metrics.get(k)
                    if val is not None and val >= 0:
                        column_values[k].append(val)

        ranks = {}
        for k in METRIC_KEYS:
            vals = sorted(list(set(column_values[k])), reverse=True)
            first = vals[0] if len(vals) > 0 else None
            second = vals[1] if len(vals) > 1 else None
            ranks[k] = (first, second)

        for i, rdata in enumerate(rows_data):
            curr_row = start_row + 3 + i
            ws.row_dimensions[curr_row].height = 20
            is_zebra = (i % 2 == 1)
            row_fill = PatternFill(start_color=FILL_ZEBRA, fill_type="solid") if is_zebra else PatternFill(fill_type=None)

            c_cfg = ws.cell(row=curr_row, column=1, value=rdata["config_label"])
            c_cfg.font = Font(name=font_family, size=9.5, bold=True if "Full" in rdata["config_label"] else False, color="0F172A")
            c_cfg.alignment = Alignment(horizontal="left", vertical="center", indent=1)
            c_cfg.border = thin_border
            if is_zebra:
                c_cfg.fill = row_fill

            c_ep = ws.cell(row=curr_row, column=2, value=rdata["ep_val"])
            c_ep.font = Font(name=font_family, size=9, color="475569")
            c_ep.alignment = Alignment(horizontal="center", vertical="center")
            c_ep.border = thin_border
            if is_zebra:
                c_ep.fill = row_fill

            metrics = rdata["metrics"]
            for col_offset, mkey in enumerate(METRIC_KEYS):
                col_idx = 3 + col_offset
                cell = ws.cell(row=curr_row, column=col_idx)
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="right", vertical="center")

                if metrics and mkey in metrics and metrics[mkey] is not None and metrics[mkey] >= 0:
                    val = metrics[mkey]
                    cell.value = val
                    cell.number_format = "0.00"

                    first_val, second_val = ranks[mkey]
                    if first_val is not None and abs(val - first_val) < 1e-5:
                        cell.font = Font(name=font_family, size=9.5, bold=True, color="14532D")
                        cell.fill = PatternFill(start_color=FILL_MAX, fill_type="solid")
                    elif second_val is not None and abs(val - second_val) < 1e-5:
                        cell.font = Font(name=font_family, size=9.5, bold=False, underline="single", color="1E3A8A")
                        cell.fill = PatternFill(start_color=FILL_2ND, fill_type="solid")
                    else:
                        cell.font = Font(name=font_family, size=9.5, color="334155")
                        if is_zebra:
                            cell.fill = row_fill
                else:
                    cell.value = "-"
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.font = Font(name=font_family, size=9.5, italic=True, color="94A3B8")
                    if is_zebra:
                        cell.fill = row_fill

            c_note = ws.cell(row=curr_row, column=13, value=rdata["note"])
            c_note.font = Font(name=font_family, size=9, color="1E293B" if rdata["note"] != "Chưa chạy" else "94A3B8")
            c_note.alignment = Alignment(horizontal="left", vertical="center", indent=1)
            c_note.border = thin_border
            if is_zebra:
                c_note.fill = row_fill

            c_log = ws.cell(row=curr_row, column=14, value=rdata["logfile"])
            c_log.font = Font(name="Consolas", size=8.5, color="475569" if rdata["logfile"] != "-" else "94A3B8")
            c_log.alignment = Alignment(horizontal="left", vertical="center")
            c_log.border = thin_border
            if is_zebra:
                c_log.fill = row_fill

            c_status = ws.cell(row=curr_row, column=15, value=rdata["status"])
            if "Hoàn thành" in rdata["status"]:
                c_status.font = Font(name=font_family, size=9, bold=True, color="15803D")
                c_status.fill = PatternFill(start_color="DCFCE7", fill_type="solid")
            elif "Đang chạy" in rdata["status"]:
                c_status.font = Font(name=font_family, size=9, bold=True, color="B45309")
                c_status.fill = PatternFill(start_color="FEF3C7", fill_type="solid")
            else:
                c_status.font = Font(name=font_family, size=9, color="94A3B8", italic=True)
                if is_zebra:
                    c_status.fill = row_fill
            c_status.alignment = Alignment(horizontal="center", vertical="center")
            c_status.border = thin_border

        return start_row + 3 + len(CONFIGS) + 2

    # -----------------------------------------------------------------
    # Sheet 1: BaCon - Unlabeled Train
    # -----------------------------------------------------------------
    ws_unl = wb.create_sheet(title="BaCon (Unlabeled Train)")
    ws_unl.views.sheetView[0].showGridLines = True

    ws_unl.merge_cells("A1:O1")
    b_banner = ws_unl.cell(row=1, column=1, value=f"BẢNG TỔNG HỢP KẾT QUẢ UNLABELED TRAIN ({metric_mode.upper()}) — MÔ HÌNH BACON")
    b_banner.font = Font(name=font_family, size=13, bold=True, color="FFFFFF")
    b_banner.fill = PatternFill(start_color="065F46", fill_type="solid")
    b_banner.alignment = Alignment(horizontal="center", vertical="center")
    ws_unl.row_dimensions[1].height = 36

    ws_unl.merge_cells("A2:O2")
    b_sub = ws_unl.cell(row=2, column=1, value="Quy ước trực quan: IN ĐẬM (nền xanh lá nhạt) = Tốt nhất (1st) | GẠCH CHÂN (nền xanh dương nhạt) = Tốt nhì (2nd) | Dấu '-' = Chưa chạy / N/A")
    b_sub.font = Font(name=font_family, size=9.5, italic=True, color="475569")
    b_sub.alignment = Alignment(horizontal="center", vertical="center")
    ws_unl.row_dimensions[2].height = 20

    current_row = 4
    for d_key, d_label, _ in DATASETS:
        ws_unl.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=15)
        d_cell = ws_unl.cell(row=current_row, column=1, value=f"DATASET: {d_label.upper()}")
        d_cell.font = Font(name=font_family, size=11, bold=True, color="065F46")
        d_cell.fill = PatternFill(start_color="D1FAE5", fill_type="solid")
        d_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws_unl.row_dimensions[current_row].height = 26
        current_row += 1

        for imb in IMB_RATIOS:
            current_row = render_mini_table(ws_unl, current_row, "bacon", "BaCon", d_key, d_label, imb, split_type="unlabeled_train")

    # -----------------------------------------------------------------
    # Sheet 2: BaCon - Disjoint Test
    # -----------------------------------------------------------------
    ws_test = wb.create_sheet(title="BaCon (Disjoint Test)")
    ws_test.views.sheetView[0].showGridLines = True

    ws_test.merge_cells("A1:O1")
    t_banner = ws_test.cell(row=1, column=1, value=f"BẢNG TỔNG HỢP KẾT QUẢ DISJOINT TEST ({metric_mode.upper()}) — MÔ HÌNH BACON")
    t_banner.font = Font(name=font_family, size=13, bold=True, color="FFFFFF")
    t_banner.fill = PatternFill(start_color="1E3A8A", fill_type="solid")
    t_banner.alignment = Alignment(horizontal="center", vertical="center")
    ws_test.row_dimensions[1].height = 36

    ws_test.merge_cells("A2:O2")
    t_sub = ws_test.cell(row=2, column=1, value="Quy ước trực quan: IN ĐẬM (nền xanh lá nhạt) = Tốt nhất (1st) | GẠCH CHÂN (nền xanh dương nhạt) = Tốt nhì (2nd) | Dấu '-' = Chưa chạy / N/A")
    t_sub.font = Font(name=font_family, size=9.5, italic=True, color="475569")
    t_sub.alignment = Alignment(horizontal="center", vertical="center")
    ws_test.row_dimensions[2].height = 20

    current_row = 4
    for d_key, d_label, _ in DATASETS:
        ws_test.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=15)
        d_cell = ws_test.cell(row=current_row, column=1, value=f"DATASET: {d_label.upper()}")
        d_cell.font = Font(name=font_family, size=11, bold=True, color="1E3A8A")
        d_cell.fill = PatternFill(start_color="DBEAFE", fill_type="solid")
        d_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws_test.row_dimensions[current_row].height = 26
        current_row += 1

        for imb in IMB_RATIOS:
            current_row = render_mini_table(ws_test, current_row, "bacon", "BaCon", d_key, d_label, imb, split_type="disjoint_test")

    # -----------------------------------------------------------------
    # Sheet 0: Tổng Quan & Tiến Độ
    # -----------------------------------------------------------------
    ws_dash = wb.create_sheet(title="Tổng Quan & Tiến Độ", index=0)
    ws_dash.views.sheetView[0].showGridLines = True
    
    ws_dash.merge_cells("A1:M1")
    dash_title = ws_dash.cell(row=1, column=1, value="BẢNG ĐIỀU KHIỂN TIẾN ĐỘ THỰC NGHIỆM BACON (TRAIN & TEST)")
    dash_title.font = Font(name=font_family, size=14, bold=True, color="FFFFFF")
    dash_title.fill = PatternFill(start_color="0F172A", fill_type="solid")
    dash_title.alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[1].height = 40

    ws_dash.merge_cells("A2:M2")
    dash_time = ws_dash.cell(row=2, column=1, value=f"Dữ liệu: {metric_mode.upper()} Unlabeled Train & Disjoint Test | Cập nhật lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    dash_time.font = Font(name=font_family, size=9.5, italic=True, color="64748B")
    dash_time.alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[2].height = 22

    total_bacon_configs = len(DATASETS) * len(IMB_RATIOS) * len(CONFIGS)
    completed_runs = sum(1 for k, v in logs_data.items() if k.startswith("bacon") and v.get("is_done"))
    running_runs = sum(1 for k, v in logs_data.items() if k.startswith("bacon") and not v.get("is_done") and (v.get("unl_metrics") or v.get("test_metrics")))
    pending_runs = total_bacon_configs - completed_runs - running_runs

    kpi_cards = [
        ("TỔNG CẤU HÌNH BACON", total_bacon_configs, "065F46", "D1FAE5", 2),
        ("ĐÃ HOÀN THÀNH", completed_runs, "14532D", "DCFCE7", 5),
        ("ĐANG CHẠY", running_runs, "78350F", "FEF3C7", 8),
        ("CHƯA CHẠY", pending_runs, "334155", "F1F5F9", 11),
    ]

    for title, val, text_c, bg_c, col in kpi_cards:
        ws_dash.merge_cells(start_row=4, start_column=col, end_row=4, end_column=col+1)
        ws_dash.merge_cells(start_row=5, start_column=col, end_row=5, end_column=col+1)
        
        c1 = ws_dash.cell(row=4, column=col, value=title)
        c1.font = Font(name=font_family, size=9, bold=True, color=text_c)
        c1.fill = PatternFill(start_color=bg_c, fill_type="solid")
        c1.alignment = Alignment(horizontal="center", vertical="center")
        
        c2 = ws_dash.cell(row=5, column=col, value=val)
        c2.font = Font(name=font_family, size=18, bold=True, color=text_c)
        c2.fill = PatternFill(start_color=bg_c, fill_type="solid")
        c2.alignment = Alignment(horizontal="center", vertical="center")
        
        for r_idx in [4, 5]:
            ws_dash.cell(row=r_idx, column=col).border = thin_border
            ws_dash.cell(row=r_idx, column=col+1).border = thin_border

    ws_dash.row_dimensions[4].height = 20
    ws_dash.row_dimensions[5].height = 32

    ws_dash.cell(row=7, column=2, value="MA TRẬN TIẾN ĐỘ THỰC NGHIỆM BACON (SO SÁNH TRAIN & TEST)").font = Font(name=font_family, size=11, bold=True, color="0F172A")
    matrix_headers = [
        "Mô hình", "Dataset", "Tỷ lệ Imb", "Số run xong", "Tiến độ",
        "Train All (%)", "Train H-Score", 
        "Test All (%)", "Test H-Score", 
        "Nơi chạy Full", "Trạng thái bảng"
    ]
    for c_i, h in enumerate(matrix_headers, start=2):
        cell = ws_dash.cell(row=8, column=c_i, value=h)
        cell.font = Font(name=font_family, size=9.5, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="334155", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws_dash.row_dimensions[8].height = 24

    m_row = 9
    for d_key, d_label, _ in DATASETS:
        for imb in IMB_RATIOS:
            done_count = 0
            full_unl = None
            full_test = None
            full_account = "-"
            for cfg_key, _, _ in CONFIGS:
                item = find_log_item(logs_data, "bacon", d_key, cfg_key, imb)
                if item and item.get("is_done"):
                    done_count += 1
                if cfg_key == "Full" and item:
                    full_unl = item.get("unl_metrics")
                    full_test = item.get("test_metrics")
                    full_account = item["account"]

            pct_done = done_count / 6.0
            ws_dash.cell(row=m_row, column=2, value="BaCon").alignment = Alignment(horizontal="center", vertical="center")
            ws_dash.cell(row=m_row, column=3, value=d_label).alignment = Alignment(horizontal="left", vertical="center", indent=1)
            ws_dash.cell(row=m_row, column=4, value=imb.upper()).alignment = Alignment(horizontal="center", vertical="center")
            ws_dash.cell(row=m_row, column=5, value=f"{done_count} / 6").alignment = Alignment(horizontal="center", vertical="center")
            
            c_pct = ws_dash.cell(row=m_row, column=6, value=pct_done)
            c_pct.number_format = "0.0%"
            c_pct.alignment = Alignment(horizontal="center", vertical="center")
            
            # Train All & H-Score
            if full_unl:
                c_u_a = ws_dash.cell(row=m_row, column=7, value=full_unl["all"])
                c_u_a.number_format = "0.00"
                c_u_a.font = Font(name=font_family, size=9.5, bold=True, color="14532D")
                c_u_h = ws_dash.cell(row=m_row, column=8, value=full_unl["h_score"])
                c_u_h.number_format = "0.00"
            else:
                ws_dash.cell(row=m_row, column=7, value="-").alignment = Alignment(horizontal="center", vertical="center")
                ws_dash.cell(row=m_row, column=8, value="-").alignment = Alignment(horizontal="center", vertical="center")

            # Test All & H-Score
            if full_test:
                c_t_a = ws_dash.cell(row=m_row, column=9, value=full_test["all"])
                c_t_a.number_format = "0.00"
                c_t_a.font = Font(name=font_family, size=9.5, bold=True, color="1E3A8A")
                c_t_h = ws_dash.cell(row=m_row, column=10, value=full_test["h_score"])
                c_t_h.number_format = "0.00"
            else:
                ws_dash.cell(row=m_row, column=9, value="-").alignment = Alignment(horizontal="center", vertical="center")
                ws_dash.cell(row=m_row, column=10, value="-").alignment = Alignment(horizontal="center", vertical="center")
            
            ws_dash.cell(row=m_row, column=11, value=full_account).alignment = Alignment(horizontal="center", vertical="center")
            
            c_st = ws_dash.cell(row=m_row, column=12, value="Hoàn thành" if done_count == 6 else ("Đang có kết quả" if done_count > 0 else "Chưa có log"))
            c_st.alignment = Alignment(horizontal="center", vertical="center")
            if done_count == 6:
                c_st.fill = PatternFill(start_color="DCFCE7", fill_type="solid")
                c_st.font = Font(name=font_family, size=9, bold=True, color="15803D")
            elif done_count > 0:
                c_st.fill = PatternFill(start_color="FEF3C7", fill_type="solid")
                c_st.font = Font(name=font_family, size=9, bold=True, color="B45309")
            else:
                c_st.font = Font(name=font_family, size=9, color="94A3B8")

            for ci in range(2, 13):
                ws_dash.cell(row=m_row, column=ci).border = thin_border
            ws_dash.row_dimensions[m_row].height = 20
            m_row += 1

    # Căn chỉnh độ rộng cột
    for ws in [ws_dash, ws_unl, ws_test]:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.row in [1, 2, 4] and ws in [ws_dash, ws_unl, ws_test]:
                    continue
                if cell.value:
                    val_str = str(cell.value)
                    if len(val_str) > max_len:
                        max_len = len(val_str)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 11)

    for ws in [ws_unl, ws_test]:
        ws.column_dimensions["A"].width = 22
        for cl in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
            ws.column_dimensions[cl].width = 11
        ws.column_dimensions["M"].width = 22
        ws.column_dimensions["N"].width = 46
        ws.column_dimensions["O"].width = 14

    if default_sheet.title in wb.sheetnames:
        wb.remove(default_sheet)

    try:
        wb.save(output_path)
        print(f"[+] Báo cáo Excel đã được lưu thành công tại: {output_path}")
    except PermissionError:
        alt_path = output_path.replace(".xlsx", "_MoiNhat.xlsx")
        wb.save(alt_path)
        print(f"[!] CẢNH BÁO: File '{os.path.basename(output_path)}' đang được mở!")
        print(f"    -> Đã tự động lưu ra file dự phòng: {alt_path}")


if __name__ == "__main__":
    import sys
    import argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="BaCon GCD Benchmark Log Parser & Exporter (Train & Test)")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR, help="Thư mục chứa log BaCon")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_EXCEL, help="Đường dẫn file Excel xuất ra")
    parser.add_argument("--csv-unl", type=str, default=DEFAULT_CSV_UNL, help="Đường dẫn file CSV xuất Unlabeled Train")
    parser.add_argument("--tsv-unl", type=str, default=DEFAULT_TSV_UNL, help="Đường dẫn file TSV xuất Unlabeled Train (Paste-ready)")
    parser.add_argument("--csv-test", type=str, default=DEFAULT_CSV_TEST, help="Đường dẫn file CSV xuất Disjoint Test")
    parser.add_argument("--tsv-test", type=str, default=DEFAULT_TSV_TEST, help="Đường dẫn file TSV xuất Disjoint Test (Paste-ready)")
    parser.add_argument("--metric-mode", type=str, choices=["best", "final"], default="best", help="Chế độ lấy kết quả (best: Best Epoch, final: Epoch 200)")
    args = parser.parse_args()

    print(f"[*] Quét thư mục log: {args.log_dir} (Chế độ: {args.metric_mode})")
    logs = parse_all_logs(args.log_dir, metric_mode=args.metric_mode)
    print(f"[*] Đã nhận diện thành công: {len(logs)} cấu hình thực nghiệm.")
    
    # 1. Xuất file Excel (gồm 3 sheet: Tổng quan, Unlabeled Train, Disjoint Test)
    build_excel_report(logs, args.output, metric_mode=args.metric_mode)
    
    # 2. Xuất CSV & TSV Unlabeled Train
    export_to_google_sheet_csv_and_tsv(logs, args.csv_unl, args.tsv_unl, split_type="unlabeled_train", metric_mode=args.metric_mode)

    # 3. Xuất CSV & TSV Disjoint Test
    export_to_google_sheet_csv_and_tsv(logs, args.csv_test, args.tsv_test, split_type="disjoint_test", metric_mode=args.metric_mode)

    print(f"[+] Hoàn tất thành công toàn bộ bảng Unlabeled Train & Disjoint Test!")
