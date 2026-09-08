import sys
import subprocess

def run_modal(experiment_name, **kwargs):
    # Dùng cờ -d (detach) và trỏ thẳng vào ::launch để submit background
    cmd = ["modal", "run", "-d", "modal_train.py::launch", "--exp-name-suffix", experiment_name]
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k.replace('_', '-')}")
        else:
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    
    print("Executing:", " ".join(cmd))
    
    # Thiết lập biến môi trường để Windows PowerShell không bị crash vì ký tự ✓ (checkmark) của Modal
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # Đã BỎ COMMENT dòng này để chạy luôn 
    subprocess.run(cmd, env=env, check=True) 

def main():
    print("--- Chạy Thực Nghiệm ---")
    
    # HƯỚNG DẪN COMMENT:
    # - Nếu bạn KHÔNG muốn chạy Row nào, hãy thêm dấu # vào đầu dòng chữ `run_modal` của Row đó.
    # - Nếu bạn MUỐN chạy Row nào, hãy xóa dấu # ở đầu dòng chữ `run_modal` của Row đó đi.

    # Row 1: BaCon gốc (Không dùng AdaPart, Không dùng Pseudo-labeling)
    # run_modal("Ablation_Row1_Baseline", dataset_name="cub200", imb_ratio=10)

    # Row 2: BaCon gốc + Pseudo-labeling Mode 1 (Không dùng AdaPart)
    # run_modal("Ablation_Row2_PseudoOnly", dataset_name="cub200", imb_ratio=10, enable_pseudo_labeling=True)

    # -------------------------------------------------------------
    # CẤU HÌNH GỐC CHO CÁC ROW ADAPART (Bắt buộc phải giữ nguyên khối này)
    base_kwargs = {
        "dataset_name": "cub200",
        "imb_ratio": 10,
        "enable_pseudo_labeling": True,
        "use_parts": True,
        "num_slots": 3
    }
    # -------------------------------------------------------------

    # Row 4: Tắt Spatial Diversity Loss
    # run_modal("Ablation_Row4_NoSpatial", ablate_spatial_loss=True, **base_kwargs)
    
    # Row 5: Tắt Fused CE Loss
    # run_modal("Ablation_Row5_NoFusedCE", ablate_fused_ce=True, **base_kwargs)

    # Row 6: Tắt Confidence Filtering
    # run_modal("Ablation_Row6_NoConf", ablate_confidence=True, **base_kwargs)

    # Row 7: Tắt Adaptive Capacity
    # run_modal("Ablation_Row7_NoAdaptiveCap", ablate_adaptive_capacity=True, **base_kwargs)

    # Row 8: Tắt Concat Eval
    # run_modal("Ablation_Row8_NoConcatEval", ablate_concat_eval=True, **base_kwargs)

    # Row 9: Đổi số lượng Part (ví dụ M=2)
    # run_modal("Ablation_Row9_M2", num_slots=2, **{k:v for k,v in base_kwargs.items() if k != 'num_slots'})

    # Row 10: Đầy đủ tính năng AdaPart (Full)
    # run_modal("Ablation_Row10_Full_AdaPart", **base_kwargs)

    # -------------------------------------------------------------
    # NEW HYBRID ROWS - CHỨNG MINH THUYẾT SPATIAL LOSS
    # -------------------------------------------------------------
    
    # Row 11: NoSpatial + NoConf (Giữ nguyên NoSpatial nhưng tắt thêm Confidence Filtering)
    # run_modal("Ablation_Row11_NoSpatial_NoConf", ablate_spatial_loss=True, ablate_confidence=True, **base_kwargs)

    # Row 12: NoSpatial + NoAdaptiveCap (Giữ nguyên NoSpatial nhưng tắt thêm Adaptive Capacity)
    # run_modal("Ablation_Row12_NoSpatial_NoAdaptiveCap", ablate_spatial_loss=True, ablate_adaptive_capacity=True, **base_kwargs)
    
    # -------------------------------------------------------------
    # ROW 13 - GOLDEN CONFIG (ĐÃ PATCH CODE TRONG BACON.PY VÀ MODAL_TRAIN.PY)
    # - M=2 Slots
    # - Spatial Loss Delay (chỉ apply sau epoch 50 với weight 0.05)
    # - Windowed Pseudo-Labeling (chỉ update EMA novel từ epoch 30 đến 60)
    # - Warm-up teacher temp kéo dài lên 50 epoch
    # -------------------------------------------------------------
    # run_modal("Ablation_Row13_GoldenConfig", num_slots=2, **{k:v for k,v in base_kwargs.items() if k != 'num_slots'})

    # -------------------------------------------------------------
    # ROW 14 - THEO YÊU CẦU: NoSpatial + Pseudo-labeling (Với Code đã Patch)
    # -------------------------------------------------------------
    run_modal("Ablation_Row14_NoSpatial_PseudoOnly", 
              ablate_spatial_loss=True, 
              ablate_fused_ce=False, 
              ablate_confidence=False, 
              ablate_adaptive_capacity=False, 
              ablate_concat_eval=False, 
              **base_kwargs)

    print("\nĐã submit job Golden Config và Row 14 lên Modal thành công! Hãy chờ xem điều kỳ diệu nhé.")

if __name__ == "__main__":
    main()
