import torch
import os

def inspect_pt_file(file_path):
    if not os.path.exists(file_path):
        print(f'[-] Không tìm thấy file: {file_path}')
        return
        
    data = torch.load(file_path, weights_only=False)
    
    print(f'=== File: {file_path} ===')
    print(f'- Kiểu dữ liệu (Type): {type(data)}')
    print(f'- Tổng số lượng ảnh (Length): {len(data)}')
    
    try:
        print(f'- 15 vị trí đầu tiên: {data[:15].tolist()}')
        print(f'- Max index: {data.max().item()}')
        print(f'- Min index: {data.min().item()}')
    except Exception:
        print(f'- 15 vị trí đầu tiên: {data[:15]}')
        print(f'- Max index: {max(data)}')
        print(f'- Min index: {min(data)}')
    print('')

if __name__ == '__main__':
    # ===== BẠN CHỈ CẦN SỬA TÊN THƯ MỤC Ở ĐÂY =====
    split_dir = 'data_uq_idxs/cub200_k100_imb10'
    
    files_to_check = ['l_k_uq_idxs.pt', 'unl_k_uq_idxs.pt', 'unl_unk_uq_idxs.pt']
    
    for f in files_to_check:
        full_path = os.path.join(split_dir, f)
        inspect_pt_file(full_path)
