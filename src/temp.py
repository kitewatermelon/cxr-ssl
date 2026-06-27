import numpy as np
import torch


def save_representations(all_h, all_hp, dataset, save_dir="/mnt/nvme1/mimic-cxr-jpg"):
    """
    all_h, all_hp : torch.Tensor (N, D) — encoder representations
    dataset       : FromTorchDataset wrapping MIMICCXRDataset
                    (dataset.dataset.df 에 subject_id, study_id, dicom_id 있어야 함)
    save_dir      : 저장 경로

    저장 파일:
        vit_s_16_h.npz  — {'h': (N, D), 'paths': (N,)}
        vit_s_16_hp.npz — {'hp': (N, D), 'paths': (N,)}
    """
    import os

    df = dataset.dataset.df
    paths = np.array([dataset.dataset._get_image_path(df.iloc[i]) for i in range(len(df))])

    h_np  = all_h .cpu().numpy() if isinstance(all_h,  torch.Tensor) else all_h
    hp_np = all_hp.cpu().numpy() if isinstance(all_hp, torch.Tensor) else all_hp

    assert len(paths) == len(h_np) == len(hp_np), \
        f"길이 불일치: paths={len(paths)}, h={len(h_np)}, hp={len(hp_np)}"

    out_h  = os.path.join(save_dir, "vit_s_16_h.npz")
    out_hp = os.path.join(save_dir, "vit_s_16_hp.npz")

    np.savez(out_h,  h=h_np,   paths=paths)
    np.savez(out_hp, hp=hp_np, paths=paths)

    print(f"Saved {len(paths)} samples")
    print(f"  {out_h}   shape={h_np.shape}")
    print(f"  {out_hp}  shape={hp_np.shape}")
    print(f"  예시 path: {paths[0]}")
