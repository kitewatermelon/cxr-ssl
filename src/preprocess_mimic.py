"""
MIMIC-CXR JPG → 256×256 grayscale npy 변환
.jpg 파일과 같은 위치에 .npy로 저장
"""

import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


ROOT = "/mnt/nvme1/mimic-cxr-jpg/files"
NUM_WORKERS = 22
SIZE = 256


def convert_one(jpg_path: str) -> str | None:
    npy_path = jpg_path[:-4] + ".npy"
    if os.path.exists(npy_path):
        return None  # 이미 변환됨

    img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"FAIL: {jpg_path}"

    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    np.save(npy_path, img)
    return None


def main():
    jpg_files = list(Path(ROOT).rglob("*.jpg"))
    print(f"총 {len(jpg_files):,}개 JPG 발견")

    jpg_paths = [str(p) for p in jpg_files]

    errors = []
    with Pool(NUM_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(convert_one, jpg_paths, chunksize=64),
            total=len(jpg_paths),
            desc="변환 중",
        ):
            if result is not None:
                errors.append(result)

    print(f"\n완료: {len(jpg_paths) - len(errors):,}개 성공, {len(errors)}개 실패")
    if errors:
        for e in errors[:10]:
            print(" ", e)


if __name__ == "__main__":
    main()
