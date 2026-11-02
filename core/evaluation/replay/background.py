import argparse
from pathlib import Path

import cv2
import numpy as np


def _default_output_path(path: Path) -> Path:
    if path.is_file():
        return path.with_name(f"{path.stem}_transparent{path.suffix}")
    return path.with_name(f"{path.name}_transparent")


def remove_border_black(image: np.ndarray, threshold: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA image, got shape {image.shape}")

    if image.shape[2] == 3:
        bgr = image
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    else:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3].copy()

    near_black = np.all(bgr <= threshold, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(near_black, connectivity=4)
    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size:
        background = np.isin(labels, border_labels)
        alpha[background] = 0

    return np.dstack([bgr, alpha])


def process_path(input_path: Path, output_path: Path, threshold: int) -> int:
    if input_path.is_file():
        files = [input_path]
        output_files = [output_path]
    else:
        files = sorted(input_path.rglob("frame_*.png"))
        output_files = [output_path / file.relative_to(input_path) for file in files]

    for src, dst in zip(files, output_files):
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {src}")
        processed = remove_border_black(image, threshold)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), processed):
            raise ValueError(f"Could not write image: {dst}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove near-black border-connected background from trajectory PNG frames."
    )
    parser.add_argument("input", type=Path, help="A frame PNG or a directory containing frame_*.png files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file/directory. Defaults to a *_transparent sibling.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Maximum channel value considered background black.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or _default_output_path(input_path)
    count = process_path(input_path, output_path, args.threshold)
    print(f"Wrote {count} transparent frame(s) to {output_path}")


if __name__ == "__main__":
    main()
