from __future__ import annotations
import argparse
from pathlib import Path

from ift import IFT
from mst import MST


def build_output_path(method: str, image_path: str, value: str) -> str:
    p = Path(image_path)
    stem = p.stem
    return str(Path("results") / method / f"{stem}_{value}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Image segmentation: IFT and MST.")
    parser.add_argument("method", choices=["ift", "mst"], help="Method: 'ift' or 'mst'")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("value", help="For IFT: number of seeds (int). For MST: k value (float).")

    args = parser.parse_args()

    method = args.method.lower()
    image_path = args.image
    value = args.value

    output_path = build_output_path(method, image_path, value)

    try:
        if method == "ift":
            num_seeds = int(value)
            algo = IFT(image_path, num_seeds=num_seeds)
            algo.run()
            algo.save_result(output_path)
        elif method == "mst":
            k = float(value)
            algo = MST(image_path, k=k)
            algo.segmentWithRealisticColors()
            algo.save_result(output_path)
        else:
            print(f"Unknown method: {method}")
            return -1
    except Exception as e:
        print(f"Error: {e}")
        return -1

    print(f"Saved result to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
