from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ift import IFT
from mst import MST
from chuliu import ChuLiu

def build_output_path(method: str, image_path: str, value: str) -> str:
    p = Path(image_path)
    stem = p.stem
    output_dir = Path("results") / method
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"{stem}_{value}.png")

def main() -> int:
    parser = argparse.ArgumentParser(description="Image segmentation: IFT, MST and Chu-Liu.")
    parser.add_argument("method", choices=["ift", "mst", "chuliu"], help="Method: 'ift', 'mst' or 'chuliu'")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("value", help="IFT: num seeds (int). MST: k (float). ChuLiu: k segments (int).")

    args = parser.parse_args()

    method = args.method.lower()
    image_path = args.image
    value = args.value

    output_path = build_output_path(method, image_path, value)

    # Variável para guardar o algoritmo executado
    algo = None
    try:
        if method == "ift":
            num_seeds = int(value)
            print(f"--- Executing IFT with {num_seeds} seeds ---")
            algo = IFT(image_path, num_seeds=num_seeds)
            algo.run()
            algo.save_result(output_path)

        elif method == "mst":
            k = float(value)
            print(f"--- Executing MST with k={k} ---")
            algo = MST(image_path, k=k)
            # Verifica se usa a versão com cores realistas ou aleatórias
            if hasattr(algo, 'segmentWithRealisticColors'):
                algo.segmentWithRealisticColors()
            else:
                algo.run()
            algo.save_result(output_path)

        elif method == "chuliu":
            num_segments = int(value)
            print(f"--- Executing Chu-Liu with K={num_segments} cuts ---")
            algo = ChuLiu(image_path, k_segments=num_segments)
            algo.run()
            algo.save_result(output_path)

        else:
            print(f"Unknown method: {method}")
            return -1

    except Exception as e:
        print(f"Error executing {method}: {e}")
        import traceback
        traceback.print_exc()
        return -1

    print(f"Saved result to: {output_path}")

    # --- Bloco de Diagnóstico EXCLUSIVO para Chu-Liu ---
    if method == "chuliu":
        final_mask = None

        # Tenta recuperar os labels finais do algoritmo ChuLiu
        if hasattr(algo, 'final_labels') and algo.final_labels is not None:
            final_mask = algo.final_labels
        elif hasattr(algo, 'labels'):
            final_mask = algo.labels

        if final_mask is not None:
            num_regioes = len(np.unique(final_mask))
            k_int = int(value)

            print("\n" + "=" * 40)
            print(f"DIAGNÓSTICO (CHULIU)")
            print(f"Alvo (K): {k_int}")
            print(f"Regiões Reais Encontradas: {num_regioes}")

            # Tolerância de +1 ou -1 é aceitável dependendo da implementação
            if abs(num_regioes - k_int) <= 1:
                print(">> SUCESSO! O algoritmo convergiu corretamente.")
            elif num_regioes > k_int:
                # Se der muito alto, indica problema de conectividade no grafo
                print(f">> AVISO: Super-segmentação ({num_regioes} regiões).")
            else:
                print(f">> AVISO: Sub-segmentação ({num_regioes} regiões).")

            print("=" * 40 + "\n")
        else:
            print("\n[Erro] Não foi possível ler 'final_labels' de ChuLiu para diagnóstico.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())