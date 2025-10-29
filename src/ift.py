from __future__ import annotations
import heapq
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

from utils import pixel_index, index_to_coord, is_inside


class IFT:
    """
    Image Foresting Transform (versão Python, multi-source).
    - Implementação otimizada usando arrays numpy e heapq.
    - Seeds: selecionadas uniformemente pela imagem (melhor que módulo fixo).
    """

    def __init__(self, image_path: str, num_seeds: int = 100) -> None:
        self.image_path = image_path
        self.num_seeds = max(1, int(num_seeds))

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Error: Could not open or find the image '{image_path}'.")

        # Mantemos a imagem no formato BGR do OpenCV
        self.image = image
        self.rows, self.cols = image.shape[:2]
        self.num_vertices = self.rows * self.cols

        # Flattened colors: shape (num_vertices, 3) dtype=int
        self.colors = self.image.reshape(-1, 3).astype(np.int32)

        # Seed mask e lista de seeds (índices lineares)
        self.seeds = self._choose_seeds()

        # Result arrays
        self.distances: np.ndarray = np.full(self.num_vertices, np.iinfo(np.int32).max, dtype=np.int64)
        self.nearest_seed: np.ndarray = np.full(self.num_vertices, -1, dtype=np.int32)
        self.output_image = np.zeros_like(self.image)

    def _choose_seeds(self) -> np.ndarray:
        # Seleciona seeds espalhadas uniformemente (melhor para segmentação regular)
        if self.num_seeds >= self.num_vertices:
            return np.arange(self.num_vertices, dtype=np.int32)

        seeds = np.linspace(0, self.num_vertices - 1, num=self.num_seeds, dtype=np.int32)
        return seeds

    @staticmethod
    def _neighbor_indices(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
        # 4-vizinhança: up, down, left, right
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [(nr, nc) for nr, nc in candidates if 0 <= nr < rows and 0 <= nc < cols]

    def build_and_segment(self) -> None:
        """
        Roda a IFT (multi-source Dijkstra-like) sem construir uma estrutura de grafo explícita.
        Calcula pesos sob demanda (diferença L1 entre cores).
        """
        # Inicializa heap com seeds
        heap: List[Tuple[int, int]] = []
        for s in self.seeds:
            self.distances[s] = 0
            self.nearest_seed[s] = int(s)
            heapq.heappush(heap, (0, int(s)))

        # Enquanto houver vértices no heap
        while heap:
            dist_u, u = heapq.heappop(heap)
            if dist_u > self.distances[u]:
                continue

            r, c = index_to_coord(u, self.cols)
            color_u = self.colors[u]

            for nr, nc in self._neighbor_indices(r, c, self.rows, self.cols):
                v = pixel_index(nr, nc, self.cols)
                color_v = self.colors[v]

                # peso = diferença L1 entre cores BGR
                weight = int(abs(int(color_u[0]) - int(color_v[0])) +
                             abs(int(color_u[1]) - int(color_v[1])) +
                             abs(int(color_u[2]) - int(color_v[2])))

                new_dist = dist_u + weight
                if new_dist < self.distances[v]:
                    self.distances[v] = new_dist
                    self.nearest_seed[v] = self.nearest_seed[u]
                    heapq.heappush(heap, (new_dist, int(v)))

    def run(self) -> None:
        """Executa todo o pipeline: segmenta e cria imagem de saída colorida pelos seeds."""
        self.build_and_segment()

        # Preencher imagem de saída com a cor do seed associado
        # nearest_seed contém índices lineares de seeds (referência para vertices)
        seed_colors = self.colors[self.nearest_seed]  # broadcasting; se -1, gera warning, mas aqui seeds cobrem tudo
        # Reformatar para a forma da imagem
        self.output_image = seed_colors.reshape((self.rows, self.cols, 3)).astype(np.uint8)

    def save_result(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.output_image)

    def get_result(self):
        return self.output_image
