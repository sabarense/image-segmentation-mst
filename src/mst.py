from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2

from utils import pixel_index, index_to_coord, is_inside


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.internal_diff = [0.0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def unite(self, a: int, b: int, weight: float) -> None:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.internal_diff[a] = max(self.internal_diff[a], self.internal_diff[b], weight)

    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]

    def get_internal_diff(self, x: int) -> float:
        return self.internal_diff[self.find(x)]


class MST:
    """
    Segmentação baseada na abordagem tipo 'graph-based image segmentation'
    (algoritmo similar ao de Felzenszwalb & Huttenlocher), usando uma heurística k/size.
    """

    def __init__(self, image_path: str, k: float = 500.0) -> None:
        self.image_path = image_path
        self.k = float(k)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Error: Could not open or find the image '{image_path}'.")

        self.image = image
        self.rows, self.cols = image.shape[:2]
        self.num_vertices = self.rows * self.cols
        self.output_image = np.zeros_like(self.image)

    def _build_edges(self) -> List[Tuple[int, int, int]]:
        """Cria lista de arestas (peso, u, v) na 4-vizinhança."""
        edges: List[Tuple[int, int, int]] = []
        img = self.image
        rows, cols = self.rows, self.cols

        for r in range(rows):
            for c in range(cols):
                u = pixel_index(r, c, cols)
                cu = img[r, c].astype(int)
                # 4-vizinhança
                neighbors = [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]
                for nr, nc in neighbors:
                    if not is_inside(nr, nc, rows, cols):
                        continue
                    v = pixel_index(nr, nc, cols)
                    cv = img[nr, nc].astype(int)
                    diff = int(abs(cu[0] - cv[0]) + abs(cu[1] - cv[1]) + abs(cu[2] - cv[2]))
                    if u < v:
                        edges.append((diff, u, v))
        return edges

    def segmentWithRealisticColors(self) -> None:
        edges = self._build_edges()

        edges.sort(key=lambda x: x[0])

        uf = UnionFind(self.num_vertices)

        def threshold(size: int) -> float:
            return self.k / float(size)

        for weight, u, v in edges:
            a = uf.find(u)
            b = uf.find(v)
            if a == b:
                continue
            MIntA = uf.get_internal_diff(a) + threshold(uf.get_size(a))
            MIntB = uf.get_internal_diff(b) + threshold(uf.get_size(b))
            if weight <= min(MIntA, MIntB):
                uf.unite(a, b, float(weight))

        comp_sums: Dict[int, np.ndarray] = {}
        comp_counts: Dict[int, int] = {}

        for idx in range(self.num_vertices):
            root = uf.find(idx)
            r, c = index_to_coord(idx, self.cols)
            if root not in comp_sums:
                comp_sums[root] = np.zeros(3, dtype=np.float64)
                comp_counts[root] = 0
            comp_sums[root] += self.image[r, c]
            comp_counts[root] += 1

        # componentes definidos pelo cálculo da cor média
        comp_color: Dict[int, Tuple[int, int, int]] = {
            root: tuple((comp_sums[root] / comp_counts[root]).astype(int))
            for root in comp_sums
        }

        out = np.zeros_like(self.image)
        for idx in range(self.num_vertices):
            root = uf.find(idx)
            r, c = index_to_coord(idx, self.cols)
            out[r, c] = comp_color[root]

        self.output_image = out.astype(np.uint8)

    def segmentWithRandomColors(self) -> None:
        edges = self._build_edges()
        # Ordena por peso ascendente
        edges.sort(key=lambda x: x[0])

        uf = UnionFind(self.num_vertices)

        def threshold(size: int) -> float:
            return self.k / float(size)

        for weight, u, v in edges:
            a = uf.find(u)
            b = uf.find(v)
            if a == b:
                continue
            MIntA = uf.get_internal_diff(a) + threshold(uf.get_size(a))
            MIntB = uf.get_internal_diff(b) + threshold(uf.get_size(b))
            if weight <= min(MIntA, MIntB):
                uf.unite(a, b, float(weight))

        # componentes de cor aleatória
        comp_color: Dict[int, Tuple[int, int, int]] = {}
        rng = np.random.default_rng(12345)
        out = np.zeros_like(self.image)

        for idx in range(self.num_vertices):
            root = uf.find(idx)
            if root not in comp_color:
                comp_color[root] = (int(rng.integers(0, 256)),
                                    int(rng.integers(0, 256)),
                                    int(rng.integers(0, 256)))
            r, c = index_to_coord(idx, self.cols)
            out[r, c] = comp_color[root]

        self.output_image = out.astype(np.uint8)

    def save_result(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), self.output_image)

    def get_result(self):
        return self.output_image
