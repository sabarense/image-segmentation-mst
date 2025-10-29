from typing import Tuple


def is_inside(row: int, col: int, rows: int, cols: int) -> bool:
    """Verifica se a posição (row, col) está dentro da imagem."""
    return 0 <= row < rows and 0 <= col < cols


def pixel_index(row: int, col: int, cols: int) -> int:
    """Converte coordenadas 2D para índice linear (row-major)."""
    return row * cols + col


def index_to_coord(index: int, cols: int) -> Tuple[int, int]:
    """Converte índice linear para coordenadas (row, col)."""
    return divmod(index, cols)
