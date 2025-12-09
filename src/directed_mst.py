from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Edge:
    u: int  # origem
    v: int  # destino
    w: float  # peso

def directed_mst(
    num_nodes: int, edges: List[Edge], root: int, minimize: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Implementação simples de uma arborescência dirigida inspirada em Chu-Liu/Edmonds.
    Retorna (lista_de_pais, lista_de_pesos_entrada).
    """

    if num_nodes <= 0:
        return [], []

    # Se quisermos maximizar, invertemos os pesos.
    if not minimize:
        edges = [Edge(e.u, e.v, -e.w) for e in edges]

    INF = float("inf")

    parent = [-1] * num_nodes
    in_weight = [0.0] * num_nodes
    best_w = [INF] * num_nodes

    # 1) Escolher, para cada v != root, a aresta de menor peso que entra em v.
    for e in edges:
        if e.v == root:
            continue
        if e.u == e.v:
            continue
        if e.w < best_w[e.v]:
            best_w[e.v] = e.w
            parent[e.v] = e.u
            in_weight[e.v] = e.w

    # 2) Remover ciclos: percorremos cadeias seguindo parent[v]
    visited_global = [False] * num_nodes
    in_stack = [False] * num_nodes

    for start in range(num_nodes):
        v = start
        path: List[int] = []
        while v != -1 and not visited_global[v]:
            if in_stack[v]:
                # Encontramos um ciclo
                cycle_nodes: List[int] = []
                for node in reversed(path):
                    cycle_nodes.append(node)
                    if node == v:
                        break
                if cycle_nodes:
                    # Cortar a aresta de MAIOR peso dentro do ciclo.
                    node_to_cut = max(cycle_nodes, key=lambda x: in_weight[x])
                    parent[node_to_cut] = -1
                    in_weight[node_to_cut] = 0.0
                break
            in_stack[v] = True
            path.append(v)
            v = parent[v]

        # Limpar marcações desta caminhada.
        for node in path:
            visited_global[node] = True
            in_stack[node] = False

    # Se estivermos maximizando, desfazemos o sinal dos pesos de volta.
    if not minimize:
        in_weight = [-w for w in in_weight]

    return parent, in_weight