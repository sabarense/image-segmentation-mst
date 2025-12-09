import numpy as np
from skimage.io import imread, imsave
from skimage.segmentation import slic, mark_boundaries
from skimage import graph, img_as_ubyte
from skimage.color import label2rgb
from directed_mst import directed_mst, Edge


class ChuLiu:
    def __init__(self, image_path: str, k_segments: int):
        self.image_path = image_path

        # Carrega imagem e trata canais (RGBA -> RGB / Cinza -> RGB)
        raw_image = imread(image_path)
        if raw_image.ndim == 3 and raw_image.shape[2] == 4:
            self.image = raw_image[:, :, :3]
        elif raw_image.ndim == 2:
            self.image = gray2rgb(raw_image)
        else:
            self.image = raw_image

        self.target_k = k_segments
        self.labels = None  # SLIC labels
        self.final_labels = None  # Labels após segmentação
        self.segmented_image = None

    def run(self):
        print("1. Gerando Superpixels (SLIC)...")
        # SLIC: Gera os superpixels iniciais
        self.labels = slic(self.image, n_segments=400, compactness=10, sigma=1, start_label=0)

        # Mapeamento: O SLIC retorna labels arbitrários. O algoritmo directed_mst precisa de 0..N.
        unique_labels = np.unique(self.labels)
        num_nodes = len(unique_labels)

        # Mapas: Label_Original <-> Índice_Algoritmo
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        idx_to_label = {i: lbl for i, lbl in enumerate(unique_labels)}

        print(f"   -> Grafo com {num_nodes} nós (superpixels).")

        print("2. Construindo Grafo Dirigido...")
        # RAG (Region Adjacency Graph) do scikit-image calcula pesos baseados na cor
        rag = graph.rag_mean_color(self.image, self.labels)

        # Converte RAG para lista de objetos Edge
        edges_list = []
        for u, v, data in rag.edges(data=True):
            w = data['weight']
            idx_u = label_to_idx[u]
            idx_v = label_to_idx[v]

            # Grafo de imagem é não-dirigido por natureza, então criamos arestas
            # nas duas direções (u->v e v->u) com o mesmo peso.
            edges_list.append(Edge(idx_u, idx_v, w))
            edges_list.append(Edge(idx_v, idx_u, w))

        print("3. Executando Algoritmo Chu-Liu (directed_mst)...")
        # Define raiz arbitrariamente como o nó 0
        root_node = 0
        parents, weights = directed_mst(num_nodes, edges_list, root=root_node, minimize=True)

        print(f"4. Realizando cortes para obter {self.target_k} regiões...")
        self._segment_tree(parents, weights, idx_to_label, num_nodes)

    def _segment_tree(self, parents, weights, idx_to_label, num_nodes):
        """
        Recebe a arborescência calculada, corta as arestas mais pesadas e reagrupa a imagem.
        """
        # Lista de candidatos a corte: (indice_nó, peso_aresta_entrada)
        candidates = []
        for i in range(num_nodes):
            if parents[i] != -1:  # Ignora quem já é raiz
                candidates.append((i, weights[i]))

        # Ordena decrescente: queremos cortar as conexões mais "caras" (maior diferença de cor)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Cálculo de quantos cortes são necessários para chegar em K componentes
        current_roots = parents.count(-1)
        needed_cuts = self.target_k - current_roots

        if needed_cuts > 0:
            for i in range(min(needed_cuts, len(candidates))):
                node_to_cut = candidates[i][0]
                parents[node_to_cut] = -1  # Transforma em nova raiz (corta conexão com pai)

        group_map = [-1] * num_nodes

        def find_root(node):
            path = []
            curr = node
            # Sobe na árvore até achar a raiz (-1)
            while parents[curr] != -1:
                path.append(curr)
                curr = parents[curr]
                # Proteção simples contra loops (caso o algoritmo falhe em remover algum)
                if curr in path: break
            return curr

        # Para cada nó, descobre quem é o "chefe" (raiz)
        for i in range(num_nodes):
            group_map[i] = find_root(i)

        # --- Pintura da Imagem ---
        self.segmented_image = np.zeros_like(self.image)
        self.final_labels = np.zeros_like(self.labels)

        # Agrupa os labels originais do SLIC por Raiz
        groups = {}
        for idx in range(num_nodes):
            root_id = group_map[idx]
            if root_id not in groups: groups[root_id] = []
            groups[root_id].append(idx_to_label[idx])

        # Recupera dados de cor para calcular a média
        rag = graph.rag_mean_color(self.image, self.labels)
        try:
            node_data = {n: (rag.nodes[n]['mean color'], rag.nodes[n]['pixel count']) for n in rag.nodes}
        except KeyError:
            # Fallback para versões antigas do skimage
            node_data = {n: (rag.nodes[n]['mean_color'], rag.nodes[n]['pixel_count']) for n in rag.nodes}

        # Itera sobre cada grupo final
        # new_id começa em 1 para diferenciar do fundo (0) na visualização de bordas
        for new_id, (root_id, slic_nodes) in enumerate(groups.items(), start=1):
            color_acc = np.zeros(3, dtype=float)
            total_pix = 0

            # Média ponderada das cores dos superpixels do grupo
            for slic_n in slic_nodes:
                c, count = node_data[slic_n]
                color_acc += np.array(c) * count
                total_pix += count

            final_color = (color_acc / total_pix).astype(np.uint8) if total_pix > 0 else [0, 0, 0]

            # Aplica na imagem final e no mapa de labels
            for slic_n in slic_nodes:
                mask = (self.labels == slic_n)
                self.segmented_image[mask] = final_color
                self.final_labels[mask] = new_id

    def save_result(self, output_path: str):
        if self.segmented_image is not None:
            # 1. Salva a imagem segmentada (colorida/pintada)
            imsave(output_path, self.segmented_image)
            print(f"   [OK] Resultado salvo: {output_path}")

            # 2. Salva a imagem de BORDAS sobre a ORIGINAL
            path_parts = output_path.rsplit('.', 1)
            bordas_path = f"{path_parts[0]}_bordas.{path_parts[1]}"

            # mark_boundaries desenha linhas amarelas nas fronteiras dos final_labels
            vis_bordas = mark_boundaries(self.image, self.final_labels, color=(1, 1, 0), mode='thick')
            vis_bordas_u8 = img_as_ubyte(vis_bordas)

            imsave(bordas_path, vis_bordas_u8)
            print(f"   [OK] Bordas salvas:   {bordas_path}")
        else:
            print("Erro: Execute .run() primeiro.")
