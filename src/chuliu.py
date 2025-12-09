import numpy as np
import networkx as nx
from skimage.io import imread, imsave
from skimage.segmentation import slic, mark_boundaries
from skimage import graph, img_as_ubyte
from skimage.color import gray2rgb


class ChuLiu:
    def __init__(self, image_path: str, k_segments: int):
        self.image_path = image_path

        # Carregamento da imagem
        raw_image = imread(image_path)
        if raw_image.ndim == 3 and raw_image.shape[2] == 4:
            self.image = raw_image[:, :, :3]
        elif raw_image.ndim == 2:
            self.image = gray2rgb(raw_image)
        else:
            self.image = raw_image

        self.target_k = k_segments
        self.labels = None
        self.final_labels = None
        self.segmented_image = None

    def run(self):
        print("1. Gerando Superpixels (SLIC)...")
        # SLIC gera os nós
        self.labels = slic(self.image, n_segments=400, compactness=10, sigma=1, start_label=0)

        unique_labels = np.unique(self.labels)
        print(f"   -> Grafo com {len(unique_labels)} nós.")

        print("2. Construindo Grafo DIRIGIDO (DiGraph)...")
        rag = graph.rag_mean_color(self.image, self.labels)

        # --- MUDANÇA CRUCIAL: Usamos DiGraph (Grafo Dirigido) ---
        G = nx.DiGraph()
        G.add_nodes_from(unique_labels)

        # Encontrar o peso máximo para configurar a raiz virtual depois
        max_w = 0.0

        # Adiciona arestas bidirecionais (u->v e v->u)
        # O algoritmo Chu-Liu vai decidir qual direção é melhor para formar a árvore
        for u, v, data in rag.edges(data=True):
            w = data['weight']
            if w > max_w: max_w = w
            G.add_edge(u, v, weight=w)
            G.add_edge(v, u, weight=w)

        # --- TÉCNICA DA RAIZ VIRTUAL ---
        # Adicionamos um nó fantasma que aponta para TODOS os nós.
        # Isso garante que o algoritmo de Edmonds encontre uma arborescência válida
        # mesmo que o grafo original tenha ilhas desconexas.
        virtual_root = "ROOT"
        virtual_weight = max_w * 10000.0  # Peso gigante para ser usado só em último caso

        for node in unique_labels:
            G.add_edge(virtual_root, node, weight=virtual_weight)

        print("3. Executando Chu-Liu/Edmonds (nx.minimum_spanning_arborescence)...")
        # Esta função implementa EXATAMENTE o algoritmo de Edmonds para MST Dirigida
        msa = nx.minimum_spanning_arborescence(G, attr='weight', default=None)

        print(f"4. Realizando cortes para obter {self.target_k} regiões...")
        self._segment_arborescence(msa, virtual_root)

    def _segment_arborescence(self, msa, virtual_root):
        """
        Processa a arborescência resultante do Chu-Liu.
        """
        # 1. Identificar arestas reais (internas) e raízes base
        real_edges = []
        base_roots = []  # Nós que ficaram filhos da raiz virtual (ilhas originais)

        for u, v, data in msa.edges(data=True):
            if u == virtual_root:
                base_roots.append(v)
            else:
                # Guarda aresta real: (peso, u, v)
                real_edges.append((data['weight'], u, v))

        # Remove a raiz virtual do grafo para ele virar uma floresta
        forest = msa.copy()
        if forest.has_node(virtual_root):
            forest.remove_node(virtual_root)

        # 2. Calcular quantos cortes faltam
        # Se temos 'len(base_roots)' componentes, precisamos chegar em 'target_k'
        current_regions = len(base_roots)
        cuts_needed = self.target_k - current_regions

        print(f"   -> Componentes base (ilhas detectadas): {current_regions}")

        if cuts_needed > 0:
            print(f"   -> Cortando {cuts_needed} arestas internas mais pesadas...")
            # Ordena decrescente (maior peso primeiro)
            real_edges.sort(key=lambda x: x[0], reverse=True)

            for i in range(min(cuts_needed, len(real_edges))):
                u, v = real_edges[i][1], real_edges[i][2]
                if forest.has_edge(u, v):
                    forest.remove_edge(u, v)

        # 3. Agrupamento (Componentes Fracamente Conexos)
        # Como é um grafo dirigido (floresta), usamos weakly connected components
        # para achar quem pertence a qual árvore.
        components = list(nx.weakly_connected_components(forest))
        print(f"   -> Regiões Finais Geradas: {len(components)}")

        # --- Pintura da Imagem ---
        self.segmented_image = np.zeros_like(self.image)
        self.final_labels = np.zeros(self.labels.shape, dtype=int)

        # Fallback seguro para pegar cores
        rag = graph.rag_mean_color(self.image, self.labels)
        try:
            node_data = {n: (rag.nodes[n]['mean color'], rag.nodes[n]['pixel count']) for n in rag.nodes}
        except:
            node_data = {n: (rag.nodes[n]['mean_color'], rag.nodes[n]['pixel_count']) for n in rag.nodes}

        new_id = 1
        for comp_set in components:
            color_acc = np.zeros(3, dtype=float)
            total_pix = 0

            for lbl in comp_set:
                if lbl in node_data:
                    c, count = node_data[lbl]
                    color_acc += np.array(c) * count
                    total_pix += count

            final_color = (color_acc / total_pix).astype(np.uint8) if total_pix > 0 else [0, 0, 0]

            for lbl in comp_set:
                mask = (self.labels == lbl)
                self.segmented_image[mask] = final_color
                self.final_labels[mask] = new_id

            new_id += 1

    def save_result(self, output_path: str):
        if self.segmented_image is not None:
            imsave(output_path, self.segmented_image)
            path_parts = output_path.rsplit('.', 1)
            bordas_path = f"{path_parts[0]}_bordas.{path_parts[1]}"
            vis_bordas = mark_boundaries(self.image, self.final_labels, color=(1, 1, 0), mode='thick')
            vis_bordas_u8 = img_as_ubyte(vis_bordas)
            imsave(bordas_path, vis_bordas_u8)
        else:
            print("Erro: Execute .run() primeiro.")