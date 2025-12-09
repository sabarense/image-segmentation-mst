
<h1 align="center"> Segmentação de Imagem Baseada em Grafos em Python </h1>

<p align="center"> <img alt="Python Badge" src="https://img.shields.io/badge/Python-%233776AB?style=for-the-badge&logo=python&logoColor=white">   <img alt="OpenCV Badge" src="https://img.shields.io/badge/OpenCV-%230d1117?style=for-the-badge&logo=opencv&logoColor=white">   <img alt="Computer Vision Badge" src="https://img.shields.io/badge/Computer%20Vision-%230d1117?style=for-the-badge"> </p>

Este projeto implementa a segmentação de imagem baseada em grafos usando algoritmos clássicos de visão computacional:

Segmentação por Árvore Geradora Mínima (MST), baseada no método de Felzenszwalb e Huttenlocher (2004);

Image Foresting Transform (IFT), inspirada na abordagem de Falcão et al. (2004).

Arborescência Dirigida (Chu-Liu/Edmonds), uma abordagem hierárquica aplicada sobre *superpixels*, utilizando grafos dirigidos.

O projeto demonstra como diferentes modelagens de grafos (dirigidos vs. não dirigidos, pixels vs. superpixels) impactam o resultado da segmentação.

---

## 🚀 Início Rápido

> [!NOTE]
> Este projeto requer **Python 3.8+**, **OpenCV**, and **NumPy**.

### 1️⃣ Clone o repositório
```bash
git clone https://github.com/sabarense/image-segmentation-mst
cd image-segmentation-mst
```

### 2️⃣ Instale as dependências
```bash
pip install -r requirements.txt
```

### 3️⃣ Execute a segmentação
```bash
# Para segmentação IFT com 250 sementes
python src/main.py ift images/painted_cat.png 250

# Para segmentação MST com k = 8000
python src/main.py mst images/bw_cat.png 8000

# Para segmentação CHULIU com k = 12
python src/main.py chuliu images/bw_cat.png 12
```

### 4️⃣ Resultados
```bash
Execute um dos algoritmos de segmentação abaixo.
As imagens resultantes serão salvas automaticamente na pasta **`results/`**, organizadas por método:

- `results/ift/` - resultados do Image Foresting Transform 
- `results/mst/` - resultados da Árvore Geradora Mínima
- `results/chuliu/ - resultados da Arborescência (inclui visualização de bordas).
```

## 📁 Project Structure
```
image-segmentation-mst/
 ├── src/
 │   ├── ift.py            # Image Foresting Transform (IFT)
 │   ├── mst.py            # Árvore Geradora Mínima (MST)
 │   ├── chuliu.py         # Lógica da Arborescência e Superpixels
 │   ├── directed_mst.py   # Algoritmo de Chu-Liu/Edmonds puro
 │   ├── utils.py          # Funções utilitárias
 │   └── main.py           # Interface de linha de comando (CLI)
 ├── images/               # Imagens de entrada para teste
 │   ├── bw_cat.png
 │   └── painted_cat.png
 ├── results/              # Resultados da segmentação (gerado automaticamente)
 ├── requirements.txt      # Lista de dependências
 ├── .gitignore
 └── README.md

```
