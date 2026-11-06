# Scanner de documentos com transformação perspectiva (OpenCV)

Atividade 3 – Visão Computacional. A partir de uma foto de um documento tirada
em ângulo, o programa encontra os 4 cantos da folha, calcula a homografia e gera
uma versão "de cima", com aparência de documento escaneado.

## Como funciona
1. Bordas com Canny e busca do maior contorno com 4 vértices (`approxPolyDP`).
2. Ordenação dos cantos (sup-esq, sup-dir, inf-dir, inf-esq).
3. `cv2.getPerspectiveTransform` calcula a matriz H (3x3).
4. `cv2.warpPerspective` retifica a folha.
5. Normalização de iluminação + `adaptiveThreshold` para o efeito de scanner.

Se a folha não for encontrada automaticamente, abre uma janela para clicar nos 4 cantos.

## Instalação
```bash
pip install -r requirements.txt
```

## Uso
```bash
# detecção automática (proporção de página de livro 14x21 cm)
python scanner.py imagens/foto1.jpg --proporcao 1.5

# forçando proporção A4
python scanner.py imagens/minha_foto.jpg --a4

# cantos com o mouse
python scanner.py imagens/minha_foto.jpg --manual

# cantos por coordenadas (usado na contracapa e na página 112)
python scanner.py imagens/foto3.jpg --proporcao 1.5 --cantos 310 78 888 352 1082 1316 306 1500
python scanner.py imagens/foto4.jpg --proporcao 1.5 --cantos 270 250 970 190 1095 1530 60 1500
```

As saídas ficam em `resultados/`:
- `*_0_bordas.jpg` – mapa de bordas
- `*_1_cantos.jpg` – cantos usados
- `*_2_corrigida.jpg` – imagem retificada
- `*_3_scan.png` – versão preto e branco

## Imagens de teste
Fotos de celular do livro *Além do Bem e do Mal* (Nietzsche):
- `foto1.jpg` – página 59 (detecção automática)
- `foto2.jpg` – página 82 (detecção automática)
- `foto3.jpg` – contracapa (cantos manuais)
- `foto4.jpg` – página 112, bem curvada (cantos manuais, mostra a limitação do método)

## Estrutura
```
scanner.py          # aplicação principal
imagens/            # imagens de entrada
resultados/         # imagens geradas
```

## Autor
Rhobhysonn Bojko
