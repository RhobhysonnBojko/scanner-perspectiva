"""
Scanner de documentos com transformação perspectiva (OpenCV).

Uso:
    python scanner.py imagens/foto.jpg              # detecção automática
    python scanner.py imagens/foto.jpg --manual     # clicar nos 4 cantos
    python scanner.py imagens/foto.jpg --a4         # força proporção A4
    python scanner.py imagens/foto.jpg --cantos x1 y1 x2 y2 x3 y3 x4 y4
"""
import argparse
import os

import cv2
import numpy as np


def ordenar_pontos(pts):
    """Ordena os 4 cantos: sup-esq, sup-dir, inf-dir, inf-esq."""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    soma = pts.sum(axis=1)
    dif = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(soma)],   # sup-esq: menor x+y
        pts[np.argmin(dif)],    # sup-dir: menor y-x
        pts[np.argmax(soma)],   # inf-dir: maior x+y
        pts[np.argmax(dif)],    # inf-esq: maior y-x
    ], dtype="float32")


def achar_documento(img):
    """Procura o maior contorno com 4 vértices (a folha)."""
    escala = 800.0 / max(img.shape[:2])
    peq = cv2.resize(img, None, fx=escala, fy=escala) if escala < 1 else img.copy()
    if escala >= 1:
        escala = 1.0

    cinza = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(cinza, 50, 150)
    bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=1)

    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

    area_min = 0.15 * peq.shape[0] * peq.shape[1]
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            break
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(aprox) == 4:
            return ordenar_pontos(aprox / escala), bordas
    return None, bordas


def selecionar_manual(img):
    """Abre uma janela para o usuário clicar nos 4 cantos."""
    pontos = []
    vis = img.copy()

    def clique(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pontos) < 4:
            pontos.append((x, y))
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Clique nos 4 cantos (ESC cancela)", vis)

    cv2.namedWindow("Clique nos 4 cantos (ESC cancela)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Clique nos 4 cantos (ESC cancela)", clique)
    cv2.imshow("Clique nos 4 cantos (ESC cancela)", vis)
    while len(pontos) < 4:
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    return ordenar_pontos(pontos) if len(pontos) == 4 else None


def corrigir_perspectiva(img, cantos, proporcao=None):
    """Calcula a homografia e retifica a folha.

    proporcao = altura/largura do documento real (A4 = 1.414).
    Se não for informada, é estimada pelo tamanho dos lados na foto.
    """
    tl, tr, br, bl = cantos
    largura = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    altura = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if proporcao:
        altura = int(round(largura * proporcao))

    destino = np.array([[0, 0], [largura - 1, 0],
                        [largura - 1, altura - 1], [0, altura - 1]], dtype="float32")
    H = cv2.getPerspectiveTransform(cantos, destino)
    return cv2.warpPerspective(img, H, (largura, altura)), H


def efeito_scanner(img):
    """Deixa com cara de digitalizado (fundo branco, texto preto)."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # remove sombra/iluminação irregular dividindo pelo fundo estimado
    fundo = cv2.medianBlur(cv2.dilate(cinza, np.ones((7, 7), np.uint8)), 21)
    norm = cv2.divide(cinza, fundo, scale=255)
    return cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 10)


def main():
    ap = argparse.ArgumentParser(description="Scanner de documentos com OpenCV")
    ap.add_argument("imagem")
    ap.add_argument("--manual", action="store_true", help="selecionar cantos com o mouse")
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--cantos", type=float, nargs=8, metavar="XY",
                    help="cantos informados à mão: x1 y1 x2 y2 x3 y3 x4 y4")
    ap.add_argument("--proporcao", type=float, help="altura/largura do documento real")
    ap.add_argument("--a4", action="store_true", help="força proporção de folha A4 (1:1.414)")
    args = ap.parse_args()

    img = cv2.imread(args.imagem)
    if img is None:
        raise SystemExit(f"Não consegui abrir {args.imagem}")
    os.makedirs(args.saida, exist_ok=True)
    nome = os.path.splitext(os.path.basename(args.imagem))[0]

    if args.cantos:
        cantos, bordas = ordenar_pontos(np.array(args.cantos).reshape(4, 2)), None
    else:
        cantos, bordas = (None, None) if args.manual else achar_documento(img)
    if cantos is None:
        print("Documento não encontrado automaticamente, selecione os cantos.")
        cantos = selecionar_manual(img)
        if cantos is None:
            raise SystemExit("Seleção cancelada.")

    corrigida, H = corrigir_perspectiva(img, cantos, 1.414 if args.a4 else args.proporcao)
    scan = efeito_scanner(corrigida)

    marcada = img.copy()
    cv2.polylines(marcada, [cantos.astype(int)], True, (0, 255, 0), 4)
    for p in cantos.astype(int):
        cv2.circle(marcada, tuple(p), 10, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(args.saida, f"{nome}_1_cantos.jpg"), marcada)
    if bordas is not None:
        cv2.imwrite(os.path.join(args.saida, f"{nome}_0_bordas.jpg"), bordas)
    cv2.imwrite(os.path.join(args.saida, f"{nome}_2_corrigida.jpg"), corrigida)
    cv2.imwrite(os.path.join(args.saida, f"{nome}_3_scan.png"), scan)

    np.set_printoptions(precision=4, suppress=True)
    print("Cantos encontrados (TL, TR, BR, BL):\n", cantos)
    print("Homografia H:\n", H)
    print(f"Tamanho da saída: {corrigida.shape[1]}x{corrigida.shape[0]}")
    print(f"Arquivos salvos em {args.saida}/")


if __name__ == "__main__":
    main()
