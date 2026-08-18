"""
Calibração visual de fundos.

Uso:
    python calibrar_fundo.py caminho/da/copia_marcada.png

Fluxo:
    1. Duplique o arquivo de fundo (o original limpo continua intocado,
       é ele que vai pra produção).
    2. Na cópia, pinte um retângulo sólido na COR_MARCADOR (magenta puro
       por padrão) exatamente onde/do tamanho que a pessoa deve ocupar.
       Se quiser testar rotação, desenhe o retângulo já inclinado.
    3. Rode este script apontando pra cópia marcada.
    4. Copia o bloco impresso (posicao/altura_pessoa/rotacao) pro dict
       FUNDOS em fundos.py, na entrada do arquivo LIMPO (sem o marcador).
"""
import sys

import cv2
import numpy as np

COR_MARCADOR = (255, 0, 255)  # magenta puro (R,G,B) -- raro em fotos reais
TOLERANCIA = 40  # quão perto da cor exata um pixel precisa estar pra contar


def calibrar(caminho_imagem_marcada):
    img = cv2.imread(caminho_imagem_marcada)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {caminho_imagem_marcada}")

    altura, largura = img.shape[:2]

    # OpenCV lê em BGR, não RGB -- inverte a cor de referência
    alvo_bgr = np.array(COR_MARCADOR[::-1])
    diferenca = np.abs(img.astype(int) - alvo_bgr).sum(axis=2)
    mascara = (diferenca < TOLERANCIA).astype(np.uint8) * 255

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise ValueError(
            "Nenhuma região com a cor marcador foi encontrada. "
            f"Confirma que pintou em RGB{COR_MARCADOR} e que a imagem é essa cópia marcada, não o fundo limpo."
        )

    maior = max(contornos, key=cv2.contourArea)
    (cx, cy), (w, h), angulo = cv2.minAreaRect(maior)

    # minAreaRect não garante qual lado é "altura" -- usa sempre o maior
    # como altura da pessoa (retângulos de marcação tendem a ser mais
    # altos que largos, como o corpo de uma pessoa)
    altura_retangulo = max(w, h)

    posicao = (round(cx / largura, 3), round(cy / altura, 3))
    altura_pessoa = round(altura_retangulo / altura, 3)

    # cv2.minAreaRect descreve um retângulo SEM rotação de duas formas
    # equivalentes (ângulo=0 com w/h numa ordem, ou ângulo=90 com w/h
    # trocados) -- é ambiguidade de representação, não inclinação real.
    # Inclinação genuína sempre cai estritamente ENTRE 0 e 90 (nunca bem
    # em cima do valor redondo). Por isso, se o ângulo estiver muito perto
    # de 0 ou 90, tratamos como "sem rotação".
    EPSILON_EIXO = 1.0
    perto_de_zero = abs(angulo) < EPSILON_EIXO
    perto_de_noventa = abs(abs(angulo) - 90) < EPSILON_EIXO

    if perto_de_zero or perto_de_noventa:
        rotacao = 0
    else:
        rotacao = round(angulo, 1)

    print(f"\n📐 Retângulo detectado: centro=({cx:.0f}, {cy:.0f})px, altura={altura_retangulo:.0f}px, ângulo={angulo:.1f}°")
    print("\nCopie isto pra entrada do FUNDO LIMPO em fundos.py:\n")
    print(f'    "posicao": {posicao},')
    print(f'    "altura_pessoa": {altura_pessoa},')
    print(f'    "rotacao": {rotacao},')
    print(
        "\n⚠️  Confirma o sinal/direção da rotação testando: o ângulo que o "
        "OpenCV devolve às vezes precisa ser invertido (-rotacao) dependendo "
        "de como o retângulo foi desenhado. Roda uma vez, olha o resultado, "
        "ajusta o sinal se a pessoa girar pro lado errado."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python calibrar_fundo.py caminho/da/copia_marcada.png")
        sys.exit(1)

    calibrar(sys.argv[1])