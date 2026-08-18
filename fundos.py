import os
import random


FUNDOS = {
    "disaster.png": {
        "estilos": ["realista"],
        "enquadramentos": ["rosto", "busto"],
        "posicao": (0.736, 0.555),
        "altura_pessoa": 0.889,
        "rotacao": 0,
        "espelhar": False,
    },

    "backrooms.png": {
        "estilos": ["realista"],
        "enquadramentos": ["corpo_inteiro", "busto"],
        "posicao": (0.489, 0.644),
        "altura_pessoa": 0.701,
        "rotacao": 0,
        "espelhar": False,
    },

    "mine.png": {
        "estilos": ["pixel_art"],
        "enquadramentos": ["corpo_inteiro", "busto"],
        "posicao": (0.315, 0.578),
        "altura_pessoa": 0.835,
        "rotacao": 0,
        "espelhar": False,
    },

    "itsfine.png": {
        "estilos": ["cartoon"],
        "enquadramentos": ["corpo_inteiro", "busto"],
        "posicao": (0.464, 0.623),
        "altura_pessoa": 0.751,
        "rotacao": 0,
        "espelhar": False,
    },
}


def sortear_fundo(enquadramento, pasta=None):
    """
    Sorteia um fundo compatível com o enquadramento.
    Retorna: (caminho_fundo, estilos_compativeis)
    """
    from config import PASTA_FUNDOS
    pasta = pasta or PASTA_FUNDOS

    candidatos = [
        (nome, meta) for nome, meta in FUNDOS.items()
        if enquadramento in meta["enquadramentos"]
    ]

    if not candidatos:
        print(f"⚠️ Nenhum fundo encontrado para o enquadramento '{enquadramento}'. Sorteando entre todos.")
        candidatos = list(FUNDOS.items())

    if not candidatos:
        print("⚠️ Nenhum fundo cadastrado.")
        return None, []

    nome, meta = random.choice(candidatos)
    caminho = os.path.join(pasta, nome)

    print(f"🖼️ Fundo sorteado: {nome}")
    print(f"🎨 Efeitos compatíveis: {', '.join(meta['estilos'])}")

    return caminho, meta["estilos"]