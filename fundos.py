import os
import random
from config import DEBUG

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
    Sorteia (ou escolhe manualmente se DEBUG=True) um fundo compatível com o enquadramento.
    Retorna: (caminho_fundo, estilos_compativeis)
    """
    from config import PASTA_FUNDOS, DEBUG
    pasta = pasta or PASTA_FUNDOS

    candidatos = [
        (nome, meta) for nome, meta in FUNDOS.items()
        if enquadramento in meta["enquadramentos"]
    ]

    if not candidatos:
        print(f"⚠️ Nenhum fundo encontrado para o enquadramento '{enquadramento}'. Usando todos disponíveis.")
        candidatos = list(FUNDOS.items())

    if not candidatos:
        print("⚠️ Nenhum fundo cadastrado.")
        return None, []

    # Se o modo DEBUG estiver ativo, permite escolher manualmente pelo terminal
    if DEBUG:
        print("\n--- 🛠️ MODO DEBUG: Escolha o fundo manualmente ---")
        print("0. Sorteio automático (aleatório)")
        
        for i, (nome, meta) in enumerate(candidatos, start=1):
            estilos_str = ", ".join(meta['estilos'])
            print(f"{i}. {nome} (Estilos: {estilos_str})")

        while True:
            try:
                escolha = input(f"\nDigite o número da opção (0 a {len(candidatos)}): ").strip()
                escolha_idx = int(escolha)
                
                if escolha_idx == 0:
                    nome, meta = random.choice(candidatos)
                    print(f"🎲 Sorteio automático selecionado.")
                    break
                elif 1 <= escolha_idx <= len(candidatos):
                    nome, meta = candidatos[escolha_idx - 1]
                    break
                else:
                    print("❌ Opção inválida. Tente novamente.")
            except ValueError:
                print("❌ Por favor, digite apenas números inteiros.")
    else:
        # Comportamento padrão (aleatório) quando DEBUG for False
        nome, meta = random.choice(candidatos)

    caminho = os.path.join(pasta, nome)

    print(f"🖼️ Fundo selecionado: {nome}")
    print(f"🎨 Efeitos compatíveis: {', '.join(meta['estilos'])}")

    return caminho, meta["estilos"]