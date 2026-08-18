import os
import random
import time
import torch

from config import (
    IMAGEM_ENTRADA,
    IMAGEM_SAIDA,
    VRAM_MINIMA_ROTA_PESADA_GB,
)

from pixel import gerar_rota_leve
from cartoon import gerar_cartoon
from realista import gerar_realista

from fundos import sortear_fundo

from utils import detectar_enquadramento


def detectar_rota():
    if not torch.cuda.is_available():
        print("🖥️  Nenhuma GPU CUDA detectada -> CPU")
        return "cpu"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    nome_gpu = torch.cuda.get_device_properties(0).name

    print(f"🖥️  GPU detectada: {nome_gpu} ({vram_gb:.1f}GB VRAM)")

    if vram_gb >= VRAM_MINIMA_ROTA_PESADA_GB:
        print("🚀 VRAM suficiente -> GPU forte")
        return "gpu_forte"

    print(f"⚠️ VRAM abaixo de {VRAM_MINIMA_ROTA_PESADA_GB}GB -> GPU fraca")
    return "gpu_fraca"


def executar_efeito(efeito, dispositivo, caminho_fundo):
    """Executa o efeito escolhido. O fundo já foi sorteado pelo main.py."""

    if efeito == "pixel_art":
        print("\n🕹️ Efeito selecionado: Pixel Art")
        return gerar_rota_leve(
            dispositivo=dispositivo,
            imagem_entrada=IMAGEM_ENTRADA,
            caminho_fundo=caminho_fundo,
        )

    elif efeito == "cartoon":
        print("\n🎨 Efeito selecionado: Cartoon")
        return gerar_cartoon(
            dispositivo=dispositivo,
            imagem_entrada=IMAGEM_ENTRADA,
            caminho_fundo=caminho_fundo,
        )

    elif efeito == "realista":
        print("\n📸 Efeito selecionado: Realista")
        return gerar_realista(
            dispositivo=dispositivo,
            imagem_entrada=IMAGEM_ENTRADA,
            caminho_fundo=caminho_fundo,
        )

    raise ValueError(f"Efeito desconhecido: {efeito}")


if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)

    print("\n" + "=" * 40)
    print("           🎨 IMAGO")
    print("=" * 40)

    # 1. Detectar enquadramento
    print("\n🔎 Analisando enquadramento...")

    from PIL import Image
    imagem = Image.open(IMAGEM_ENTRADA).convert("RGB")
    enquadramento = detectar_enquadramento(imagem)
    print(f"📐 Enquadramento detectado: {enquadramento}")

    # 2. Sortear fundo
    caminho_fundo, estilos = sortear_fundo(enquadramento=enquadramento)

    if caminho_fundo is None:
        print("❌ Não foi possível encontrar um fundo.")
        exit()

    # 3. Sortear efeito compatível com o fundo escolhido
    if not estilos:
        print("❌ O fundo sorteado não possui efeitos compatíveis.")
        exit()

    efeito = random.choice(estilos)
    print(f"🎲 Efeito sorteado: {efeito}")

    # 4. Detectar dispositivo
    rota = detectar_rota()
    dispositivo = "cpu" if rota == "cpu" else "cuda"
    print(f"⚙️ Dispositivo de processamento: {dispositivo}")

    # 5. Executar efeito
    tempo_inicial = time.time()

    imagem_final = executar_efeito(
        efeito=efeito,
        dispositivo=dispositivo,
        caminho_fundo=caminho_fundo,
    )

    # 6. Salvar
    imagem_final.save(IMAGEM_SAIDA)

    tempo_gasto = time.time() - tempo_inicial
    print(f"\n✨ Concluído em {tempo_gasto / 60:.2f} minutos!")
    print(f"📂 Resultado: '{IMAGEM_SAIDA}'")