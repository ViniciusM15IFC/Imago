import os
import time
import torch

from config import IMAGEM_ENTRADA, IMAGEM_SAIDA, VRAM_MINIMA_ROTA_PESADA_GB
from pixel import gerar_rota_leve

def detectar_rota():
    if not torch.cuda.is_available():
        print("🖥️  Nenhuma GPU CUDA detectada -> rota leve (CPU, SD 1.5)")
        return "cpu"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    nome_gpu = torch.cuda.get_device_properties(0).name
    print(f"🖥️  GPU detectada: {nome_gpu} ({vram_gb:.1f}GB VRAM)")

    if vram_gb >= VRAM_MINIMA_ROTA_PESADA_GB:
        print("🚀 VRAM suficiente -> rota pesada (GPU, SDXL + ControlNet)")
        return "gpu_forte"
    else:
        print(f"⚠️  VRAM abaixo de {VRAM_MINIMA_ROTA_PESADA_GB}GB -> rota leve (GPU, SD 1.5) pra evitar OOM")
        return "gpu_fraca"


if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)

    rota = detectar_rota()
    tempo_inicial = time.time()

    if rota == "cpu":
        imagem_final = gerar_rota_leve(dispositivo="cpu", imagem_entrada=IMAGEM_ENTRADA)
    else:
        imagem_final = gerar_rota_leve(dispositivo="cuda", imagem_entrada=IMAGEM_ENTRADA)

    imagem_final.save(IMAGEM_SAIDA)

    tempo_gasto = time.time() - tempo_inicial
    print(f"\n✨ Concluído em {tempo_gasto/60:.2f} minutos!")
    print(f"📂 Verifique o resultado em: '{IMAGEM_SAIDA}'")
