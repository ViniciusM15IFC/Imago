IMAGEM_ENTRADA = "img/foto.png"
IMAGEM_SAIDA = "img/resultado.png"

# VRAM mínima (em GB) pra considerar a rota "pesada" (SDXL + ControlNet) segura.
# Abaixo disso, mesmo tendo GPU, cai pra rota leve pra evitar out-of-memory.
VRAM_MINIMA_ROTA_PESADA_GB = 8

# Modo debug: quando True, salva uma imagem numerada em PASTA_DEBUG a cada
# etapa do pipeline -- pra inspecionar visualmente onde um problema está
# sendo introduzido, em vez de adivinhar pelo resultado final.
DEBUG = True
PASTA_DEBUG = "img/debug"

# Pasta única com todos os arquivos de fundo -- metadados de cada um (estilo
# compatível, enquadramento, posição, escala, rotação, espelhamento) ficam
# no dict FUNDOS em fundos.py.
PASTA_FUNDOS = "img/fundos"