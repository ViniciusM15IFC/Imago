import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import time

IMAGEM_ENTRADA = "img/foto.jpeg"  # Coloque sua foto com este nome na mesma pasta
IMAGEM_SAIDA = "img/resultado_pixel_art_v2.png"

# 1. Função para cortar a foto em um quadrado perfeito (evita deformação facial)
def cortar_quadrado_central(img, tamanho=384):
    largura, altura = img.size
    novo_tamanho = min(largura, altura)
    esquerdo = (largura - novo_tamanho) / 2
    superior = (altura - novo_tamanho) / 2
    direito = (largura + novo_tamanho) / 2
    inferior = (altura + novo_tamanho) / 2
    return img.crop((esquerdo, superior, direito, inferior)).resize((tamanho, tamanho))

# 2. Função de Limpeza Matemática (Garante os quadradinhos nítidos de videogame)
def aplicar_limpeza_pixel_art(img_ia, res_retro=64, cores=28):
    # Usa LANCZOS pra não borrar tanto antes de paletizar
    pequena = img_ia.resize((res_retro, res_retro), Image.Resampling.LANCZOS)
    # method=MEDIANCUT preserva melhor gradientes suaves de pele que MAXCOVERAGE
    # dither=NONE evita as manchas/ruído do quantize padrão
    paletizada = pequena.quantize(
        colors=cores,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE
    )
    # Amplia de volta usando Vizinho Mais Próximo para cravar as bordas duras
    return paletizada.convert("RGB").resize((512, 512), Image.Resampling.NEAREST)

if __name__ == "__main__":
    # Garante que a pasta de saída existe (evita erro só depois do render de 4-6min)
    os.makedirs("img", exist_ok=True)

    print("⏳ Carregando motores gráficos do ImagoFX na CPU...")
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    # Inicializa o modelo de IA configurado para a CPU
    # safety_checker=None evita que rostos sejam bloqueados e virem imagem preta
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cpu")

    # Ativa o agendador de passos rápidos (DPM) para economizar tempo no i5
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Carrega a inteligência estética de Pixel Art
    print("🎨 Carregando adaptadores estéticos...")
    pipe.load_lora_weights("SedatAl/pixel-art-LoRa")
    # Funde o LoRA no modelo com peso mais alto, pra ele realmente dominar o estilo
    # Se ficar exagerado/artefatado, tente baixar pra 0.7-0.8
    pipe.fuse_lora(lora_scale=0.55)

    # Gerenciador de memória para os seus 8GB de RAM
    pipe.enable_attention_slicing()

    # Prepara a imagem de entrada
    print("🖼️ Ajustando proporções e enquadramento da foto...")
    img_original = Image.open(IMAGEM_ENTRADA).convert("RGB")
    img_quadrada = cortar_quadrado_central(img_original)

    # IMPORTANTE: confira no model card do SedatAl/pixel-art-LoRa se existe
    # uma trigger word diferente (ex: "pixelart", "pxlart") e ajuste abaixo.
    prompt_positivo = (
        "pixelart, pixel art style, 16-bit retro video game sprite, "
        "crisp pixel edges, flat colors, detailed portrait"
    )
    prompt_negativo = "blurry, photo, realistic, 3d, smooth gradients, oil painting, messy, noise, dithering"

    print("\n⚡ Iniciando renderização profunda na CPU...")
    print("⏱️ Isso levará cerca de 4 a 6 minutos devido ao hardware local. Aguarde...")
    tempo_inicial = time.time()

    # Seed fixa: garante que, ao mudar strength/guidance_scale entre testes,
    # a comparação seja justa (mesmo ponto de partida aleatório)
    generator = torch.Generator("cpu").manual_seed(42)

    # 1º Passo: A IA redesenha a imagem baseada na sua foto
    imagem_ia = pipe(
        prompt=prompt_positivo,
        negative_prompt=prompt_negativo,
        image=img_quadrada,
        strength=0.38,           # baixou de 0.6 -> preserva mais a identidade/cores reais
        guidance_scale=7.5,
        num_inference_steps=30,  # subiu de 20 -> mais nitidez nas bordas
        generator=generator
    ).images[0]

    # 2º Passo: O filtro matemático limpa os borrões e crava os pixels
    print("🧼 Aplicando algoritmo de limpeza e achatamento de pixels duros...")
    imagem_final = aplicar_limpeza_pixel_art(imagem_ia, res_retro=64, cores=28)

    # Salva o arquivo final
    imagem_final.save(IMAGEM_SAIDA)

    tempo_gasto = (time.time() - tempo_inicial) / 60
    print(f"\n✨ Pixel art concluída com sucesso em {tempo_gasto:.2f} minutos!")
    print(f"📂 Verifique o seu sprite pronto em: '{IMAGEM_SAIDA}'")