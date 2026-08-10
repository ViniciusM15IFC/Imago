import os
import time
import cv2
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from PIL import Image, ImageFilter

IMAGEM_ENTRADA = "img/foto.jpeg"
IMAGEM_SAIDA = "img/resultado_pixel_art_gpu.png"


def cortar_quadrado_central(img, tamanho=1024):
    largura, altura = img.size
    novo_tamanho = min(largura, altura)
    esquerdo = (largura - novo_tamanho) / 2
    superior = (altura - novo_tamanho) / 2
    direito = (largura + novo_tamanho) / 2
    inferior = (altura + novo_tamanho) / 2
    return img.crop((esquerdo, superior, direito, inferior)).resize((tamanho, tamanho))


def gerar_mapa_canny(img, low=100, high=200):
    # Gera o mapa de bordas que o ControlNet usa pra travar a estrutura do rosto
    # (posição de olhos, nariz, boca, contorno) enquanto o resto fica livre pra estilizar
    array = np.array(img)
    bordas = cv2.Canny(array, low, high)
    bordas_rgb = np.stack([bordas] * 3, axis=-1)
    return Image.fromarray(bordas_rgb)


def aplicar_limpeza_pixel_art(img_ia, res_retro=128, cores=48):
    # Suaviza transições de luz antes de reduzir (evita quebra dura de tom)
    suavizada = img_ia.filter(ImageFilter.GaussianBlur(radius=2))
    # res_retro mais alto que na versão CPU: com SDXL em 1024px, dá pra ter blocos
    # de pixel maiores em quantidade sem perder legibilidade do rosto
    pequena = suavizada.resize((res_retro, res_retro), Image.Resampling.LANCZOS)
    paletizada = pequena.quantize(
        colors=cores,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE
    )
    return paletizada.convert("RGB").resize((1024, 1024), Image.Resampling.NEAREST)


if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)

    print("⏳ Carregando ControlNet (Canny) para SDXL...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )

    print("⏳ Carregando modelo base SDXL na GPU...")
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print("🎨 Carregando LoRA de pixel art treinado para SDXL...")
    # LoRA específico para SDXL (o da versão CPU era treinado pra SD 1.5, não serve aqui)
    pipe.load_lora_weights("nerijs/pixel-art-xl")
    pipe.fuse_lora(lora_scale=0.7)

    # Com 10GB de VRAM não é obrigatório, mas ajuda a evitar picos de memória
    pipe.enable_vae_slicing()

    print("🖼️ Preparando imagem de entrada e mapa de bordas (Canny)...")
    img_original = Image.open(IMAGEM_ENTRADA).convert("RGB")
    img_quadrada = cortar_quadrado_central(img_original, tamanho=1024)
    mapa_canny = gerar_mapa_canny(img_quadrada)

    prompt_positivo = (
        "pixel art, pixel art style, 16-bit retro video game character portrait, "
        "crisp pixel edges, flat shading, vibrant saturated colors, game sprite, "
        "detailed character art"
    )
    prompt_negativo = "blurry, photo, realistic, 3d render, smooth gradients, noise, dithering, low quality, deformed"

    generator = torch.Generator("cuda").manual_seed(42)

    print("\n⚡ Renderizando na GPU...")
    tempo_inicial = time.time()

    imagem_ia = pipe(
        prompt=prompt_positivo,
        negative_prompt=prompt_negativo,
        image=img_quadrada,
        control_image=mapa_canny,
        strength=0.55,                     # SDXL + ControlNet aguenta um valor maior sem perder identidade
        controlnet_conditioning_scale=0.6,  # quanto o mapa de bordas trava a estrutura do rosto
        guidance_scale=7.5,
        num_inference_steps=50,             # sem pressa de CPU, dá pra caprichar
        generator=generator
    ).images[0]

    print("🧼 Aplicando limpeza e achatamento de pixels...")
    imagem_final = aplicar_limpeza_pixel_art(imagem_ia, res_retro=128, cores=48)
    imagem_final.save(IMAGEM_SAIDA)

    tempo_gasto = time.time() - tempo_inicial
    print(f"\n✨ Concluído em {tempo_gasto:.1f} segundos!")
    print(f"📂 Verifique o resultado em: '{IMAGEM_SAIDA}'")
