import os
import torch
from PIL import Image

from utils import (
    preparar_pessoa,
    remover_fundo,
    recortar_pessoa,
    ajustar_para_quadrado,
    compor_fundo_final,
    salvar_debug,
    salvar_mascara_debug,
)
from config import DEBUG, PASTA_DEBUG
from fundos import FUNDOS


CONFIG = {
    "cartoon": {
        "strength": 0.20,
        "guidance_scale": 7.0,
        "num_inference_steps": 30,
        "tamanho_crop": 384,
        "seed": 28,
        "cor_fundo": (90, 90, 90),

        "prompt_positivo": (
            "cartoon portrait, stylized character, flat color, "
            "clean lineart, cel shading, simplified shapes, "
            "expressive face, smooth flat colors, "
            "vibrant colors, detailed illustration"
        ),
        "prompt_negativo": (
            "photo, realistic, photorealistic, 3d render, "
            "blurry, noisy, deformed face, distorted eyes, "
            "extra limbs, bad anatomy, messy lineart, "
            "stars, night sky, space, full body, different pose, "
            "changed composition, background elements"
        ),
    }
}


def gerar_cartoon(dispositivo, imagem_entrada, caminho_fundo):
    from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

    cfg = CONFIG["cartoon"]

    print("⏳ Carregando SD 1.5 para cartoon...")
    dtype = torch.float16 if dispositivo == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(dispositivo)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if dispositivo == "cpu":
        pipe.enable_attention_slicing()

    # --- 1. Remover fundo ---------------------------------------------
    img_original = Image.open(imagem_entrada).convert("RGB")
    salvar_debug(img_original, "cartoon_1_original", DEBUG, PASTA_DEBUG)

    img_rgba = remover_fundo(img_original)
    salvar_mascara_debug(img_rgba, "cartoon_2_mascara", DEBUG, PASTA_DEBUG)

    img_rgba = preparar_pessoa(img_rgba)

    # --- 2. Crop da pessoa (bbox real) ---------------------------------
    pessoa_recortada, mascara_recortada = recortar_pessoa(img_rgba, img_rgba.getchannel("A"))
    salvar_debug(pessoa_recortada, "cartoon_3_pessoa_recortada", DEBUG, PASTA_DEBUG)

    # --- 3. Canvas quadrado só pra IA -----------------------------------
    img_quadrada, mascara_quadrada = ajustar_para_quadrado(
        pessoa_recortada, tamanho=cfg["tamanho_crop"], cor_fundo=cfg["cor_fundo"]
    )
    salvar_debug(img_quadrada, "cartoon_4_canvas_ia", DEBUG, PASTA_DEBUG)

    generator = torch.Generator(dispositivo).manual_seed(cfg["seed"])

    print("🎨 Renderizando cartoon...")
    imagem_final = pipe(
        prompt=cfg["prompt_positivo"],
        negative_prompt=cfg["prompt_negativo"],
        image=img_quadrada,
        strength=cfg["strength"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator,
    ).images[0]
    salvar_debug(imagem_final, "cartoon_5_saida_ia", DEBUG, PASTA_DEBUG)

    # --- 5. Transformar + compor no fundo -------------------------------
    meta = FUNDOS.get(os.path.basename(caminho_fundo), {})
    resultado_composto = compor_fundo_final(
        imagem_final,
        mascara_quadrada,
        caminho_fundo,
        posicao=meta.get("posicao", (0.5, 0.5)),
        altura_pessoa=meta.get("altura_pessoa", 0.6),
        rotacao=meta.get("rotacao", 0),
        espelhar=meta.get("espelhar", False),
        resample=Image.Resampling.LANCZOS,
    )
    salvar_debug(resultado_composto, "cartoon_6_composto", DEBUG, PASTA_DEBUG)

    return resultado_composto