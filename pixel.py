import os
import torch
from PIL import Image, ImageFilter

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
    "leve": {
        "strength": 0.38,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "lora_scale": 0.55,
        "tamanho_crop": 384,   # resolução quadrada de TRABALHO da IA --
                                # não é mais o crop da foto inteira, é o
                                # canvas onde a pessoa (já recortada) é
                                # centralizada antes de gerar
        "res_retro": 64,
        "cores": 28,
        "resolucao_final": 512,
        "blur_radius": 2,
        "seed": 7,
        "cor_fundo": (30, 30, 30),  # preenchimento neutro só durante a
                                     # geração -- descartado depois, na
                                     # composição com o fundo de verdade
        "prompt_positivo": (
            "pixelart, pixel art style, 16-bit retro video game sprite, "
            "crisp pixel edges, flat colors, detailed portrait, "
            "sharp detailed eyes, clear iris and pupils"
        ),
        "prompt_negativo": "blurry, photo, realistic, 3d, smooth gradients, oil painting, messy, noise, dithering",
    },
}


def aplicar_limpeza_pixel_art(img_ia, resolucao_final, res_retro, cores, blur_radius):
    suavizada = img_ia.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    pequena = suavizada.resize((res_retro, res_retro), Image.Resampling.LANCZOS)
    paletizada = pequena.quantize(
        colors=cores,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE
    )
    return paletizada.convert("RGB").resize((resolucao_final, resolucao_final), Image.Resampling.NEAREST)


def gerar_rota_leve(dispositivo, imagem_entrada, caminho_fundo):
    from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

    cfg = CONFIG["leve"]

    print("⏳ Carregando SD 1.5...")
    dtype = torch.float16 if dispositivo == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(dispositivo)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print("🎨 Carregando LoRA de pixel art (SD 1.5)...")
    pipe.load_lora_weights("SedatAl/pixel-art-LoRa")
    pipe.fuse_lora(lora_scale=cfg["lora_scale"])

    if dispositivo == "cpu":
        pipe.enable_attention_slicing()

    # --- 1. Remover fundo -------------------------------------------------
    img_original = Image.open(imagem_entrada).convert("RGB")
    salvar_debug(img_original, "leve_1_original", DEBUG, PASTA_DEBUG)

    img_rgba = remover_fundo(img_original)
    salvar_mascara_debug(img_rgba, "leve_2_mascara", DEBUG, PASTA_DEBUG)

    img_rgba = preparar_pessoa(img_rgba)

    # --- 2. Crop da pessoa (bbox real, não crop quadrado da foto toda) ----
    pessoa_recortada, mascara_recortada = recortar_pessoa(img_rgba, img_rgba.getchannel("A"))
    salvar_debug(pessoa_recortada, "leve_3_pessoa_recortada", DEBUG, PASTA_DEBUG)

    # --- 3. Canvas quadrado só pra IA trabalhar ----------------------------
    img_quadrada, mascara_quadrada = ajustar_para_quadrado(
        pessoa_recortada, tamanho=cfg["tamanho_crop"], cor_fundo=cfg["cor_fundo"]
    )
    salvar_debug(img_quadrada, "leve_4_canvas_ia", DEBUG, PASTA_DEBUG)

    generator = torch.Generator(dispositivo).manual_seed(cfg["seed"])

    print("⚡ Renderizando (rota leve)...")
    imagem_ia = pipe(
        prompt=cfg["prompt_positivo"],
        negative_prompt=cfg["prompt_negativo"],
        image=img_quadrada,
        strength=cfg["strength"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator
    ).images[0]
    salvar_debug(imagem_ia, "leve_5_saida_ia", DEBUG, PASTA_DEBUG)

    # --- 4. Efeito (pixelização) -------------------------------------------
    resultado = aplicar_limpeza_pixel_art(
        imagem_ia,
        resolucao_final=cfg["resolucao_final"],
        res_retro=cfg["res_retro"],
        cores=cfg["cores"],
        blur_radius=cfg["blur_radius"]
    )
    salvar_debug(resultado, "leve_6_pixelizado", DEBUG, PASTA_DEBUG)

    mascara_final = mascara_quadrada.resize(
        (cfg["resolucao_final"], cfg["resolucao_final"]), Image.Resampling.NEAREST
    )

    # --- 5. Transformar + compor no fundo (posição/escala vêm de FUNDOS) --
    meta = FUNDOS.get(os.path.basename(caminho_fundo), {})
    resultado_composto = compor_fundo_final(
        resultado,
        mascara_final,
        caminho_fundo,
        posicao=meta.get("posicao", (0.5, 0.5)),
        altura_pessoa=meta.get("altura_pessoa", 0.6),
        rotacao=meta.get("rotacao", 0),
        espelhar=meta.get("espelhar", False),
        resample=Image.Resampling.NEAREST,  # preserva bloco duro do pixel art
    )
    salvar_debug(resultado_composto, "leve_7_composto", DEBUG, PASTA_DEBUG)

    return resultado_composto