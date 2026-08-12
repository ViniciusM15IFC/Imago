import torch
from PIL import Image, ImageFilter

from utils import cortar_quadrado_central, preparar_imagem, remover_fundo

# Único ponto de edição pra todos os parâmetros de IA das duas rotas.
# Os valores são diferentes entre "leve" e "pesada" de propósito: o ControlNet
# da rota pesada segura a identidade do rosto, então ela aguenta strength/lora_scale
# mais altos sem perder a cara -- na rota leve isso faria o resultado degenerar.

CONFIG = {
    "leve": {
        "strength": 0.40,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "lora_scale": 0.55,
        "tamanho_crop": 384,
        "res_retro": 64,
        "cores": 28,
        "resolucao_final": 512,
        "blur_radius": 2,
        "seed": 42,
        "cor_fundo": (30, 30, 30),
        "prompt_positivo": (
            "pixelart, pixel art style, 16-bit retro video game sprite, "
            "crisp pixel edges, flat colors, detailed portrait, "
            "sharp detailed eyes, clear iris and pupils"
        ),
        "prompt_negativo": "blurry, photo, realistic, 3d, smooth gradients, oil painting, messy, noise, dithering",
    },
    "pesada": {
        "strength": 0.55,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "lora_scale": 0.7,
        "controlnet_conditioning_scale": 0.6,
        "tamanho_crop": 1024,
        "res_retro": 128,
        "cores": 48,
        "resolucao_final": 1024,
        "blur_radius": 2,
        "canny_low": 100,
        "canny_high": 200,
        "seed": 42,
        "cor_fundo": (30, 30, 30),
        "prompt_positivo": (
            "pixel art, pixel art style, 16-bit retro video game character portrait, "
            "crisp pixel edges, flat shading, vibrant saturated colors, game sprite, "
            "detailed character art, sharp detailed eyes, clear iris and pupils"
        ),
        "prompt_negativo": "blurry, photo, realistic, 3d render, smooth gradients, noise, dithering, low quality, deformed",
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


def gerar_rota_leve(dispositivo, imagem_entrada):
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

    img_original = Image.open(imagem_entrada).convert("RGB")
    img_original = preparar_imagem(img_original)
    img_original = remover_fundo(img_original, cor_fundo=cfg["cor_fundo"])
    img_quadrada = cortar_quadrado_central(img_original, tamanho=cfg["tamanho_crop"])

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

    return aplicar_limpeza_pixel_art(
        imagem_ia,
        resolucao_final=cfg["resolucao_final"],
        res_retro=cfg["res_retro"],
        cores=cfg["cores"],
        blur_radius=cfg["blur_radius"]
    )


def gerar_rota_pesada(imagem_entrada):
    import cv2
    import numpy as np
    from diffusers import (
        StableDiffusionXLControlNetImg2ImgPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
    )

    cfg = CONFIG["pesada"]

    def gerar_mapa_canny(img, low, high):
        array = np.array(img)
        bordas = cv2.Canny(array, low, high)
        bordas_rgb = np.stack([bordas] * 3, axis=-1)
        return Image.fromarray(bordas_rgb)

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

    print("🎨 Carregando LoRA de pixel art (SDXL)...")
    pipe.load_lora_weights("nerijs/pixel-art-xl")
    pipe.fuse_lora(lora_scale=cfg["lora_scale"])
    pipe.enable_vae_slicing()

    img_original = Image.open(imagem_entrada).convert("RGB")
    img_original = preparar_imagem(img_original)
    img_original = remover_fundo(img_original, cor_fundo=cfg["cor_fundo"])
    img_quadrada = cortar_quadrado_central(img_original, tamanho=cfg["tamanho_crop"])
    mapa_canny = gerar_mapa_canny(img_quadrada, low=cfg["canny_low"], high=cfg["canny_high"])

    generator = torch.Generator("cuda").manual_seed(cfg["seed"])

    print("⚡ Renderizando (rota pesada)...")
    imagem_ia = pipe(
        prompt=cfg["prompt_positivo"],
        negative_prompt=cfg["prompt_negativo"],
        image=img_quadrada,
        control_image=mapa_canny,
        strength=cfg["strength"],
        controlnet_conditioning_scale=cfg["controlnet_conditioning_scale"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator
    ).images[0]

    return aplicar_limpeza_pixel_art(
        imagem_ia,
        resolucao_final=cfg["resolucao_final"],
        res_retro=cfg["res_retro"],
        cores=cfg["cores"],
        blur_radius=cfg["blur_radius"]
    )
