import torch
from PIL import Image, ImageFilter

from utils import (
    cortar_quadrado_central,
    preparar_imagem,
    remover_fundo,
    salvar_debug,
)
from config import DEBUG, PASTA_DEBUG

# Único ponto de edição pra todos os parâmetros de IA.
# Rota pesada (SDXL + ControlNet) foi descontinuada: rapidez importa demais
# pra esse uso (evento, atração) pra justificar o tempo/hardware que ela pedia.
#
# Tratamento específico de esclera/olho (Haar cascade + forçar branco na
# grade) foi descontinuado também: depois de várias abordagens (pré e pós
# quantize, com/sem espelhamento, validação de formato), o resultado nunca
# convergiu pra algo parecido com olho de verdade -- sempre virava mancha ou
# tarja visível. O prompt pedindo "sharp detailed eyes, clear iris and
# pupils" + o CLAHE já dão o melhor resultado disponível sem esse risco.

CONFIG = {
    "leve": {
        "strength": 0.38,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "lora_scale": 0.55,
        "tamanho_crop": 384,
        "res_retro": 64,
        "cores": 28,
        "resolucao_final": 512,
        "blur_radius": 2,
        "seed": 28,
        "cor_fundo": (30, 30, 30),
        "prompt_positivo": (
            "pixelart, pixel art style, 16-bit retro video game sprite, "
            "crisp pixel edges, flat colors, detailed portrait, "
            "sharp detailed eyes, clear iris and pupils"
        ),
        "prompt_negativo": "blurry, photo, realistic, 3d, smooth gradients, oil painting, messy, noise, dithering, merged hands, extra limbs, disfigured face, blended features",
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
    salvar_debug(img_original, "leve_1_original", DEBUG, PASTA_DEBUG)

    img_original = preparar_imagem(img_original)
    salvar_debug(img_original, "leve_2_clahe", DEBUG, PASTA_DEBUG)

    img_original = remover_fundo(img_original, cor_fundo=cfg["cor_fundo"])
    salvar_debug(img_original, "leve_3_sem_fundo", DEBUG, PASTA_DEBUG)

    img_quadrada = cortar_quadrado_central(img_original, tamanho=cfg["tamanho_crop"])
    salvar_debug(img_quadrada, "leve_4_crop", DEBUG, PASTA_DEBUG)

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

    resultado = aplicar_limpeza_pixel_art(
        imagem_ia,
        resolucao_final=cfg["resolucao_final"],
        res_retro=cfg["res_retro"],
        cores=cfg["cores"],
        blur_radius=cfg["blur_radius"]
    )
    salvar_debug(resultado, "leve_6_final", DEBUG, PASTA_DEBUG)
    return resultado
