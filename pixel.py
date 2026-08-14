import torch

from PIL import Image, ImageFilter

from utils import (
    cortar_quadrado_central,
    preparar_pessoa,
    remover_fundo,
    aplicar_fundo,
    salvar_debug,
    salvar_mascara_debug,
)

from config import DEBUG, PASTA_DEBUG


# ============================================================
# CONFIGURAÇÃO
# ============================================================

CONFIG = {
    "leve": {
        "strength": 0.38,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "lora_scale": 0.55,

        # Deve ser divisível por 8 para funcionar melhor
        # com a arquitetura do Stable Diffusion.
        "tamanho_crop": 384,

        # Resolução da pixelização.
        "res_retro": 64,

        "cores": 28,

        "resolucao_final": 512,

        "blur_radius": 2,

        "seed": 28,

        # Fundo temporário usado atualmente.
        # Posteriormente isso poderá ser substituído
        # pelo fundo específico de cada efeito.
        "cor_fundo": (30, 30, 30),

        "prompt_positivo": (
            "pixelart, pixel art style, "
            "16-bit retro video game sprite, "
            "crisp pixel edges, flat colors, "
            "detailed portrait, "
            "sharp detailed eyes, "
            "clear iris and pupils"
        ),

        "prompt_negativo": (
            "blurry, photo, realistic, 3d, "
            "smooth gradients, oil painting, "
            "messy, noise, dithering, "
            "merged hands, extra limbs, "
            "disfigured face, blended features"
        ),
    },
}


# ============================================================
# PIXELIZAÇÃO
# ============================================================

def aplicar_limpeza_pixel_art(
    img_ia,
    resolucao_final,
    res_retro,
    cores,
    blur_radius
):
    """
    Converte a saída da IA em pixel art.

    Pipeline:

        imagem IA
            ↓
        Gaussian Blur
            ↓
        redução para resolução retro
            ↓
        quantização de cores
            ↓
        aumento com NEAREST
    """

    suavizada = img_ia.filter(
        ImageFilter.GaussianBlur(
            radius=blur_radius
        )
    )

    pequena = suavizada.resize(
        (res_retro, res_retro),
        Image.Resampling.LANCZOS
    )

    paletizada = pequena.quantize(
        colors=cores,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE
    )

    resultado = (
        paletizada
        .convert("RGB")
        .resize(
            (resolucao_final, resolucao_final),
            Image.Resampling.NEAREST
        )
    )

    return resultado


# ============================================================
# ROTA LEVE
# ============================================================

def gerar_rota_leve(
    dispositivo,
    imagem_entrada
):
    """
    Gera a versão pixel art utilizando Stable Diffusion 1.5.

    Pipeline:

        1. Imagem original
        2. Remoção de fundo
        3. Debug da máscara
        4. CLAHE somente na pessoa
        5. Aplicação de fundo temporário
        6. Crop
        7. Stable Diffusion
        8. Pixelização
    """

    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        DPMSolverMultistepScheduler,
    )

    cfg = CONFIG["leve"]

    # ========================================================
    # CARREGAMENTO DO MODELO
    # ========================================================

    print("⏳ Carregando SD 1.5...")

    dtype = (
        torch.float16
        if dispositivo == "cuda"
        else torch.float32
    )

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(dispositivo)

    pipe.scheduler = (
        DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    )

    # ========================================================
    # LORA
    # ========================================================

    print("🎨 Carregando LoRA de pixel art (SD 1.5)...")

    pipe.load_lora_weights(
        "SedatAl/pixel-art-LoRa"
    )

    pipe.fuse_lora(
        lora_scale=cfg["lora_scale"]
    )

    if dispositivo == "cpu":
        pipe.enable_attention_slicing()

    # ========================================================
    # 1 — IMAGEM ORIGINAL
    # ========================================================

    img_original = Image.open(
        imagem_entrada
    ).convert("RGB")

    salvar_debug(
        img_original,
        "leve_1_original",
        DEBUG,
        PASTA_DEBUG
    )

    # ========================================================
    # 2 — REMOÇÃO DE FUNDO
    # ========================================================

    # IMPORTANTE:
    # O rembg recebe a imagem ORIGINAL.
    #
    # Não aplicamos CLAHE antes da segmentação, porque isso
    # pode alterar as características utilizadas pelo modelo
    # para distinguir pessoa e fundo.

    img_sem_fundo = remover_fundo(
        img_original
    )

    salvar_debug(
        img_sem_fundo,
        "leve_2_sem_fundo",
        DEBUG,
        PASTA_DEBUG
    )

    # ========================================================
    # 2 — DEBUG DA MÁSCARA
    # ========================================================

    salvar_mascara_debug(
        img_sem_fundo,
        "leve_2_mascara",
        DEBUG,
        PASTA_DEBUG
    )

    # ========================================================
    # 3 — CLAHE SOMENTE NA PESSOA
    # ========================================================

    img_sem_fundo = preparar_pessoa(
        img_sem_fundo
    )

    salvar_debug(
        img_sem_fundo,
        "leve_3_clahe",
        DEBUG,
        PASTA_DEBUG
    )

    # ========================================================
    # 4 — FUNDO
    # ========================================================

    # Neste momento colocamos o fundo temporário.
    #
    # Posteriormente esta etapa deverá ser substituída pelo
    # sistema de fundos dos efeitos, por exemplo:
    #
    # disaster_girl.png
    # mario.png
    # backrooms.png
    # etc.

    img_com_fundo = aplicar_fundo(
        img_sem_fundo,
        cfg["cor_fundo"]
    )

    salvar_debug(
        img_com_fundo,
        "leve_4_com_fundo",
        DEBUG,
        PASTA_DEBUG
    )

    # ========================================================
    # 5 — CROP
    # ========================================================

    img_quadrada = cortar_quadrado_central(
        img_com_fundo,
        tamanho=cfg["tamanho_crop"]
    )

    salvar_debug(
        img_quadrada,
        "leve_5_crop",
        DEBUG,
        PASTA_DEBUG
    )

    # ========================================================
    # SEED
    # ========================================================

    generator = torch.Generator(
        dispositivo
    ).manual_seed(
        cfg["seed"]
    )

    # ========================================================
    # 6 — STABLE DIFFUSION
    # ========================================================

    print("⚡ Renderizando (rota leve)...")

    imagem_ia = pipe(
        prompt=cfg["prompt_positivo"],
        negative_prompt=cfg["prompt_negativo"],

        image=img_quadrada,

        strength=cfg["strength"],

        guidance_scale=cfg["guidance_scale"],

        num_inference_steps=(
            cfg["num_inference_steps"]
        ),

        generator=generator,
    ).images[0]

    salvar_debug(
        imagem_ia,
        "leve_6_saida_ia",
        DEBUG,
        PASTA_DEBUG
    )

    # ========================================================
    # 7 — PIXELIZAÇÃO
    # ========================================================

    resultado = aplicar_limpeza_pixel_art(
        imagem_ia,

        resolucao_final=(
            cfg["resolucao_final"]
        ),

        res_retro=(
            cfg["res_retro"]
        ),

        cores=cfg["cores"],

        blur_radius=(
            cfg["blur_radius"]
        ),
    )

    salvar_debug(
        resultado,
        "leve_7_final",
        DEBUG,
        PASTA_DEBUG
    )

    return resultado