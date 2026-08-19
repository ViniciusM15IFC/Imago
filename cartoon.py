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
        "strength": 0.40,           # subiu um pouco de 0.20 -- com lora_scale
                                     # baixo, a IA precisa de mais liberdade
                                     # pra simplificar de verdade
        "guidance_scale": 7.0,
        "num_inference_steps": 30,
        "tamanho_crop": 384,
        "seed": 28,
        "cor_fundo": (90, 90, 90),

        # Reaproveita o LoRA de pixel art (já provado estável), mas com peso
        # BEM mais baixo que os 0.55 usados no pixel art de verdade -- só o
        # suficiente pra puxar a IA na direção de simplificação/achatamento
        # de cor, sem empurrar pra estética de sprite/bloco duro.
        "lora_repo": "SedatAl/pixel-art-LoRa",
        "lora_scale": 0.22,

        "prompt_positivo": (
            "portrait, detailed face, sharp focus, "
            "sharp detailed eyes, clear iris and pupils, high contrast eyes, "
            "cartoon style illustration"
        ),
        "prompt_negativo": (
            "photo, realistic, photorealistic, 3d render, "
            "blurry, soft focus, out of focus, low detail, smooth blur, "
            "noisy, deformed face, distorted eyes, closed eyes, "
            "extra limbs, bad anatomy, messy lineart, "
            "pixelated, blocky, 8-bit, pixel art, "
            "stars, night sky, space, full body, different pose, "
            "changed composition, background elements"
        ),

        # Parâmetros do pós-processamento de cartoonização (achatar cor +
        # desenhar contorno onde uma cor termina e outra começa)
        "achatamento_sp": 15,   # raio espacial do mean-shift -- maior =
                                 # regiões de cor mais largas/simplificadas
        "achatamento_sr": 65,   # subiu de 40 -- fundir sombra e pele
                                 # iluminada na mesma região de cor, pra não
                                 # sobrar fronteira de contraste que o Canny
                                 # desenharia como linha (era o problema da
                                 # divisão dura de rosto que aparecia)
        "canny_low": 40,     # sensibilidade mínima do Canny -- mais baixo =
                              # pega mais transições fracas
        "canny_high": 100,   # sensibilidade máxima -- mais alto = só
                              # transições fortes viram linha
        "contorno_espessura": 1,  # dilatação da linha (1 = fina, 2+ = mais grossa)

        # Realce de brilho -- empurra pixels já claros (esclera, dentes,
        # brilho no olho) ainda mais pra perto do branco. Só afeta pixels
        # ACIMA do limiar; pele/sombra normal não muda. Não depende de
        # detecção de rosto/olho (evita os problemas de mancha/tarja que
        # tivemos antes com abordagem baseada em Haar cascade).
        "realce_limiar": 190,   # a partir de qual brilho (0-255) começa a empurrar
        "realce_ganho": 1.6,    # intensidade do empurrão pro branco

        # Alívio de sombra -- espelho do realce de brilho, mas pro lado
        # escuro: clareia pixels abaixo do limiar proporcionalmente a
        # quão escuros são, deixando a região mais lisa/uniforme. Pixels
        # ACIMA do limiar não são tocados (não mexe em tom médio/claro).
        "sombra_limiar": 90,    # abaixo de qual brilho (0-255) começa a clarear
        "sombra_forca": 0.55,   # intensidade do clareamento (0=nada, 1=forte)
    }
}


def aliviar_sombras(img, limiar, forca):
    """
    Clareia pixels escuros (sombra) proporcionalmente a quão escuros são --
    clareamento MULTIPLICATIVO (não empurra até o branco absoluto como o
    realce de brilho faz), preservando alguma variação de tom em vez de
    achatar tudo pra um cinza uniforme. Pixels acima do limiar não mudam.
    """
    import numpy as np

    array = np.array(img.convert("RGB")).astype(np.float32)
    cinza = array.mean(axis=2)

    peso = np.clip((limiar - cinza) / max(limiar, 1), 0, 1)  # 0 em tom claro, até 1 no mais escuro
    fator_multiplicativo = 1 + peso * forca

    array = array * fator_multiplicativo[..., None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def realcar_brilhos(img, limiar, ganho):
    """
    Empurra pixels já claros (candidatos a esclera do olho, dentes, brilho)
    ainda mais pra perto do branco. Pixels abaixo do limiar não são tocados.
    Aplica-se DEPOIS do achatamento de cor e ANTES do Canny -- assim o
    contorno ainda separa visualmente a esclera clareada da pele/íris ao
    redor.
    """
    import numpy as np

    array = np.array(img.convert("RGB")).astype(np.float32)
    cinza = array.mean(axis=2)

    fator = np.clip((cinza - limiar) / max(255 - limiar, 1), 0, 1) * ganho
    fator = np.clip(fator, 0, 1)[..., None]

    array = array + (255 - array) * fator
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def aplicar_cartoonizacao(img, sp, sr, canny_low, canny_high, contorno_espessura, realce_limiar, realce_ganho, sombra_limiar, sombra_forca):
    """
    Achata a imagem em regiões de cor sólida (mean-shift), alivia sombras,
    realça brilhos, e desenha contorno onde há transição real de intensidade
    (Canny).
    """
    import cv2
    import numpy as np

    array = np.array(img.convert("RGB"))

    achatada_img = Image.fromarray(cv2.pyrMeanShiftFiltering(array, sp=sp, sr=sr))
    achatada_img = aliviar_sombras(achatada_img, limiar=sombra_limiar, forca=sombra_forca)
    achatada_img = realcar_brilhos(achatada_img, limiar=realce_limiar, ganho=realce_ganho)
    achatada = np.array(achatada_img)

    cinza = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    cinza_suave = cv2.bilateralFilter(cinza, d=9, sigmaColor=75, sigmaSpace=75)

    bordas = cv2.Canny(cinza_suave, canny_low, canny_high)
    if contorno_espessura > 1:
        kernel = np.ones((contorno_espessura, contorno_espessura), np.uint8)
        bordas = cv2.dilate(bordas, kernel, iterations=1)

    # Canny devolve branco=borda; pro bitwise_and precisamos do oposto
    # (branco=área lisa que deve passar, preto=borda que deve escurecer)
    bordas_invertido = cv2.bitwise_not(bordas)

    bordas_rgb = cv2.cvtColor(bordas_invertido, cv2.COLOR_GRAY2RGB)
    resultado = cv2.bitwise_and(achatada, bordas_rgb)

    return Image.fromarray(resultado)


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

    print("🎨 Carregando LoRA (reaproveitado do pixel art, peso baixo)...")
    pipe.load_lora_weights(cfg["lora_repo"])
    pipe.fuse_lora(lora_scale=cfg["lora_scale"])

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

    print("🎨 Renderizando (simplificação de cor via IA)...")
    imagem_ia = pipe(
        prompt=cfg["prompt_positivo"],
        negative_prompt=cfg["prompt_negativo"],
        image=img_quadrada,
        strength=cfg["strength"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator,
    ).images[0]
    salvar_debug(imagem_ia, "cartoon_5_saida_ia", DEBUG, PASTA_DEBUG)

    # --- 4. Pós-processamento: achatar cor + contorno -------------------
    print("🖊️  Aplicando contorno de cartoon...")
    imagem_final = aplicar_cartoonizacao(
        imagem_ia,
        sp=cfg["achatamento_sp"],
        sr=cfg["achatamento_sr"],
        canny_low=cfg["canny_low"],
        canny_high=cfg["canny_high"],
        contorno_espessura=cfg["contorno_espessura"],
        realce_limiar=cfg["realce_limiar"],
        realce_ganho=cfg["realce_ganho"],
        sombra_limiar=cfg["sombra_limiar"],
        sombra_forca=cfg["sombra_forca"],
    )
    salvar_debug(imagem_final, "cartoon_6_cartoonizado", DEBUG, PASTA_DEBUG)

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
    salvar_debug(resultado_composto, "cartoon_7_composto", DEBUG, PASTA_DEBUG)

    return resultado_composto