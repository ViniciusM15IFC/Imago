from PIL import Image, ImageFilter, ImageOps, ImageEnhance


def remover_fundo(img, cor_fundo=None, limiar_alfa=128):
    """
    Remove o fundo da imagem usando o modelo U2-Net (via rembg).
    Retorna imagem RGBA com fundo transparente por padrão.
    Se cor_fundo for passado (ex: (255, 255, 255)), preenche o fundo
    com essa cor sólida em vez de deixar transparente.

    limiar_alfa binariza a máscara de transparência (0 ou 255, sem valores
    intermediários) antes do paste -- evita pixels de "mistura" nas bordas
    (meio pele, meio cor_fundo) que aparecem quando a máscara original do
    rembg tem transição gradual em vez de um corte definido.
    """
    from rembg import remove

    img_sem_fundo = remove(img)  # retorna RGBA com alpha=0 onde é fundo

    if cor_fundo is None:
        return img_sem_fundo

    canal_alfa = img_sem_fundo.split()[-1]
    mascara_binaria = canal_alfa.point(lambda p: 255 if p > limiar_alfa else 0)

    fundo_solido = Image.new("RGBA", img_sem_fundo.size, cor_fundo + (255,))
    fundo_solido.paste(img_sem_fundo, mask=mascara_binaria)
    return fundo_solido.convert("RGB")


def aplicar_clahe(img, clip_limit=1.5, tile_size=8):
    """
    Equalização de histograma adaptativa (CLAHE): ajusta contraste por região
    da imagem, em vez de esticar o histograma inteiro de uma vez (como o
    autocontrast fazia). Resolve o caso de rosto com luz forte de um lado só,
    onde a normalização global deixava metade do rosto sem contraste suficiente.

    Opera no canal L (luminância) do espaço LAB, preservando as cores originais
    e mexendo só no brilho/contraste local.
    """
    import cv2
    import numpy as np

    array = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_equalizado = clahe.apply(l)

    lab_equalizado = cv2.merge((l_equalizado, a, b))
    resultado = cv2.cvtColor(lab_equalizado, cv2.COLOR_LAB2RGB)
    return Image.fromarray(resultado)


def ajustar_contraste(img, fator=1.15):
    """Ajuste fino adicional, leve, depois do CLAHE."""
    return ImageEnhance.Contrast(img).enhance(fator)


def preparar_imagem(img, aplicar_normalizacao=True):
    if aplicar_normalizacao:
        img = aplicar_clahe(img)
        img = ajustar_contraste(img)
    return img


def cortar_quadrado_central(img, tamanho):
    largura, altura = img.size
    novo_tamanho = min(largura, altura)
    esquerdo = (largura - novo_tamanho) / 2
    superior = (altura - novo_tamanho) / 2
    direito = (largura + novo_tamanho) / 2
    inferior = (altura + novo_tamanho) / 2
    return img.crop((esquerdo, superior, direito, inferior)).resize((tamanho, tamanho))


def aplicar_limpeza_pixel_art(img_ia, resolucao_final, res_retro, cores):
    suavizada = img_ia.filter(ImageFilter.GaussianBlur(radius=2))
    pequena = suavizada.resize((res_retro, res_retro), Image.Resampling.LANCZOS)
    paletizada = pequena.quantize(
        colors=cores,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE
    )
    return paletizada.convert("RGB").resize((resolucao_final, resolucao_final), Image.Resampling.NEAREST)
