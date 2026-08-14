import os

from PIL import Image, ImageEnhance


# ============================================================
# DEBUG
# ============================================================

def salvar_debug(img, nome, ativo, pasta):
    """
    Salva a imagem da etapa atual em pasta/nome.png, se ativo=True.

    O nome deve incluir um prefixo numérico para manter a ordem
    do pipeline visível na pasta de debug.

    Exemplo:
        1_original
        2_sem_fundo
        2_mascara
        3_clahe
    """
    if not ativo:
        return

    os.makedirs(pasta, exist_ok=True)

    img.convert("RGB").save(
        os.path.join(pasta, f"{nome}.png")
    )


def salvar_mascara_debug(img_rgba, nome, ativo, pasta):
    """
    Salva somente o canal alpha da imagem RGBA.

    Na máscara:
        preto  = fundo
        branco = pessoa
        cinza  = transparência parcial / transição

    Isso permite verificar exatamente o que o rembg
    considerou como pertencente à pessoa.
    """
    if not ativo:
        return

    os.makedirs(pasta, exist_ok=True)

    mascara = img_rgba.getchannel("A")

    mascara.save(
        os.path.join(pasta, f"{nome}.png")
    )


# ============================================================
# REMOÇÃO DE FUNDO
# ============================================================

# Sessão carregada sob demanda.
# Assim o modelo não é carregado novamente a cada fotografia.
_SESSAO_REMBG = None


def remover_fundo(img):
    """
    Remove o fundo da imagem usando segmentação específica
    para pessoas.

    IMPORTANTE:
    A imagem original é enviada diretamente ao rembg, antes
    de CLAHE ou qualquer outra alteração de contraste.

    Retorna:
        PIL.Image.Image em RGBA

    O canal alpha original da segmentação é preservado.
    Não fazemos binarização da máscara, pois isso poderia
    destruir detalhes de cabelo, roupas e bordas.
    """
    from rembg import remove, new_session

    global _SESSAO_REMBG

    if _SESSAO_REMBG is None:
        print("⏳ Carregando modelo de remoção de fundo...")
        _SESSAO_REMBG = new_session("u2net_human_seg")

    img_sem_fundo = remove(
        img,
        session=_SESSAO_REMBG,

        # Refinamento das bordas.
        alpha_matting=True,

        # Parâmetros do alpha matting.
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=2,
    )

    return img_sem_fundo


# ============================================================
# TRATAMENTO DE IMAGEM
# ============================================================

def aplicar_clahe(
    img,
    clip_limit=2.5,
    tile_size=8,
    intensidade=0.6
):
    """
    Equalização de histograma adaptativa (CLAHE).

    Opera no canal L do espaço LAB, preservando as cores
    originais e alterando principalmente luminosidade/contraste.

    intensidade:
        0.0 = imagem original
        1.0 = CLAHE completo
    """
    import cv2
    import numpy as np

    array = np.array(img.convert("RGB"))

    lab = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    l_equalizado = clahe.apply(l)

    l_misturado = cv2.addWeighted(
        l_equalizado,
        intensidade,
        l,
        1 - intensidade,
        0
    )

    lab_misturado = cv2.merge(
        (l_misturado, a, b)
    )

    resultado = cv2.cvtColor(
        lab_misturado,
        cv2.COLOR_LAB2RGB
    )

    return Image.fromarray(resultado)


def ajustar_contraste(img, fator=1.15):
    """
    Ajuste fino adicional de contraste.
    """
    return ImageEnhance.Contrast(img).enhance(fator)


def preparar_imagem(img, aplicar_normalizacao=True):
    """
    Tratamento normal da imagem RGB.

    Usado quando não há necessidade de preservar alpha.
    """
    if aplicar_normalizacao:
        img = aplicar_clahe(img)
        img = ajustar_contraste(img)

    return img


def preparar_pessoa(img_rgba, aplicar_normalizacao=True):
    """
    Aplica o tratamento de imagem SOMENTE à pessoa.

    O canal alpha é preservado integralmente.

    Isso é importante porque o CLAHE deve melhorar a aparência
    da pessoa sem alterar a máscara criada pelo rembg.
    """
    if not aplicar_normalizacao:
        return img_rgba

    alpha = img_rgba.getchannel("A")

    rgb = img_rgba.convert("RGB")

    rgb = aplicar_clahe(rgb)
    rgb = ajustar_contraste(rgb)

    resultado = rgb.convert("RGBA")

    resultado.putalpha(alpha)

    return resultado


# ============================================================
# COMPOSIÇÃO DE FUNDO
# ============================================================

def aplicar_fundo(img_rgba, cor_fundo):
    """
    Coloca a pessoa RGBA sobre um fundo sólido.

    A máscara alpha original é utilizada para fazer a composição.
    """
    fundo = Image.new(
        "RGBA",
        img_rgba.size,
        cor_fundo + (255,)
    )

    fundo.alpha_composite(img_rgba)

    return fundo.convert("RGB")


# ============================================================
# CROP
# ============================================================

def cortar_quadrado_central(img, tamanho):
    """
    Corta o maior quadrado possível no centro da imagem
    e redimensiona para o tamanho especificado.
    """
    largura, altura = img.size

    novo_tamanho = min(
        largura,
        altura
    )

    esquerdo = (largura - novo_tamanho) / 2
    superior = (altura - novo_tamanho) / 2

    direito = (
        largura + novo_tamanho
    ) / 2

    inferior = (
        altura + novo_tamanho
    ) / 2

    return img.crop(
        (
            esquerdo,
            superior,
            direito,
            inferior
        )
    ).resize(
        (tamanho, tamanho)
    )