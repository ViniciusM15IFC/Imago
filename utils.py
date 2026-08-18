import os
import random

from PIL import Image, ImageEnhance


# ============================================================
# DEBUG
# ============================================================

def salvar_debug(img, nome, ativo, pasta):
    if not ativo:
        return
    os.makedirs(pasta, exist_ok=True)
    img.convert("RGB").save(os.path.join(pasta, f"{nome}.png"))


def salvar_mascara_debug(img_rgba_ou_mascara, nome, ativo, pasta):
    """
    Salva uma máscara de transparência. Aceita tanto uma imagem RGBA
    (extrai o canal A) quanto uma imagem já em modo "L".
    """
    if not ativo:
        return
    os.makedirs(pasta, exist_ok=True)
    mascara = img_rgba_ou_mascara.getchannel("A") if img_rgba_ou_mascara.mode == "RGBA" else img_rgba_ou_mascara
    mascara.save(os.path.join(pasta, f"{nome}.png"))


# ============================================================
# REMOÇÃO DE FUNDO
# ============================================================

_SESSAO_REMBG = None


def remover_fundo(img, lado_maximo=1200):
    """
    Remove o fundo usando segmentação específica para pessoas. Recebe a
    foto crua (antes de CLAHE). Retorna RGBA, com alpha matting pra
    preservar detalhe de cabelo/borda.

    O alpha matting (estimate_alpha_cf) monta uma matriz esparsa cujo
    tamanho cresce com o número de pixels -- em fotos de resolução alta
    (câmera de celular, facilmente 3000-4000px+) isso estoura memória
    (visto na prática: ArrayMemoryError tentando alocar ~2GB). Por isso,
    a imagem é reduzida ANTES do matting se ultrapassar lado_maximo --
    o resultado final não depende da foto de entrada ser gigante, já que
    a pessoa é redimensionada de qualquer forma mais adiante no pipeline
    (pro canvas da IA, ou pra altura dela dentro do fundo escolhido).
    """
    from rembg import remove, new_session

    global _SESSAO_REMBG
    if _SESSAO_REMBG is None:
        print("⏳ Carregando modelo de remoção de fundo...")
        _SESSAO_REMBG = new_session("u2net_human_seg")

    maior_lado = max(img.width, img.height)
    if maior_lado > lado_maximo:
        escala = lado_maximo / maior_lado
        novo_tamanho = (int(img.width * escala), int(img.height * escala))
        print(f"🔽 remover_fundo: reduzindo {img.width}x{img.height} -> {novo_tamanho[0]}x{novo_tamanho[1]} antes do matting")
        img = img.resize(novo_tamanho, Image.Resampling.LANCZOS)

    return remove(
        img,
        session=_SESSAO_REMBG,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=2,
    )


# ============================================================
# TRATAMENTO DE IMAGEM (CLAHE)
# ============================================================

def aplicar_clahe(img, clip_limit=2.5, tile_size=8, intensidade=0.6):
    import cv2
    import numpy as np

    array = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_equalizado = clahe.apply(l)
    l_misturado = cv2.addWeighted(l_equalizado, intensidade, l, 1 - intensidade, 0)

    lab_misturado = cv2.merge((l_misturado, a, b))
    resultado = cv2.cvtColor(lab_misturado, cv2.COLOR_LAB2RGB)
    return Image.fromarray(resultado)


def ajustar_contraste(img, fator=1.15):
    return ImageEnhance.Contrast(img).enhance(fator)


def preparar_imagem(img, aplicar_normalizacao=True):
    if aplicar_normalizacao:
        img = aplicar_clahe(img)
        img = ajustar_contraste(img)
    return img


def preparar_pessoa(img_rgba, aplicar_normalizacao=True):
    """CLAHE só na pessoa, preservando o canal alpha."""
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
# RECORTE E ENQUADRAMENTO DA PESSOA
# ============================================================

def recortar_pessoa(imagem, mascara, margem=0.05):
    """
    Recorta somente a região onde existe pessoa, com margem extra. imagem
    e mascara precisam estar na mesma geometria.
    """
    bbox = mascara.getbbox()
    if bbox is None:
        return imagem, mascara

    esquerda, superior, direita, inferior = bbox
    largura = direita - esquerda
    altura = inferior - superior

    margem_x = int(largura * margem)
    margem_y = int(altura * margem)

    esquerda = max(0, esquerda - margem_x)
    superior = max(0, superior - margem_y)
    direita = min(imagem.width, direita + margem_x)
    inferior = min(imagem.height, inferior + margem_y)

    return (
        imagem.crop((esquerda, superior, direita, inferior)),
        mascara.crop((esquerda, superior, direita, inferior)),
    )


def ajustar_para_quadrado(img_rgba, tamanho, cor_fundo, resample=Image.Resampling.LANCZOS):
    """
    Centraliza a pessoa (RGBA, já recortada por recortar_pessoa) num canvas
    quadrado preenchido com cor_fundo neutra, e redimensiona pro tamanho
    final. Preserva a proporção da pessoa -- sem esticar/deformar -- porque
    o quadrado é só o espaço de trabalho que a IA precisa (resolução
    quadrada, múltiplo de 8), não o formato do fundo real (tratado à parte
    em compor_fundo_final, cada fundo com sua própria resolução/proporção).

    Retorna (imagem_rgb, mascara), ambos tamanho x tamanho, na mesma
    geometria -- a máscara serve pra recortar só a pessoa de volta do
    resultado estilizado depois, descartando o que a IA desenhar por cima
    do preenchimento neutro.
    """
    lado = max(img_rgba.width, img_rgba.height)
    canvas = Image.new("RGBA", (lado, lado), cor_fundo + (255,))
    offset_x = (lado - img_rgba.width) // 2
    offset_y = (lado - img_rgba.height) // 2
    canvas.alpha_composite(img_rgba, (offset_x, offset_y))

    mascara_canvas = Image.new("L", (lado, lado), 0)
    mascara_canvas.paste(img_rgba.getchannel("A"), (offset_x, offset_y))

    imagem_final = canvas.resize((tamanho, tamanho), resample).convert("RGB")
    mascara_final = mascara_canvas.resize((tamanho, tamanho), resample)

    return imagem_final, mascara_final


def detectar_enquadramento(img, limiar_rosto=0.50, limiar_busto=0.20):
    """
    Classifica em "rosto", "busto" ou "corpo_inteiro", considerando o MESMO
    crop quadrado central que seria aplicado à imagem inteira -- garante que
    a classificação não seja distorcida pela proporção original da foto
    (ex: foto vertical 9:16 sempre "perderia" corpo nas laterais do crop).

    Usado pelo main.py pra escolher fundo/efeito compatível -- não influencia
    mais o processamento dentro de pixel.py/cartoon.py/realista.py, que agora
    recortam a pessoa pelo bbox real da máscara, não por um crop quadrado.
    """
    import cv2
    import numpy as np

    array = np.array(img.convert("RGB"))
    altura, largura = array.shape[:2]

    tamanho_crop = min(largura, altura)
    esquerdo = (largura - tamanho_crop) // 2
    superior = (altura - tamanho_crop) // 2
    array_crop = array[superior:superior + tamanho_crop, esquerdo:esquerdo + tamanho_crop]

    cinza = cv2.cvtColor(array_crop, cv2.COLOR_RGB2GRAY)
    altura_imagem = array_crop.shape[0]

    classificador = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rostos = classificador.detectMultiScale(cinza, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))

    if len(rostos) == 0:
        print("🖼️  detectar_enquadramento: rosto não encontrado no crop, assumindo 'busto'")
        return "busto"

    _, _, _, raltura = max(rostos, key=lambda r: r[2] * r[3])
    proporcao = raltura / altura_imagem

    if proporcao >= limiar_rosto:
        resultado = "rosto"
    elif proporcao >= limiar_busto:
        resultado = "busto"
    else:
        resultado = "corpo_inteiro"

    print(f"🖼️  detectar_enquadramento: rosto no crop = {proporcao:.2f} -> '{resultado}'")
    return resultado


# ============================================================
# COMPOSIÇÃO FINAL COM O FUNDO
# ============================================================

def aplicar_fundo(img_rgba, cor_fundo):
    """Coloca a pessoa RGBA sobre um fundo sólido, usando o alpha original."""
    fundo = Image.new("RGBA", img_rgba.size, cor_fundo + (255,))
    fundo.alpha_composite(img_rgba)
    return fundo.convert("RGB")


def compor_fundo_final(
    imagem_resultado,
    mascara,
    caminho_fundo,
    posicao=(0.5, 0.5),
    altura_pessoa=0.6,
    rotacao=0,
    espelhar=False,
    resample=Image.Resampling.LANCZOS,
):
    """
    Composição da pessoa (já estilizada) sobre um fundo, com controle por
    fundo de posição/tamanho/rotação/espelhamento (valores vêm de
    fundos.FUNDOS, lidos pelo módulo de efeito antes de chamar essa função).

    O fundo mantém sua resolução/proporção NATIVA -- não é forçado a
    512x512 nem a nenhum tamanho fixo. posicao é o centro da pessoa em
    porcentagem do fundo; altura_pessoa é a fração da altura do fundo que
    a pessoa deve ocupar.
    """
    fundo = Image.open(caminho_fundo).convert("RGB")
    largura_fundo, altura_fundo = fundo.size

    pessoa, mascara = recortar_pessoa(imagem_resultado, mascara, margem=0.05)

    if espelhar:
        pessoa = pessoa.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mascara = mascara.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    altura_desejada = int(altura_fundo * altura_pessoa)
    proporcao = pessoa.width / pessoa.height
    largura_desejada = int(altura_desejada * proporcao)

    pessoa = pessoa.resize((largura_desejada, altura_desejada), resample)
    mascara = mascara.resize((largura_desejada, altura_desejada), resample)

    if rotacao != 0:
        pessoa = pessoa.rotate(rotacao, expand=True, resample=resample)
        mascara = mascara.rotate(rotacao, expand=True, resample=resample)

    centro_x = int(largura_fundo * posicao[0])
    centro_y = int(altura_fundo * posicao[1])
    esquerda = int(centro_x - pessoa.width / 2)
    superior = int(centro_y - pessoa.height / 2)

    fundo_rgba = fundo.convert("RGBA")
    pessoa_rgba = pessoa.convert("RGBA")
    pessoa_rgba.putalpha(mascara)

    fundo_rgba.alpha_composite(pessoa_rgba, (esquerda, superior))
    return fundo_rgba.convert("RGB")


# ============================================================
# CROP (mantido -- ainda usado como utilitário genérico)
# ============================================================

def cortar_quadrado_central(img, tamanho):
    largura, altura = img.size
    novo_tamanho = min(largura, altura)
    esquerdo = (largura - novo_tamanho) / 2
    superior = (altura - novo_tamanho) / 2
    direito = (largura + novo_tamanho) / 2
    inferior = (altura + novo_tamanho) / 2
    return img.crop((esquerdo, superior, direito, inferior)).resize((tamanho, tamanho))