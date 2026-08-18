import os
from PIL import Image

from utils import (
    preparar_pessoa,
    remover_fundo,
    recortar_pessoa,
    compor_fundo_final,
    salvar_debug,
    salvar_mascara_debug,
)
from config import DEBUG, PASTA_DEBUG
from fundos import FUNDOS


CONFIG = {
    "realista": {
        "aplicar_clahe": True,
    }
}


def gerar_realista(dispositivo, imagem_entrada, caminho_fundo):
    """
    Sem IA nenhuma -- só remove o fundo da foto real e compõe com o fundo
    temático sorteado. Não precisa de canvas quadrado (não há resolução de
    modelo pra respeitar): a pessoa recortada vai direto pra composição.
    """
    cfg = CONFIG["realista"]

    img_original = Image.open(imagem_entrada).convert("RGB")
    salvar_debug(img_original, "realista_1_original", DEBUG, PASTA_DEBUG)

    img_rgba = remover_fundo(img_original)
    salvar_mascara_debug(img_rgba, "realista_2_mascara", DEBUG, PASTA_DEBUG)

    img_rgba = preparar_pessoa(img_rgba, aplicar_normalizacao=cfg["aplicar_clahe"])

    pessoa_recortada, mascara_recortada = recortar_pessoa(img_rgba, img_rgba.getchannel("A"))
    salvar_debug(pessoa_recortada, "realista_3_pessoa_recortada", DEBUG, PASTA_DEBUG)

    meta = FUNDOS.get(os.path.basename(caminho_fundo), {})
    resultado = compor_fundo_final(
        pessoa_recortada.convert("RGB"),
        mascara_recortada,
        caminho_fundo,
        posicao=meta.get("posicao", (0.5, 0.5)),
        altura_pessoa=meta.get("altura_pessoa", 0.6),
        rotacao=meta.get("rotacao", 0),
        espelhar=meta.get("espelhar", False),
        resample=Image.Resampling.LANCZOS,
    )
    salvar_debug(resultado, "realista_4_composto", DEBUG, PASTA_DEBUG)

    return resultado