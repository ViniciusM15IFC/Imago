from PIL import Image, ImageFilter


def cortar_quadrado_central(img, tamanho):
    largura, altura = img.size
    novo_tamanho = min(largura, altura)
    esquerdo = (largura - novo_tamanho) / 2
    superior = (altura - novo_tamanho) / 2
    direito = (largura + novo_tamanho) / 2
    inferior = (altura + novo_tamanho) / 2
    return img.crop((esquerdo, superior, direito, inferior)).resize((tamanho, tamanho))


