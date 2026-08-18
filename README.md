````markdown
# 🎨 Imago

Projeto de edição automática de imagens desenvolvido para a **FICE / Fábrica de Software**.

O Imago recebe uma fotografia e aplica automaticamente diferentes efeitos visuais, utilizando Inteligência Artificial e processamento tradicional de imagens.

Atualmente, o projeto possui efeitos como:

- 🕹️ Pixel Art
- 🎨 Cartoon
- 📷 Realista

O sistema também possui um sistema de **remoção de fundo, detecção de enquadramento e aplicação de fundos temáticos**, permitindo criar resultados diferentes para cada efeito.

---

# 1. Requisitos

Recomenda-se:

- Python 3.11
- Windows 10/11
- Pelo menos 8 GB de RAM
- GPU NVIDIA com CUDA é recomendada para acelerar os efeitos de IA

> O projeto também funciona em CPU, porém os efeitos baseados em Stable Diffusion podem ser bastante lentos.

---

# 2. Clonar o projeto

Clone o repositório:

```powershell
git clone https://github.com/ViniciusM15IFC/Imago.git
````

Entre na pasta:

```powershell
cd Imago
```

---

# 3. Criar o ambiente virtual

Crie o ambiente virtual:

```powershell
python -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se aparecer `(.venv)` no início da linha do terminal, o ambiente está ativado.

Exemplo:

```text
(.venv) PS C:\...\Imago>
```

> O projeto deve ser executado com o ambiente virtual ativado.

---

# 4. Instalar o PyTorch

## Computador com GPU NVIDIA

Para utilizar uma GPU NVIDIA compatível com CUDA, instale o PyTorch com suporte à GPU:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> A versão do PyTorch/CUDA deve ser compatível com a GPU e com os drivers instalados.

### Testar CUDA

Execute:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NENHUMA')"
```

Se a GPU estiver disponível, o resultado será semelhante a:

```text
PyTorch: 2.x.x+cu128
CUDA: 12.8
CUDA disponível: True
GPU: NVIDIA GeForce RTX 3080
```

---

## Computador sem GPU NVIDIA

Caso o computador não possua uma GPU NVIDIA compatível com CUDA, o projeto pode ser executado em CPU.

Nesse caso, instale a versão apropriada do PyTorch para CPU:

```powershell
python -m pip install torch torchvision torchaudio
```

O programa detectará automaticamente que não existe CUDA disponível e utilizará a CPU.

---

# 5. Instalar as dependências

Instale as bibliotecas utilizadas pelo projeto:

```powershell
python -m pip install diffusers transformers accelerate peft safetensors pillow huggingface_hub rembg opencv-python==4.13.0.92 onnxruntime
```

### Principais bibliotecas

| Biblioteca        | Utilização                                           |
| ----------------- | ---------------------------------------------------- |
| `torch`           | Execução dos modelos de IA                           |
| `torchvision`     | Componentes de visão computacional do PyTorch        |
| `diffusers`       | Stable Diffusion e pipelines de geração              |
| `transformers`    | Modelos e componentes de IA                          |
| `accelerate`      | Gerenciamento da execução dos modelos                |
| `peft`            | LoRAs e adaptações dos modelos                       |
| `safetensors`     | Carregamento dos pesos dos modelos                   |
| `Pillow`          | Leitura e processamento de imagens                   |
| `opencv-python`   | Processamento de imagens e detecção de enquadramento |
| `rembg`           | Remoção automática do fundo                          |
| `onnxruntime`     | Execução do modelo utilizado pelo `rembg`            |
| `huggingface_hub` | Download dos modelos hospedados no Hugging Face      |

> **Não instale `opencv-python-headless` junto com `opencv-python`.**
>
> O projeto utiliza funcionalidades do OpenCV como `CascadeClassifier`, portanto deve ser utilizada a versão `opencv-python`.

---

# 6. Verificar as dependências

Execute:

```powershell
python -c "import torch, diffusers, transformers, accelerate, peft, cv2, PIL, rembg; print('Todas as dependências foram carregadas com sucesso!')"
```

O resultado esperado é:

```text
Todas as dependências foram carregadas com sucesso!
```

Também é possível verificar especificamente o OpenCV:

```powershell
python -c "import cv2; print('OpenCV:', cv2.__version__); print('CascadeClassifier:', hasattr(cv2, 'CascadeClassifier'))"
```

O resultado esperado deve conter:

```text
CascadeClassifier: True
```

---

# 7. Executar o Imago

Com o ambiente virtual ativado:

```powershell
python main.py
```

O programa exibirá o menu de efeitos:

```text
========================================
           🎨 IMAGO
========================================

1 - 🕹️  Pixel Art
2 - 🎨 Cartoon
3 - 📷 Realista
0 - ❌ Sair
========================================
```

Depois de escolher o efeito, o programa detectará automaticamente o dispositivo disponível.

Exemplo com GPU:

```text
🖥️ GPU detectada: NVIDIA GeForce RTX 3080 (10.0GB VRAM)
⚙️ Dispositivo de processamento: cuda
```

Em um computador sem CUDA:

```text
🖥️ Nenhuma GPU CUDA detectada -> CPU
⚙️ Dispositivo de processamento: cpu
```

---

# 8. Fundos temáticos

O Imago possui um sistema de fundos temáticos localizado em:

```text
img/
└── fundos/
```

Os fundos são cadastrados em:

```text
fundos.py
```

Cada fundo pode ser associado a:

* um ou mais efeitos;
* um ou mais enquadramentos.

Exemplo:

```python
FUNDOS = {
    "mine.png": {
        "estilos": ["pixel_art"],
        "enquadramentos": ["corpo_inteiro", "busto"],
    },

    "itsfine.png": {
        "estilos": ["cartoon"],
        "enquadramentos": ["corpo_inteiro", "busto"],
    },

    "disaster.png": {
        "estilos": ["realista"],
        "enquadramentos": ["rosto"],
    },
}
```

O sistema utiliza essas informações para selecionar fundos compatíveis com o efeito e com o enquadramento da fotografia.

---

# 9. Debug

O projeto possui um sistema de debug para permitir visualizar as etapas do processamento.

As imagens intermediárias são armazenadas na pasta configurada em:

```text
config.py
```

Exemplos de etapas:

```text
1_original.png
2_mascara.png
3_clahe.png
4_fundo.png
5_crop.png
6_saida_ia.png
7_final.png
```

Isso permite acompanhar o processamento da imagem e também pode ser utilizado durante a apresentação do projeto na FICE.

---

# 10. Estrutura do projeto

A estrutura principal é semelhante a:

```text
Imago/
│
├── main.py
├── config.py
├── utils.py
├── fundos.py
│
├── pixel.py
├── cartoon.py
├── realista.py
│
├── img/
│   ├── entrada/
│   ├── fundos/
│   └── ...
│
├── debug/
│
└── .venv/
```

### Principais arquivos

**`main.py`**

Controla o fluxo principal do programa, incluindo a escolha do efeito e do processamento.

**`pixel.py`**

Implementa o efeito de Pixel Art utilizando Stable Diffusion e pós-processamento para criar a aparência pixelada.

**`cartoon.py`**

Implementa o efeito Cartoon utilizando Stable Diffusion.

**`realista.py`**

Implementa o processamento realista, utilizando principalmente remoção de fundo e composição da fotografia.

**`utils.py`**

Contém funções compartilhadas de processamento de imagem, como:

* remoção de fundo;
* CLAHE;
* ajuste de contraste;
* detecção de enquadramento;
* composição;
* crop;
* geração de imagens de debug.

**`fundos.py`**

Contém o cadastro e o sistema de seleção dos fundos temáticos.

**`config.py`**

Contém configurações gerais do projeto, como caminhos de entrada/saída, debug e diretórios.

---

# 11. Modelos

Na primeira execução, alguns modelos podem ser baixados automaticamente.

O Stable Diffusion 1.5 é baixado do Hugging Face e pode ocupar vários gigabytes.

O `rembg` também realiza o download dos modelos necessários para a remoção do fundo.

> A primeira execução pode ser significativamente mais demorada devido ao download e carregamento dos modelos.

---

# 12. Solução de problemas

## `CUDA disponível: False`

Verifique:

1. Se a GPU NVIDIA está instalada.
2. Se os drivers NVIDIA estão atualizados.
3. Se o PyTorch instalado possui suporte a CUDA.

Teste:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

## `cv2 has no attribute CascadeClassifier`

Provavelmente existem versões conflitantes do OpenCV instaladas.

Remova ambas:

```powershell
python -m pip uninstall opencv-python opencv-python-headless -y
```

Instale somente:

```powershell
python -m pip install opencv-python
```

Teste:

```powershell
python -c "import cv2; print(hasattr(cv2, 'CascadeClassifier'))"
```

O resultado deve ser:

```text
True
```

---

## O processamento está muito lento

Os efeitos que utilizam Stable Diffusion exigem bastante processamento.

Em CPU, especialmente em computadores com hardware mais antigo, o processamento pode levar vários minutos.

Sempre que possível, utilize uma GPU NVIDIA compatível com CUDA.

---

# Comandos resumidos

## Primeira instalação

```powershell
git clone https://github.com/ViniciusM15IFC/Imago.git

cd Imago

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -m pip install diffusers transformers accelerate peft safetensors pillow huggingface_hub rembg opencv-python onnxruntime

python main.py
```

## Execuções seguintes

```powershell
cd Imago

.\.venv\Scripts\Activate.ps1

python main.py
```

---

# ⚠️ Importante

Sempre execute o projeto com o ambiente virtual ativado.

Verifique se o terminal começa com:

```text
(.venv) PS C:\...
```

Antes de executar:

```powershell
python main.py
```

```

**Uma correção importante em relação ao seu README antigo:** eu tirei `opencv-python-headless` deliberadamente. No seu caso, instalar os dois foi justamente o que acabou levando ao problema do `CascadeClassifier`. Também removi a afirmação de que o projeto necessariamente usa **RTX 3080 + SDXL + ControlNet**, porque isso já não representa o estado atual do projeto.
```
