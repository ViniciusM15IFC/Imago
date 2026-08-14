# Script Python para edição automática de fotos

Este projeto utiliza **PyTorch, Diffusers, Transformers e ControlNet** para realizar edições automáticas de imagens.

## 1. Ativar o ambiente virtual

Abra o PowerShell na pasta do projeto:

```powershell
cd C:\Users\Usuario\Downloads\Imago-main
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se aparecer `(.venv)` no início da linha do terminal, o ambiente está ativado.

---

## 2. Instalar o PyTorch com suporte à GPU

Para utilizar a **NVIDIA GeForce RTX 3080**, instale uma versão do PyTorch com CUDA:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> Não use a versão `+cpu`, pois ela não consegue utilizar a placa de vídeo.

### Testar a GPU

Execute:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NENHUMA')"
```

O resultado esperado é semelhante a:

```text
PyTorch: 2.11.0+cu128
CUDA: 12.8
CUDA disponível: True
GPU: NVIDIA GeForce RTX 3080
```

---

## 3. Instalar as dependências do projeto

Instale as bibliotecas necessárias:

```powershell
python -m pip install diffusers transformers accelerate peft safetensors pillow huggingface_hub rembg onnxruntime torchvision opencv-python-headless
```

Essas bibliotecas são utilizadas para:

* `diffusers` — Stable Diffusion e ControlNet
* `transformers` — modelos e componentes de linguagem/visão
* `accelerate` — gerenciamento de execução na GPU
* `peft` — componentes de adaptação de modelos
* `safetensors` — carregamento seguro dos pesos
* `opencv-python` — processamento de imagens
* `pillow` — leitura e gravação de imagens
* `huggingface_hub` — acesso aos modelos hospedados no Hugging Face

---

## 4. Verificar as dependências

Antes de executar o programa, teste:

```powershell
python -c "import torch, diffusers, transformers, accelerate, peft, cv2, PIL; print('Todas as dependências foram carregadas com sucesso!')"
```

Se aparecer:

```text
Todas as dependências foram carregadas com sucesso!
```

o ambiente está pronto.

---

## 5. Executar o programa

Com o ambiente virtual ativado:

```powershell
python main.py
```

O programa deverá detectar automaticamente a GPU e escolher a rota adequada.

Na RTX 3080, a saída deverá ser semelhante a:

```text
🖥️ GPU detectada: NVIDIA GeForce RTX 3080 (10.0GB VRAM)
🚀 VRAM suficiente -> rota pesada (GPU, SDXL + ControlNet)
```

Ao terminar:

```text
✨ Concluído em X.XX minutos!
📂 Verifique o resultado em: '...'
```

---

## Comandos resumidos

Se o ambiente virtual já estiver criado, o procedimento completo é:

```powershell
.\.venv\Scripts\Activate.ps1

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -m pip install diffusers transformers accelerate peft safetensors opencv-python pillow huggingface_hub

python main.py
```

### Importante

O projeto deve ser executado **com o `.venv` ativado**. Sempre verifique se o terminal começa com:

```text
(.venv) PS C:\...
```

Antes de executar `python main.py`.
