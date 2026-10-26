# NPU-enabler

Get an AMD Ryzen AI NPU (Phoenix / Ryzen 8000 series) validated on Windows, then run Phi-4-Multimodal screen description through AMD's onnxruntime-genai stack. The long term goal is a local screen agent: capture the screen, describe it with a vision model, let a text LLM (Ollama) decide an action, and execute it with pyautogui.

## Honest status

| Piece | Status |
|---|---|
| NPU driver installed and passing `xrt-smi validate` | Works |
| Phi-4-Multimodal vision inference | Works on CPU |
| Phi-4-Multimodal on the NPU | Not working. See "Why the NPU is not running the model" |
| Phi-4-Multimodal on GPU via DirectML | Not working in the RyzenAI conda env. Needs a separate env (untested) |
| Screen agent loop, Ollama, pyautogui actions | Not built yet |

If you only want to confirm your NPU is alive, do Steps 1 and 2 and stop there.

## Official AMD docs (read these for installing the software)

- Windows ML on Ryzen AI: https://ryzenai.docs.amd.com/projects/WinML/en/latest/

This is AMD's documentation for running your own ONNX models on the NPU through Microsoft's Windows ML runtime. Windows downloads and registers the VitisAI execution provider on demand, and you pick CPU, GPU or NPU by policy. Its sections include Installation (prerequisites and setup), VitisAI EP Model Support, Model Deployment, Model Conversion and Quantization, Execution Providers, Troubleshooting and FAQ. Key pages:

- Installation: https://ryzenai.docs.amd.com/projects/WinML/en/latest/installation.html
- VitisAI EP Model Support: https://ryzenai.docs.amd.com/projects/WinML/en/latest/model_support.html
- Release notes (supported processors): https://ryzenai.docs.amd.com/projects/WinML/en/stable/relnotes.html
- Troubleshooting: https://ryzenai.docs.amd.com/projects/WinML/en/latest/troubleshooting.html
- FAQ: https://ryzenai.docs.amd.com/projects/WinML/en/latest/faq.html
- Examples: https://github.com/amd/RyzenAI-SW/tree/main/WinML

What those pages say (checked 2026-10-26):

- Processors: the 1.8 release notes say Ryzen AI 1.8 Software "supports AMD processors codenamed Phoenix, Hawk Point, Strix, Strix Halo, and Krackan Point", covering Ryzen 7000, 8000 and AI 300 series. So Phoenix and Hawk Point (Ryzen 8845HS class) are supported by the software in general.
- Driver: NPU driver 32.0.203.280 or newer. The `NPU_RAI_280_WHQL` package used above matches that number.
- OS: Windows 11 24H2 (build 26100) or later.
- Other prerequisites: Windows App SDK (Windows ML ships inside it), Python 3.10 to 3.12 for Python samples, Visual Studio 2022 with the C++ desktop workload for C++ samples. Conda is only used to create a test env (`winml_env`, Python 3.11) in the verification step.
- Models: CNN and Transformer models (float with automatic BF16 conversion, or QDQ quantized) compile for the NPU through the VitisAI EP. LLMs are via Foundry Local with pre-compiled models, or custom models through an Olive recipe or the Ryzen AI OGA workflow.
- Not stated anywhere I could find: whether Phoenix or Hawk Point can run LLMs or VLMs on the NPU. The separate LLM page still lists NPU only and Hybrid LLM modes for Ryzen AI 300 only. Treat Phi-4-Multimodal on a Phoenix NPU as unconfirmed.

### Windows ML ResNet test on Phoenix (2026-10-26)

Steps that worked: install the Windows App SDK, clone `amd/RyzenAI-SW`, `conda create -n winml_resnet python==3.12`, then in `WinML/CNN/ResNet` run `pip install -r requirements.txt --no-input --keyring-provider disabled`. Notes from doing it:

- The requirements file uses the public ORT-Nightly feed, which needs no login. If pip prompts for credentials, do not type any: a saved wrong login makes every request return 401. Use the flags above.
- Pin `wasdk-Microsoft.Windows.AI.MachineLearning==2.3.0` and `wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap==2.3.0` so they match the documented runtime. The 2.3.2.dev10 packages pip picks by default demand a newer Windows App Runtime and fail with "Package dependency criteria could not be resolved".
- If the Windows App Runtime is missing: `curl -L -o windowsappruntimeinstall-x64.exe "https://aka.ms/windowsappsdk/2.3/2.3.1/windowsappruntimeinstall-x64.exe"` then `.\windowsappruntimeinstall-x64.exe --quiet`.

Result: `run_model.py --ep_policy NPU` classified the dog image correctly, but it ran on `CPUExecutionProvider`, not the NPU. `winml_probe.py` in this repo shows why: Windows ML offered only `MIGraphXExecutionProvider`, `NvTensorRTRTXExecutionProvider` and `WebGpuExecutionProvider`. There was no `VitisAIExecutionProvider`, and onnxruntime listed no NPU device. Onnxruntime did list two `DmlExecutionProvider` GPU devices. So on this Phoenix machine the Windows ML route does not currently expose the NPU even though `xrt-smi validate` passes. The cause is not confirmed.

AMD also has a separate Ryzen AI flow built on onnxruntime-genai with an NPU only path. That is the stack this repo uses today (RyzenAI 1.8.0 plus `vlm_run.py`).

AMD's example code for Windows ML lives in the RyzenAI-SW GitHub repository.

## Tested hardware

- AMD Ryzen 9 8845HS (Phoenix, NPU shows up as "NPU Phoenix")
- NVIDIA RTX 4060 Laptop GPU
- Windows 11 (10.0.26200)
- 32 GB RAM

## Why the NPU is not running the model

AMD's Ryzen AI documentation lists NPU only and Hybrid LLM execution as supported on Ryzen AI 300 series. Ryzen AI 7000 and 8000 series are listed as GPU/CPU only for LLMs. So on Phoenix the NPU driver and runtime work, but AMD does not ship NPU compiled LLM or VLM models for it.

The Phi-4-Multimodal ONNX files from Microsoft (`gpu/gpu-int4-rtn-block-32`) are generic builds. Their `genai_config.json` has an empty `provider_options`, so they load on CPU. Pointing them at the VitisAI execution provider initializes the provider, then the process exits without output because the model was never compiled for the NPU.

## Step 0: Install conda first (do this before any AMD installer)

AMD's Ryzen AI installer and driver packages expect `conda` to be reachable from the command line. If it is not on PATH, the AMD installs fail with errors. Set conda up first.

1. Download Miniconda or Miniforge (Anaconda also works) for Windows.
2. Run the installer. When it asks about PATH, TICK the option to add conda to your PATH environment variable. The Anaconda and Miniconda installers label it "not recommended", but you need it here.
3. Close every terminal, open a new one, and verify:

```
conda --version
```

If that prints a version, you are good. If it says conda is not recognized, you skipped the PATH option. Either reinstall with it ticked, or add these to your user PATH manually (adjust for your install folder and type) and reopen the terminal:

```
C:\Users\<you>\miniconda3
C:\Users\<you>\miniconda3\Scripts
C:\Users\<you>\miniconda3\condabin
```

## Step 1: Install the NPU driver

1. Download the AMD NPU driver package (the one named like `NPU_RAI_..._WHQL`) from AMD's Ryzen AI software download page and extract it.
2. Run the installer inside it, or install the driver from Device Manager.
3. Open Device Manager and confirm an NPU device is present under Neural processors or System devices.

## Step 2: NPU smoke test

`xrt-smi` is NOT on PATH after install. Call it by its full path from wherever you extracted the driver package. Example:

```powershell
$XRT = "C:\Users\<you>\Downloads\NPU_RAI_280_WHQL\npu_mcdm_stack_prod\xrt-smi.exe"

# 1. Find your device id (look for the NPU line, format like 0007:00:01.1)
& $XRT examine

# 2. Run the smoke test against that id
& $XRT validate -d 0007:00:01.1
```

Expected output:

```
Platform              : NPU Phoenix
Power Mode            : Performance
Test 1 : latency      Average latency: 83.6 us      [PASSED]
Test 2 : throughput   Average throughput: 33670.5 ops  [PASSED]
```

What it does: `validate` runs two small synthetic workloads on the NPU, a latency test (dispatch one job, wait for the result) and a throughput test (many jobs back to back). It does not run a neural network. A pass means the driver, firmware and hardware path work. Your numbers will differ.

Add `--verbose` to see more detail. Use `examine --report all` for full device info.

## Step 3: Install Ryzen AI Software 1.8.0

1. Install AMD Ryzen AI Software 1.8.0. Default location is `C:\Program Files\RyzenAI\1.8.0`.
2. It creates a conda env named `ryzen-ai-1.8.0` (Python 3.12). Activate it:

```
conda activate ryzen-ai-1.8.0
```

3. Install extras used by this repo:

```
pip install pillow pyautogui
```

This env ships a custom onnxruntime-genai build that targets NPU and CPU. It rejects generic DirectML.

## Step 4: Download the model

```
pip install huggingface_hub
huggingface-cli download microsoft/Phi-4-multimodal-instruct-onnx --include "gpu/gpu-int4-rtn-block-32/*" --local-dir C:\Users\<you>\phi-4-mm-onnx
```

About 8 GB, 29 files. Model weights are not stored in this repo.

## Step 5: Run AMD's reference script on CPU

AMD ships `vlm_run.py`:

```
python "C:\Program Files\RyzenAI\1.8.0\LLM\example\vlm\vlm_run.py" ^
  -m "C:\Users\<you>\phi-4-mm-onnx\gpu\gpu-int4-rtn-block-32" ^
  -i C:\Temp\test_screen.png ^
  -p "Describe what you see on this screen in detail." ^
  --image_size 448 448 --max_tokens 150 -e cpu
```

Notes:

- Keep the image in a plain ASCII path like `C:\Temp`. A folder with accented characters (for example OneDrive `Imágenes`) gets mangled and gives `FileNotFoundError: Image not found`.
- Smaller images are much faster. AMD's own benchmark shows roughly 539 image tokens at 448x448 and 2619 at 1024x1024, with first token latency growing from seconds to tens of seconds.
- `-e` accepts `cpu`, `dml`, `follow_config` (default).

How provider selection works in that script:

```python
config = og.Config(model_path)       # reads genai_config.json
config.clear_providers()             # drop whatever the config specified
config.append_provider("dml")        # add a provider (skipped for cpu)
model = og.Model(config)
```

## Step 6: Use vlm_backend.py from this repo

`vlm_backend.py` wraps the same calls in a small class with provider fallback and an optional live screenshot.

```
# describe an image
python vlm_backend.py C:/Temp/test_screen.png --provider cpu

# capture the screen and describe it
python vlm_backend.py --screenshot --provider cpu
```

Always pass `--provider cpu` for now. The default (`auto`) picks VitisAI whenever `onnxruntime_vitisai_ep.dll` exists, and with the generic model that run initializes and then exits silently with no description.

As a library:

```python
from vlm_backend import VLMBackend
vlm = VLMBackend("C:/Users/<you>/phi-4-mm-onnx/gpu/gpu-int4-rtn-block-32", provider="cpu")
print(vlm.describe(screenshot=True))
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `xrt-smi is not recognized` | Not on PATH, and RyzenAI 1.8.0 does not bundle it | Call it by full path from the extracted NPU driver package |
| `FileNotFoundError: Image not found` with a `?` in the path | Accented characters in the folder name | Copy the image to `C:\Temp` |
| `SyntaxError: (unicode error) 'unicodeescape'` | Windows path like `C:\Users` inside a normal Python string or docstring | Use forward slashes or raw strings |
| `RuntimeError: This session cannot use graph capture ... DmlExecutionProvider` | RyzenAI conda env's onnxruntime-genai has no generic DirectML support | Use a separate env with `onnxruntime-genai-directml` |
| Run prints VitisAI log lines then returns to the prompt with no output | VitisAI EP selected but the model is not NPU compiled | Use `--provider cpu` |
| `gh auth login` fails with connection refused | Network blocks github.com | Set `$env:HTTPS_PROXY` and `$env:HTTP_PROXY` for that terminal session only |

## Roadmap

- [x] NPU driver validated with xrt-smi
- [x] Phi-4-Multimodal inference on CPU
- [x] Provider wrapper with CPU fallback
- [ ] Make `auto` default to CPU unless a VitisAI compiled model is detected
- [ ] DirectML GPU inference in a separate conda env
- [ ] Screen capture loop
- [ ] Ollama action LLM
- [ ] pyautogui executor
- [ ] NPU inference, pending an NPU compiled Phi-4-MM or another route for Phoenix
