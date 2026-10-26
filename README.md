# NPU-enabler

Get an AMD Ryzen AI NPU (Phoenix / Ryzen 8000 series) validated on Windows, then run Phi-4-Multimodal screen description through AMD's onnxruntime-genai stack. The long term goal is a local screen agent: capture the screen, describe it with a vision model, let a text LLM (Ollama) decide an action, and execute it with pyautogui.

## Honest status

| Piece | Status |
|---|---|
| NPU driver installed and passing `xrt-smi validate` | Works |
| Phi-4-Multimodal vision inference | Works on CPU |
| Phi-4-Multimodal on the NPU | Not possible on Phoenix/Hawk Point: AMD supports only CNN INT8 there. See "Why the NPU is not running the model" |
| Phi-4-Multimodal on GPU via DirectML | Works in a separate conda env (`phi4-dml`, see Step 7): about 24 tokens/s decode and about 12 s to first token at 448x448 on an RTX 4060 laptop GPU |
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
- About 15 GB usable RAM (Task Manager shows 15.2 GB total). Phi-4-Multimodal needs roughly 8 GB at peak, so close heavy apps before running it.
- NPU 0 visible in Task Manager with driver 32.0.203.280

## Why the NPU is not running the model

AMD's Ryzen AI documentation lists NPU only and Hybrid LLM execution as supported on Ryzen AI 300 series. Ryzen AI 7000 and 8000 series are listed as GPU/CPU only for LLMs. So on Phoenix the NPU driver and runtime work, but AMD does not ship NPU compiled LLM or VLM models for it.

The Ryzen AI 1.8 release notes (https://ryzenai.docs.amd.com/en/latest/relnotes.html) say it directly: "PHX/HPT supports only CNN INT8. CNN BF16, NLP BF16, and LLM (OGA) are not supported on PHX/HPT." So Phi-4-Multimodal cannot run on a Phoenix or Hawk Point NPU with this software, no matter how the model is exported. What the Phoenix NPU can run is small INT8 convolutional networks (image classifiers, detectors, OCR style models). Also from the release notes: PHX/HPT models must pass the `xclbin` provider option and `target` set to `X1`, and `4x4.xclbin` replaced the old `1x4.xclbin`.

### quicktest result on Phoenix (2026-10-26)

AMD's `quicktest/quicktest.py` has a `PHX/HPT` branch that points at `voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin`. On this machine that folder does not exist: `voe-4.0-win_amd64` contains only `vaip_config.json`, and no `.xclbin` exists anywhere under `C:\Program Files\RyzenAI` or the conda env. The script still printed `Test Finished`, but with four `Failed to open xclbin` errors and without the "Actually running on NPU" line AMD's docs describe for a good run, so it did not run on the NPU. The current install docs also say quicktest "is expected to work for Strix (STX) or newer devices".

The driver does ship xclbins in `C:\Windows\System32\AMD`. Pointing `npu_quicktest.py` at `4x4_3.5.0.0-2352.xclbin` from there fails immediately with `Cannot find or create target with fingerprint=0x080002050018eec1`, meaning the VitisAI runtime does not recognize that overlay. The Phoenix `4x4.xclbin` AMD expects is not in the driver folder and was not installed.

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

In the RyzenAI env pass `--provider cpu`. In the `phi4-dml` env (Step 7) the default `auto` picks DirectML on the GPU. `auto` never picks the NPU, because Phi-4-Multimodal cannot run there (see "Why the NPU is not running the model"). Forcing `--provider vitisai` with the generic model initializes the provider and then exits silently with no description.

Each run prints `Stats: {...}` with token count, time to first token (including image processing) and decode tokens per second.

As a library:

```python
from vlm_backend import VLMBackend
vlm = VLMBackend("C:/Users/<you>/phi-4-mm-onnx/gpu/gpu-int4-rtn-block-32", provider="cpu")
print(vlm.describe(screenshot=True))
```

## Step 7: Run on the GPU with DirectML (recommended)

The RyzenAI env rejects generic DirectML, so use a separate env with the DirectML build of onnxruntime-genai:

```
conda create -n phi4-dml python=3.11 -y
conda activate phi4-dml
pip install onnxruntime-genai-directml pillow numpy pyautogui --no-input --keyring-provider disabled
```

Tested versions: onnxruntime-genai-directml 0.13.1 with onnxruntime-directml 1.24.4. Then:

```
python vlm_backend.py C:/Temp/test_screen.png --max_tokens 100
```

You should see `Device: DirectML`. Measured on an RTX 4060 Laptop GPU (448x448 input, 100 tokens): about 12.4 s to first token and about 24 tokens/s decode. The first load of the model takes a while. The same model on CPU gave an identical description for the same image. During loading the NVIDIA GPU showed about 4 GB of memory in use, so it is using the RTX card rather than the integrated Radeon.

## Step 8: OCR baseline and NPU wake test

OCR baseline (CPU), in its own env so it cannot disturb the DirectML one:

```
conda create -n ocr-test python=3.11 -y
conda activate ocr-test
pip install rapidocr-onnxruntime pillow pyautogui --no-input --keyring-provider disabled
python ocr_test.py C:/Temp/test_screen.png
```

Measured on the 8845HS CPU, 1918x1079 desktop screenshot: 142 text lines read in 3.8 s (detection 1.2 s, classifier 0.14 s, recognition 2.3 s), model load 0.7 s. RapidOCR ships three ONNX models: `ch_PP-OCRv4_det_infer.onnx` (4.7 MB), `ch_PP-OCRv4_rec_infer.onnx` (10.9 MB) and `ch_ppocr_mobile_v2.0_cls_infer.onnx` (0.6 MB). The detection model is a CNN and the only plausible NPU candidate after INT8 quantization. The recognition model probably is not (it likely uses attention layers); unverified.

Waking the NPU: `xrt-smi validate` is the one thing that reliably does it on Phoenix. Open Task Manager, Performance, NPU 0 and run it a few times in a row:

```
$XRT = "C:\Users\<you>\Downloads\NPU_RAI_280_WHQL\npu_mcdm_stack_prod\xrt-smi.exe"
1..5 | ForEach-Object { & $XRT validate -d 0007:00:01.1 }
```

Running a real model (including the OCR detector) on the NPU is blocked: none of the RyzenAI 1.8.0 packages contain a Phoenix xclbin (only a Strix `base.xclbin` appears inside `voe-1.8.0`), and quicktest's expected `xclbins\phoenix\4x4.xclbin` is absent. A release that still ships it (for example 1.5 or 1.6, unverified) is the next thing to try.

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
- [x] `auto` no longer selects the NPU; it picks DirectML when available, else CPU
- [x] DirectML GPU inference in a separate conda env (`phi4-dml`)
- [ ] OpenAI compatible /v1/chat/completions wrapper (FastAPI) so tools like OmniRoute can use the local vision model
- [ ] Screen capture loop
- [ ] Ollama action LLM
- [ ] pyautogui executor
- [ ] NPU for small INT8 CNN helpers (UI element detector, OCR detection) once the Phoenix `4x4.xclbin` is installed. The NPU cannot run Phi-4-Multimodal itself.
