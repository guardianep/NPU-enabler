# screen-agent

A hardware-aware screen agent that captures the screen, describes it using a vision model (Phi-4-Multimodal), and routes decisions to a text LLM for action execution.

## Hardware target

- AMD Ryzen 9 8845HS (Phoenix, Ryzen AI 8000 series)
- NVIDIA RTX 4060 Laptop GPU
- AMD NPU Phoenix (AIE2, 16 TOPS) -- driver validated, awaiting AMD NPU model support
- 32GB RAM

## Stack

- Vision: Microsoft Phi-4-Multimodal-Instruct (ONNX, int4)
- Runtime: onnxruntime-genai via AMD RyzenAI 1.8.0
- Execution: CPU (current) / DirectML GPU (next) / NPU VitisAI (when AMD releases NPU model)
- Action LLM: Ollama (local)
- Screen capture: pyautogui

## Setup

```
conda activate ryzen-ai-1.8.0
pip install pillow pyautogui
```

Model path (not included in repo -- too large):
```
C:/Users/ehps1/phi-4-mm-onnx/gpu/gpu-int4-rtn-block-32
```

## Usage

```bash
# Describe an image file
python vlm_backend.py C:/Temp/screen.png --provider cpu

# Take a live screenshot and describe it
python vlm_backend.py --screenshot --provider cpu

# Use GPU via DirectML (requires phi4-dml conda env)
python vlm_backend.py --screenshot --provider dml

# Force NPU (requires AMD NPU-compiled model variant)
python vlm_backend.py --screenshot --provider vitisai
```

## Provider auto-detection order

1. NPU (VitisAI) -- if `onnxruntime_vitisai_ep.dll` found in RyzenAI deployment
2. DirectML (GPU) -- if `onnxruntime-genai-directml` installed
3. CPU -- always available fallback

## Roadmap

- [x] vlm_backend.py with provider auto-detection and CPU fallback
- [ ] FastAPI wrapper exposing OpenAI-compatible /v1/chat/completions
- [ ] Screen capture loop feeding frames to VLM
- [ ] Ollama integration for action decision LLM
- [ ] pyautogui action executor (click, type, scroll)
- [ ] Full screen agent pipeline
- [ ] DirectML GPU inference (phi4-dml env)
- [ ] NPU inference (pending AMD Phi-4-MM NPU model release)
