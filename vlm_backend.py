# vlm_backend.py -- Hardware-aware VLM loader for Phi-4-Multimodal.
#
# Auto-detects best execution provider:
#   1. DirectML (GPU) if onnxruntime-genai-directml installed (conda env phi4-dml)
#   2. CPU fallback
# NPU (VitisAI) is never auto selected, see README. Force it with --provider vitisai.
#
# CLI usage:
#   python vlm_backend.py C:/Temp/screen.png
#   python vlm_backend.py C:/Temp/screen.png --provider cpu
#   python vlm_backend.py C:/Temp/screen.png --provider dml
#   python vlm_backend.py --screenshot

import os
import sys
import time
import argparse
import tempfile
import importlib.util
from pathlib import Path

try:
    import onnxruntime_genai as og
except ImportError:
    raise ImportError("onnxruntime_genai not found. Run: conda activate ryzen-ai-1.8.0")

from PIL import Image


# ── Provider detection ─────────────────────────────────────────────────────────

VITISAI_DLL = Path("C:/Program Files/RyzenAI/1.8.0/deployment/onnxruntime_vitisai_ep.dll")


def _dml_available():
    # True when the DirectML build of onnxruntime-genai is installed in this env.
    try:
        from importlib import metadata
        metadata.version("onnxruntime-genai-directml")
        return True
    except Exception:
        return False


def detect_provider():
    # NPU is never auto selected: on Phoenix/Hawk Point AMD supports only CNN INT8,
    # so Phi-4-Multimodal cannot run there. Use --provider vitisai to force it.
    if _dml_available():
        print("[vlm_backend] DirectML found -- using GPU")
        return "dml"
    print("[vlm_backend] Using CPU")
    return "cpu"


# ── Image helpers ──────────────────────────────────────────────────────────────

def prepare_image(image_path, size=(448, 448)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tmp.name, format="JPEG", quality=95)
    tmp.close()
    return tmp.name


def screenshot_to_temp():
    try:
        import pyautogui
    except ImportError:
        raise ImportError("pip install pyautogui")
    img = pyautogui.screenshot()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(tmp.name)
    tmp.close()
    return tmp.name


# ── Core class ─────────────────────────────────────────────────────────────────

class VLMBackend:
    def __init__(self, model_path, provider="auto", image_size=(448, 448), max_tokens=256):
        self.model_path = model_path
        self.image_size = image_size
        self.max_tokens = max_tokens

        if provider == "auto":
            provider = detect_provider()

        # Try requested provider, fall back to CPU on failure
        self.provider = provider
        self.model = self._load_model(provider)

    def _load_model(self, provider):
        print(f"[vlm_backend] Loading: {self.model_path}")
        print(f"[vlm_backend] Provider: {provider}")
        try:
            config = og.Config(self.model_path)
            config.clear_providers()
            if provider != "cpu":
                config.append_provider(provider)
            model = og.Model(config)
            print(f"[vlm_backend] Model type: {model.type}")
            print(f"[vlm_backend] Device: {model.device_type}")
            self.processor = model.create_multimodal_processor()
            self.tokenizer = og.Tokenizer(model)
            return model
        except Exception as e:
            if provider != "cpu":
                print(f"[vlm_backend] {provider} failed ({e}), falling back to CPU")
                self.provider = "cpu"
                return self._load_model("cpu")
            raise

    def _build_prompt(self, text):
        return f"<|user|>\n<|image_1|>\n{text}<|end|>\n<|assistant|>\n"

    def describe(self, image_path=None, prompt="Describe what you see on this screen.", screenshot=False, resize=True):
        if screenshot:
            raw_path = screenshot_to_temp()
        elif image_path:
            raw_path = image_path
        else:
            raise ValueError("Provide image_path or set screenshot=True")

        proc_path = prepare_image(raw_path, self.image_size) if resize else raw_path

        t_start = time.time()
        images = og.Images.open(proc_path)
        inputs = self.processor(self._build_prompt(prompt), images=images)

        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=16384)

        generator = og.Generator(self.model, params)
        generator.set_inputs(inputs)

        stream = self.processor.create_stream()
        chunks = []
        count = 0
        t_first = None

        while not generator.is_done() and count < self.max_tokens:
            generator.generate_next_token()
            if t_first is None:
                t_first = time.time()
            token = generator.get_next_tokens()[0]
            chunks.append(stream.decode(token))
            count += 1

        t_end = time.time()
        ttft = (t_first - t_start) if t_first else 0.0
        decode_time = (t_end - t_first) if t_first else 0.0
        self.last_stats = {
            "tokens": count,
            "ttft_s": round(ttft, 2),
            "decode_tps": round((count - 1) / decode_time, 2) if decode_time > 0 and count > 1 else 0.0,
        }

        del generator

        if resize and proc_path != raw_path:
            try:
                os.unlink(proc_path)
            except Exception:
                pass
        if screenshot:
            try:
                os.unlink(raw_path)
            except Exception:
                pass

        return "".join(chunks)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM screen description")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--screenshot", action="store_true", help="Take a live screenshot")
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "dml", "vitisai"])
    parser.add_argument("--prompt", default="Describe what you see on this screen in detail.")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--size", type=int, nargs=2, default=[448, 448], metavar=("W", "H"))
    args = parser.parse_args()

    MODEL = "C:/Users/ehps1/phi-4-mm-onnx/gpu/gpu-int4-rtn-block-32"

    vlm = VLMBackend(
        model_path=MODEL,
        provider=args.provider,
        image_size=tuple(args.size),
        max_tokens=args.max_tokens,
    )

    print("\nRunning inference...")
    result = vlm.describe(
        image_path=args.image,
        prompt=args.prompt,
        screenshot=args.screenshot,
    )
    print(f"\nResult:\n{result}")
    print(f"\nStats: {vlm.last_stats}")
