# ocr_test.py: baseline OCR on a screenshot using RapidOCR (ONNX models, CPU).
# Gives us the text, timing, and the ONNX model files that could later be
# quantized to INT8 and tried on the NPU (CNN INT8 is what Phoenix supports).
#
# Run in the ocr-test conda env:
#   python ocr_test.py C:/Temp/test_screen.png
#   python ocr_test.py --screenshot

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", help="Path to image file")
parser.add_argument("--screenshot", action="store_true", help="Capture the screen first")
args = parser.parse_args()

if args.screenshot:
    import pyautogui
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pyautogui.screenshot().save(tmp.name)
    tmp.close()
    image = tmp.name
elif args.image:
    image = args.image
else:
    sys.exit("Give an image path or --screenshot")

from rapidocr_onnxruntime import RapidOCR

t0 = time.time()
engine = RapidOCR()
t_load = time.time() - t0

t1 = time.time()
result, elapse = engine(image)
t_run = time.time() - t1

print(f"Model load: {t_load:.2f}s   OCR run: {t_run:.2f}s   (det/cls/rec seconds: {elapse})")
print(f"Lines found: {0 if not result else len(result)}")
for box, text, conf in (result or [])[:40]:
    print(f"  [{float(conf):.2f}] {text}")

models = Path(sys.modules["rapidocr_onnxruntime"].__file__).parent / "models"
print("\nONNX model files (candidates for INT8 quantization):")
for f in sorted(models.glob("*.onnx")):
    print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")

if args.screenshot:
    os.unlink(image)
