# npu_quicktest.py: AMD's quicktest, but with a configurable xclbin and a loop
# so the NPU load is visible in Task Manager. AMD's original is not modified.
#
# Run inside the ryzen-ai-1.8.0 conda env:
#   $env:RYZEN_AI_INSTALLATION_PATH = "C:\Program Files\RyzenAI\1.8.0"
#   python npu_quicktest.py --xclbin "C:\Windows\System32\AMD\4x4_3.5.0.0-2352.xclbin" --iters 3000

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--xclbin", required=True, help="Path to an .xclbin file")
parser.add_argument("--iters", type=int, default=3000, help="Inference loops (more = longer NPU load)")
args = parser.parse_args()

if not os.path.exists(args.xclbin):
    sys.exit(f"xclbin not found: {args.xclbin}")

install_dir = os.environ["RYZEN_AI_INSTALLATION_PATH"]
model = os.path.join(install_dir, "quicktest", "test_model.onnx")

provider_options = [{
    "target": "X1",
    "xlnx_enable_py3_round": 0,
    "xclbin": args.xclbin,
}]

so = ort.SessionOptions()
so.log_severity_level = 1

print(f"xclbin: {args.xclbin}")
print("Creating session (first compile takes about 30 seconds)...")
try:
    session = ort.InferenceSession(
        model,
        sess_options=so,
        providers=["VitisAIExecutionProvider"],
        provider_options=provider_options,
    )
except Exception as e:
    sys.exit(f"Failed to create session: {e}")

print("Providers:", session.get_providers())

x = np.expand_dims(np.random.rand(3, 32, 32).astype(np.float32), axis=0)
print(f"Running {args.iters} inferences. Watch NPU 0 in Task Manager now.")
t0 = time.time()
try:
    for _ in range(args.iters):
        session.run(None, {"input": x})
except Exception as e:
    sys.exit(f"Run failed: {e}")
dt = time.time() - t0
print(f"Done: {args.iters} runs in {dt:.2f}s ({args.iters / dt:.0f} runs/s)")
