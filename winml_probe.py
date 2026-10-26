# winml_probe.py: list which execution providers Windows ML offers on this PC.
# Read only. Does not download or register anything.
# Run inside the winml_resnet conda env:  python winml_probe.py

from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
    InitializeOptions,
    initialize,
)
import winui3.microsoft.windows.ai.machinelearning as winml

with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
    catalog = winml.ExecutionProviderCatalog.get_default()
    providers = catalog.find_all_providers()
    print(f"Providers offered by Windows ML: {len(providers)}")
    for p in providers:
        print(f"  name={p.name}  ready_state={p.ready_state}  library_path={p.library_path!r}")

import onnxruntime as ort
print("onnxruntime version:", ort.__version__)
print("available providers:", ort.get_available_providers())
try:
    for d in ort.get_ep_devices():
        print(f"  ep_device: {d.ep_name}  type={d.device.type}")
except Exception as e:
    print("get_ep_devices failed:", e)
