import re
import pandas as pd

LOG_FILE = "shared_hopper.o31031"
ARCH = "Ampere"

# --- Regex patterns tuned for Nsight Systems output ---
re_run = re.compile(r"Running\s+(\d+)x(\d+)\s+grid,\s+(\d+)\s+iterations")
re_baseline = re.compile(r"shared\s+GPU\s+time\s+is\s+taken\s*=\s*([\d.]+)\s*ms")

# Kernel total line (ends with compute_gpu(...))
re_kernel_total = re.compile(
    r"100\.0\s+([\d.]+)\s+[0-9]+\s+[0-9.]+\s+[0-9.]+\s+[0-9]+\s+[0-9]+\s+[0-9.]+\s+compute_shared_gpuV1",
    re.MULTILINE,
)

# Memcpy totals (allow ints/decimals/scientific)
re_device_to_host = re.compile(
    r"\s*\d+\.\d+\s+([\d.eE+-]+)\s+\d+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+\[CUDA memcpy Device-to-Host\]",
    re.MULTILINE
)
re_host_to_device = re.compile(
    r"\s*\d+\.\d+\s+([\d.eE+-]+)\s+\d+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+\[CUDA memcpy Host-to-Device\]",
    re.MULTILINE
)

# --- Parse file ---
text = open(LOG_FILE).read()
records = []

for m in re.finditer(re_run, text):
    start = m.start()
    next_m = re_run.search(text, m.end())
    end = next_m.start() if next_m else len(text)
    section = text[start:end]

    N = int(m.group(1))
    iterations = int(m.group(3))

    baseline_ms = float(re_baseline.search(section).group(1)) if re_baseline.search(section) else None
    kernel_val = re_kernel_total.search(section)
    kernel_ns = float(kernel_val.group(1))/1000000 if kernel_val else None

    dev_to_host_match = re_device_to_host.search(section)
    host_to_dev_match = re_host_to_device.search(section)
    dev_to_host_ns = float(dev_to_host_match.group(1))/1000000 if dev_to_host_match else None
    host_to_dev_ns = float(host_to_dev_match.group(1))/1000000 if host_to_dev_match else None

    records.append({
        "arch": ARCH,
        "size": N,
        "iterations": iterations,
        "baseline_ms": baseline_ms,
        "kernel_total_ns": kernel_ns,
        "device_to_host_ns": dev_to_host_ns,
        "host_to_device_ns": host_to_dev_ns,
    })

# --- Build and sort DataFrame ---
df = pd.DataFrame(records)
df.sort_values(by=["iterations", "size"], inplace=True, ignore_index=True)

# --- Save results ---
csv_name = f"gol_perf_{ARCH}_parsed.csv"
df.to_csv(csv_name, index=False)

print(f"✅ Saved parsed data to {csv_name}")
print(df)
