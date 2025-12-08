import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Prepare your data manually in a CSV-like structure ===
# Each block will be read into a DataFrame and tagged by type.
data = {
    "Base (hopper)": [
        [500,1000,18.931681,4.990096,0.871143,0.401378],
        [1000,1000,24.162975,12.700329,3.005656,0.578726],
        [5000,1000,264.052826,260.98913,45.827717,20.250812],
        [10000,1000,1037.83252,1034.91548,153.332794,59.588815],
        [15000,1000,2327.901367,2324.63207,328.18458,143.469279],
        [20000,1000,4133.914551,4130.819899,592.742775,233.218791],
    ],
    "Base (ampere)": [
        [500,1000,16.223904,5.443014,0.808487,0.400674],
        [1000,1000,24.257759,14.016785,4.262463,1.674957],
        [5000,1000,389.349792,378.602952,41.121946,14.855943],
        [10000,1000,1740.924316,1738.829184,144.283148,60.80951],
        [15000,1000,4156.018066,4153.787009,311.106713,131.964138],
        [20000,1000,7799.206055,7796.902288,545.221374,227.516916],
    ],
    "Shared (ampere)": [
        [500,1000,16.830433,4.475018,0.121058,0.080096],
        [1000,1000,22.051329,10.734276,2.644245,0.557221],
        [5000,1000,289.233246,282.682176,36.222964,14.905725],
        [10000,1000,1120.085449,1113.743424,149.085941,58.037206],
        [15000,1000,2553.01001,2546.52141,333.38014,129.570084],
        [20000,1000,4598.858887,4592.46383,561.462891,226.312463],
    ],
    "Shared (hopper)": [
        [500,1000,19.09824,3.353396,0.111363,0.074528],
        [1000,1000,15.040352,7.585247,2.867319,0.530693],
        [5000,1000,173.8535,167.847481,41.196222,18.057074],
        [10000,1000,707.882568,701.602398,223.958874,56.864679],
        [15000,1000,1637.044189,1630.563973,376.891786,129.533287],
        [20000,1000,2922.654785,2916.052595,602.483646,230.59259],
    ],
}

cols = ["size","iterations","baseline_ms","kernel_total_ns","device_to_host_ns","host_to_device_ns"]

dfs = []
for label, rows in data.items():
    df = pd.DataFrame(rows, columns=cols)
    df["config"] = label
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# Convert kernel_total_ns etc. from milliseconds scale if needed (already ms in your table)
# If they’re actually ns, multiply/divide accordingly — assuming these are ms-scale values.
df_all["kernel_ms"] = df_all["kernel_total_ns"]
df_all["device_to_host_ms"] = df_all["device_to_host_ns"]
df_all["host_to_device_ms"] = df_all["host_to_device_ns"]

# === Step 2: Plot baseline GPU time vs size ===
plt.figure(figsize=(8,5))
for label, group in df_all.groupby("config"):
    plt.plot(group["size"], group["baseline_ms"], marker="o", label=label)
plt.xlabel("Matrix Size (N)")
plt.ylabel("Baseline Time (ms)")
plt.title("Baseline GPU Time vs Size (iterations=1000)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("baseline_time_comparison.png", dpi=300)

# === Step 3: Plot kernel execution time vs size ===
plt.figure(figsize=(8,5))
for label, group in df_all.groupby("config"):
    plt.plot(group["size"], group["kernel_ms"], marker="o", label=label)
plt.xlabel("Matrix Size (N)")
plt.ylabel("Kernel Time (ms)")
plt.title("Kernel Execution Time vs Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("kernel_time_comparison.png", dpi=300)

# === Step 4: Plot data transfer times ===
plt.figure(figsize=(8,5))
for label, group in df_all.groupby("config"):
    plt.plot(group["size"], group["device_to_host_ms"], marker="o", label=f"{label} D→H")
    plt.plot(group["size"], group["host_to_device_ms"], marker="s", label=f"{label} H→D")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Transfer Time (ms)")
plt.title("CUDA Memcpy Transfer Times vs Size")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("transfer_time_comparison.png", dpi=300)

# === Step 5: Compute speedup ratios (Ampere vs Hopper) for Base and Shared ===
def compute_speedup(label_ampere, label_hopper):
    df_amp = df_all[df_all["config"] == label_ampere].set_index("size")
    df_hop = df_all[df_all["config"] == label_hopper].set_index("size")
    common_sizes = df_amp.index.intersection(df_hop.index)
    df_speed = pd.DataFrame(index=common_sizes)
    df_speed["kernel_speedup"] = df_hop.loc[common_sizes, "kernel_ms"] / df_amp.loc[common_sizes, "kernel_ms"]
    df_speed["baseline_speedup"] = df_hop.loc[common_sizes, "baseline_ms"] / df_amp.loc[common_sizes, "baseline_ms"]
    return df_speed

speed_base = compute_speedup("Base (ampere)", "Base (hopper)")
speed_shared = compute_speedup("Shared (ampere)", "Shared (hopper)")

# === Step 6: Plot speedup comparison ===
plt.figure(figsize=(8,5))
plt.plot(speed_base.index, speed_base["kernel_speedup"], "o-", label="Base Hopper/Ampere Kernel")
plt.plot(speed_shared.index, speed_shared["kernel_speedup"], "s--", label="Shared Hopper/Ampere Kernel")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Speedup (Hopper / Ampere)")
plt.title("Kernel Execution Speedup: Hopper vs Ampere")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("speedup_kernel_comparison.png", dpi=300)

plt.show()
print("✅ Saved plots: baseline_time_comparison.png, kernel_time_comparison.png, transfer_time_comparison.png, speedup_kernel_comparison.png")
