import matplotlib.pyplot as plt

# ==== x 轴预算 ====
budgets = [100, 80, 60, 40, 20]

# ==== 九组数据 ====
all_time1 = [
    [30.71, 27.43, 30.28, 35.62, 39.95],  # Llama2-7B xsum
    [30.03, 33.43, 36.76, 40.59, 48.42],  # Llama2-7B gov-report
    [29.54, 28.83, 36.11, 39.24, 42.80],  # Llama2-7B lcc
    [38.54, 33.01, 39.27, 42.76, 50.10],  # Qwen2.5-7B cnn
    [37.69, 37.65, 43.92, 48.41, 53.07],  # Qwen2.5-7B lcc
    [38.30, 38.31, 39.87, 40.65, 47.32],  # Qwen2.5-7B repobench-p
    [32.94, 28.46, 29.69, 34.01, 40.77],  # Tinyllama-1.1B xsum
    [33.10, 36.21, 39.82, 42.00, 49.16],  # Tinyllama-1.1B gov-report
    [36.87, 36.19, 37.74, 38.39, 45.07],  # Tinyllama-1.1B repobench-p
]

all_time2 = [
    [30.79, 26.19, 26.92, 27.56, 29.28],
    [30.03, 29.15, 29.99, 33.09, 36.67],
    [29.55, 28.21, 28.54, 29.09, 30.73],
    [38.55, 30.15, 32.94, 33.55, 36.94],
    [37.64, 33.74, 34.98, 36.93, 37.11],
    [38.33, 32.46, 31.99, 34.25, 35.40],
    [32.95, 25.57, 26.10, 28.00, 29.14],
    [33.11, 32.93, 33.79, 34.08, 37.56],
    [36.88, 32.78, 33.05, 34.66, 36.23],
]

all_time3 = [
    [30.79, 35.60, 38.45, 45.97, 57.17],
    [30.03, 37.71, 42.09, 47.88, 59.02],
    [29.55, 34.01, 38.95, 46.78, 57.48],
    [38.54, 42.28, 49.40, 56.75, 61.29],
    [37.64, 41.53, 47.70, 51.18, 60.59],
    [38.33, 42.80, 49.05, 56.22, 66.19],
    [32.95, 35.99, 39.86, 46.46, 54.25],
    [33.11, 36.87, 42.65, 45.81, 50.84],
    [36.88, 37.63, 41.29, 48.57, 61.01],
]

titles = [
    "Llama2-7B xsum",
    "Llama2-7B gov-report",
    "Llama2-7B lcc",
    "Qwen2.5-7B cnn",
    "Qwen2.5-7B lcc",
    "Qwen2.5-7B repobench-p",
    "Tinyllama-1.1B xsum",
    "Tinyllama-1.1B gov-report",
    "Tinyllama-1.1B repobench-p",
]

# ==== 创建 3x3 子图 ====
fig, axes = plt.subplots(3, 3, figsize=(18, 15), dpi=600)
axes = axes.flatten()

for i, ax in enumerate(axes):
    line_ours, = ax.plot(budgets, all_time2[i], marker='s', linestyle='-', linewidth=2, color='tab:orange', label="SimCalKV (ours)")
    line_keep, = ax.plot(budgets, all_time1[i], marker='o', linestyle='-', linewidth=2, color='tab:blue', label="KeepKV")
    line_pyramid, = ax.plot(budgets, all_time3[i], marker='^', linestyle='-', linewidth=2, color='tab:green', label="PyramidInfer")
    
    ax.axhline(y=all_time1[i][0], color='red', linestyle=':', linewidth=2, label="100% Cache")
    
    ax.set_xticks(budgets)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("Budgets (%)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title(titles[i], fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

# ==== 整张图统一图例 ====
fig.legend(handles=[line_ours, line_keep, line_pyramid, ax.lines[-1]],
           labels=["SimCalKV", "KeepKV", "PyramidInfer", "100% Cache"],
           loc='upper center', ncol=4, fontsize=14, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存 PNG（高分辨率位图）
plt.savefig("result_9subplots.png", dpi=600, bbox_inches='tight')

# 保存 PDF（矢量图）
plt.savefig("result_9subplots.pdf", bbox_inches='tight')

plt.show()
