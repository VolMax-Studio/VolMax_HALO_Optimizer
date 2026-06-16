import matplotlib.pyplot as plt
import numpy as np

# Set design styles for premium look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette
color_lora = '#2c3e50'       # Slate Blue
color_halo_pre = '#00a8cc'   # Electric Cyan
color_halo_foton = '#ff2e93' # Vibrant Magenta
color_no_adapt = '#e74c3c'   # Soft Red

# --- Plot 1: Accuracy vs Depth ---
depths = [1, 2, 4, 8, 16]
lora_acc = [0.893, 0.893, 0.894, 0.891, 0.894]
halo_pre_acc = [0.851, 0.833, 0.791, 0.768, 0.775]
halo_foton_acc = [0.850, 0.820, 0.776, 0.730, 0.697]
no_adapt_acc = [0.135, 0.075, 0.079, 0.139, 0.094]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(depths, lora_acc, label='LoRA (Backprop)', color=color_lora, marker='o', linewidth=2, linestyle='--')
ax.plot(depths, halo_pre_acc, label='HALO-DFA + Precond (O(1) memory)', color=color_halo_pre, marker='s', linewidth=2.5)
ax.plot(depths, halo_foton_acc, label='HALO-DFA + FOTON (O(1) memory)', color=color_halo_foton, marker='^', linewidth=1.8)
ax.plot(depths, no_adapt_acc, label='No adaptation (frozen baseline)', color=color_no_adapt, marker='x', linewidth=1.5, linestyle=':')

ax.set_xscale('log', base=2)
ax.set_xticks(depths)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_title('MNIST Test Accuracy vs. Network Depth (L)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Depth (L) - Residual Layers', fontsize=11, labelpad=8)
ax.set_ylabel('Test Accuracy', fontsize=11, labelpad=8)
ax.set_ylim(0.0, 1.0)
ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5', loc='lower left')
ax.grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plot1_path = 'assets/mnist_depth_accuracy_precond.png'
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"Plot 1 saved to {plot1_path}")


# --- Plot 2: Accuracy & Memory Footprint vs Rank (r) at L=8 ---
ranks = [8, 16, 32, 64]
lora_rank_acc = [0.891, 0.899, 0.895, 0.897]
halo_rank_acc = [0.768, 0.812, 0.836, 0.857]
lora_mem_kb = [1180, 1311, 1573, 2097]
halo_mem_kb = [283, 304, 348, 442]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot A: Accuracy vs Rank
ax1.plot(ranks, lora_rank_acc, label='LoRA (Backprop)', color=color_lora, marker='o', linewidth=2, linestyle='--')
ax1.plot(ranks, halo_rank_acc, label='HALO-DFA + Precond', color=color_halo_pre, marker='s', linewidth=2.5)
ax1.set_title('MNIST Accuracy vs. Adapter Rank (r)', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Adapter Rank (r)', fontsize=10)
ax1.set_ylabel('Test Accuracy', fontsize=10)
ax1.set_xticks(ranks)
ax1.set_ylim(0.70, 0.95)
ax1.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5', loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot B: Memory Footprint vs Rank
x = np.arange(len(ranks))
width = 0.35

rects1 = ax2.bar(x - width/2, lora_mem_kb, width, label='LoRA (Backprop)', color=color_lora, alpha=0.9)
rects2 = ax2.bar(x + width/2, halo_mem_kb, width, label='HALO-DFA (O(1) memory)', color=color_halo_pre, alpha=0.9)

# Add value labels on top of the bars
for rect in rects1:
    height = rect.get_height()
    ax2.annotate(f'{height}K',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8)

for rect in rects2:
    height = rect.get_height()
    ax2.annotate(f'{height}K',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8)

ax2.set_title('Memory Footprint per Update Step (Bytes)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Adapter Rank (r)', fontsize=10)
ax2.set_ylabel('VRAM Usage (KB)', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels([str(r) for r in ranks])
ax2.set_ylim(0, 2500)
ax2.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5', loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('VolMax HALO-DFA Capacity & Memory Scaling (L=8)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plot2_path = 'assets/mnist_rank_accuracy_memory.png'
plt.savefig(plot2_path, dpi=300)
plt.close()
print(f"Plot 2 saved to {plot2_path}")
