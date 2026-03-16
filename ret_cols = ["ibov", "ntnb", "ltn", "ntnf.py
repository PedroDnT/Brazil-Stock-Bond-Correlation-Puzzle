ret_cols = ["ibov", "ntnb", "ltn", "ntnf", "lft_proxy"]

# Monthly availability matrix (1 = data present, 0 = NaN)
avail = master[ret_cols].resample("ME").apply(lambda x: x.notna().mean())
avail.columns = [ASSET_LABELS[c] for c in avail.columns]

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    avail.T,
    cmap="RdYlGn", vmin=0, vmax=1,
    ax=ax, 
    cbar_kws={
        "label": "Data Availability (%)",
        "shrink": 0.8,
        "aspect": 20
    },
    linewidths=0.5,
    linecolor="#E0E0E0",
    fmt=".0%"
)
ax.set_title("Data Availability by Month and Asset Class", 
             fontsize=14, pad=15, fontweight="semibold")
ax.set_xlabel("Date", fontsize=11, labelpad=10)
ax.set_ylabel("")

# Improve x-axis labels
xtick_positions = np.arange(0, len(avail), len(avail)//12)
xtick_labels = [avail.index[i].strftime("%Y") for i in xtick_positions if i < len(avail)]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, rotation=0)

# Mark crisis periods with improved visibility
for name, (s, e) in CRISES.items():
    s_idx = avail.index.searchsorted(pd.Timestamp(s))
    e_idx = avail.index.searchsorted(pd.Timestamp(e))
    ax.axvspan(s_idx, e_idx, color=CRISIS_COLORS[name], alpha=0.2, zorder=0)

plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs("../outputs", exist_ok=True)

plt.savefig("../outputs/fig_data_coverage.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print("✓ Saved: outputs/fig_data_coverage.png")
