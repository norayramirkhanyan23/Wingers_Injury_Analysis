import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os

sns.set_theme(style="whitegrid", font_scale=1.2)
os.makedirs("visuals", exist_ok=True)

df = pd.read_csv("Wingers_Injury.csv")

df.rename(columns={
    "Missed_days_after_22": "missed_days_after_22",
    "Total Matches ": "Total Matches",
}, inplace=True)

df["total_matches"] = df["matches_before_22"] + df["matches_after_22"]
df["total_injuries"] = df["injuries_before_22"] + df["injuries_after_22"]
df["total_missed_days"] = df["missed_days_before_22"] + df["missed_days_after_22"]
df["pct_matches_before"] = df["matches_before_22"] / df["total_matches"] * 100
df["injury_growth_factor"] = df["injuries_after_22"] / (df["injuries_before_22"].replace(0, 0.1))
df["missed_days_growth_factor"] = df["missed_days_after_22"] / (df["missed_days_before_22"].replace(0, 0.1))

print("\n=== Dataset Preview ===")
print(df.head())

plt.figure(figsize=(10, 6))
sns.barplot(x="Player_Name", y="matches_before_22", data=df, color="#3498db", label="Before 22")
sns.barplot(x="Player_Name", y="matches_after_22", data=df, color="#f1c40f",
            bottom=df["matches_before_22"], label="After 22")
plt.xticks(rotation=45, ha="right")
plt.title("Career Match Distribution: Before vs After Age 22", weight="bold")
plt.xlabel("Player", weight="bold")
plt.ylabel("Matches Played", weight="bold")
plt.legend(title="Age Period")
plt.tight_layout()
plt.savefig("visuals/match_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Player_Name", y="injuries_before_22", data=df, color="#ff99cc", label="Before 22")
sns.barplot(x="Player_Name", y="injuries_after_22", data=df, color="#000000",
            bottom=df["injuries_before_22"], label="After 22")
plt.xticks(rotation=45, ha="right")
plt.title("Injury Count Growth: Before vs After Age 22", weight="bold")
plt.xlabel("Player", weight="bold")
plt.ylabel("Number of Injuries", weight="bold")
plt.legend(title="Age Period")
sns.despine()
plt.tight_layout()
plt.savefig("visuals/injury_growth.png", dpi=300)
plt.show()

import numpy as np
from matplotlib.lines import Line2D

plt.figure(figsize=(11, 7))
palette = {"RW": "#000000", "LW": "#ff99cc"}

ax = sns.scatterplot(
    data=df,
    x="matches_before_22",
    y="missed_days_after_22",
    hue="Position",
    size="injuries_after_22",
    sizes=(60, 300),
    alpha=0.85,
    palette=palette,
    legend=False
)

for i, row in df.iterrows():
    ax.text(
        row["matches_before_22"],
        row["missed_days_after_22"] + 40,
        row["Player_Name"],
        fontsize=9,
        weight="bold",
        ha="center",
        va="bottom",
        alpha=0.9
    )

pos_handles = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=palette['RW'], markersize=10, label='RW'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=palette['LW'], markersize=10, label='LW'),
]
leg1 = ax.legend(handles=pos_handles, title="Position",
                 loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
ax.add_artist(leg1)

size_levels = [10, 20, 30, 40, 50]
size_handles = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor="#666666",
           markersize=np.interp(s, [min(size_levels), max(size_levels)], [6, 14]),
           label=str(s))
    for s in size_levels
]
ax.legend(handles=size_handles, title="Injuries after 22",
          loc='upper left', bbox_to_anchor=(1.02, 0.45), frameon=True)

ax.set_title("Does Early Workload Predict Later Injury Burden?", fontsize=14, weight="bold")
ax.set_xlabel("Matches Before Age 22", fontsize=12, weight="bold")
ax.set_ylabel("Missed Days After 22", fontsize=12, weight="bold")
sns.despine()
plt.tight_layout()
plt.savefig("visuals/workload_vs_durability_labeled.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
corr_vars = {
    "matches_before_22": "Matches Before 22",
    "matches_after_22": "Matches After 22",
    "injuries_after_22": "Injuries After 22",
    "missed_days_after_22": "Missed Days After 22"
}
corr = df[list(corr_vars.keys())].corr()
corr.rename(index=corr_vars, columns=corr_vars, inplace=True)
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=1,
    cbar_kws={'label': 'Correlation Strength'},
    annot_kws={"weight": "bold", "fontsize": 10}
)
plt.title("Correlation Matrix: Workload & Injury Metrics", fontsize=13, weight="bold", pad=12)
plt.xticks(rotation=25, ha="right", weight="bold")
plt.yticks(rotation=0, weight="bold")
sns.despine()
plt.tight_layout()
plt.savefig("visuals/correlation_heatmap_styled.png", dpi=300)
plt.show()

from math import pi

players = ["Lionel Messi", "Cristiano Ronaldo"]
metrics = ["matches_before_22", "matches_after_22", "injuries_after_22", "missed_days_after_22"]
metric_labels = ["Matches Before 22", "Matches After 22", "Injuries After 22", "Missed Days After 22"]
subset = df[df["Player_Name"].isin(players)][["Player_Name"] + metrics].copy()

norm = subset.copy()
for col in metrics:
    maxv = subset[col].max()
    norm[col] = 0 if maxv == 0 else subset[col] / maxv

N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.set_facecolor("#f9f9f9")
ax.grid(color="#d0d0d0", linewidth=0.8)
ax.spines["polar"].set_color("#bdbdbd")

colors = {"Lionel Messi": "#3498db", "Cristiano Ronaldo": "#ffbf00"}

for _, row in norm.iterrows():
    values = row[metrics].tolist() + [row[metrics].tolist()[0]]
    color = colors[row["Player_Name"]]
    ax.plot(angles, values, linewidth=2.8, color=color, label=row["Player_Name"], marker="o", markersize=6)
    ax.fill(angles, values, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="#666666")
plt.title("Best of the Best: Messi vs Ronaldo", fontsize=15, fontweight="bold", pad=18)
legend = plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True, title="Players")
legend.get_frame().set_alpha(0.9)
plt.tight_layout()
plt.savefig("visuals/radar_messi_vs_ronaldo.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid", font_scale=1.2)
os.makedirs("visuals", exist_ok=True)

lamine = {
    "Player_Name": "Lamine Yamal",
    "matches_before_22": 122,
    "injuries_before_22": 8,
    "missed_days_before_22": 200
}

avg_injury_growth_factor = 4.4
avg_missed_days_growth_factor = 4.3

lamine["pred_injuries_after_22"] = lamine["injuries_before_22"] * avg_injury_growth_factor
lamine["pred_missed_days_after_22"] = lamine["missed_days_before_22"] * avg_missed_days_growth_factor

print("=== LAMINE YAMAL DURABILITY PROJECTION ===")
print(f"Matches before 22: {lamine['matches_before_22']}")
print(f"Injuries before 22: {lamine['injuries_before_22']}")
print(f"Missed days before 22: {lamine['missed_days_before_22']}")
print(f"Projected injuries after 22: {lamine['pred_injuries_after_22']:.0f}")
print(f"Projected missed days after 22: {lamine['pred_missed_days_after_22']:.0f}")

plt.figure(figsize=(10, 6))
gs = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.25)

ax1 = plt.subplot(gs[0, 0])
bars1 = ax1.bar(["Before 22", "Predicted After 22"],
                [lamine["injuries_before_22"], lamine["pred_injuries_after_22"]],
                color=["#66b3ff", "#ff66cc"], edgecolor="black", linewidth=0.8)
ax1.set_title("Injuries", weight="bold")
ax1.set_ylabel("Count")
for b in bars1:
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
             f"{b.get_height():.0f}", ha="center", va="bottom", weight="bold")

ax2 = plt.subplot(gs[0, 1])
bars2 = ax2.bar(["Before 22", "Predicted After 22"],
                [lamine["missed_days_before_22"], lamine["pred_missed_days_after_22"]],
                color=["#66b3ff", "#ff66cc"], edgecolor="black", linewidth=0.8)
ax2.set_title("Missed Days", weight="bold")
ax2.set_ylabel("Days")
for b in bars2:
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 10,
             f"{b.get_height():.0f}", ha="center", va="bottom", weight="bold")

ax3 = plt.subplot(gs[1, :])
ax3.axis("off")
summary = (
    f"Lamine Yamal — Durability Projection\n"
    f"• Matches before 22: {lamine['matches_before_22']}\n"
    f"• Injuries before 22: {lamine['injuries_before_22']}\n"
    f"• Missed days before 22: {lamine['missed_days_before_22']}\n"
    f"• Predicted injuries after 22: {lamine['pred_injuries_after_22']:.0f}\n"
    f"• Predicted missed days after 22: {lamine['pred_missed_days_after_22']:.0f}\n\n"
    f"Note: Projection uses average post-22 growth factors from elite winger dataset "

)
ax3.text(0.02, 0.9, summary, fontsize=11, va="top")

plt.suptitle("Visual 6 — Lamine Yamal: Single-Player Durability Projection",
             fontsize=14, weight="bold", y=0.98)
plt.tight_layout()
plt.savefig("visuals/visual6_lamine_projection.png", dpi=300)
plt.show()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

summary_df = (
    df[["Player_Name", "injury_growth_factor", "missed_days_growth_factor"]]
    .sort_values("injury_growth_factor", ascending=False)
    .reset_index(drop=True)
)

avg_injury_growth = float(df["injury_growth_factor"].mean())
avg_missed_days_growth = float(df["missed_days_growth_factor"].mean())

top_3_injured = (
    df.sort_values('missed_days_after_22', ascending=False)
    .loc[:, ['Player_Name', 'missed_days_after_22', 'injuries_after_22']]
    .head(3)
)
top_3_ironmen = (
    df.sort_values('injuries_after_22', ascending=True)
    .loc[:, ['Player_Name', 'injuries_after_22', 'missed_days_after_22']]
    .head(3)
)

console.rule("[bold magenta]FOOTBALL INJURY ANALYSIS — Summary[/]")

insight_text = (
    f"[bold]Average injury growth factor:[/] [bold yellow]{avg_injury_growth:.2f}x[/]\n"
    f"[bold]Average missed days growth factor:[/] [bold yellow]{avg_missed_days_growth:.2f}x[/]\n"
    "\n[dim]Note:[/] Growth factors > 1 indicate an increase after age 22."
)
console.print(Panel(insight_text, title="KEY INSIGHTS", title_align="left", border_style="magenta"))

t = Table(title="Injury Growth Summary (sorted by injury growth)", box=box.SIMPLE_HEAVY, show_lines=False,
          header_style="bold")
t.add_column("#", justify="right", style="dim")
t.add_column("Player", style="bold")
t.add_column("Injury Growth (x)", justify="right")
t.add_column("Missed Days Growth (x)", justify="right")

hi_injury = summary_df["injury_growth_factor"].quantile(0.75)
hi_missed = summary_df["missed_days_growth_factor"].quantile(0.75)

for i, row in summary_df.iterrows():
    ig = row["injury_growth_factor"]
    mg = row["missed_days_growth_factor"]
    ig_style = "red" if ig >= hi_injury else "white"
    mg_style = "red" if mg >= hi_missed else "white"
    t.add_row(
        str(i + 1),
        str(row["Player_Name"]),
        f"[{ig_style}]{ig:.2f}[/{ig_style}]",
        f"[{mg_style}]{mg:.2f}[/{mg_style}]",
    )
console.print(t)

t2 = Table(title="Top 3 — Most Missed Days After 22", box=box.SIMPLE_HEAVY, header_style="bold")
t2.add_column("#", justify="right", style="dim")
t2.add_column("Player", style="bold")
t2.add_column("Missed Days After 22", justify="right")
t2.add_column("Injuries After 22", justify="right")
for i, row in top_3_injured.reset_index(drop=True).iterrows():
    t2.add_row(str(i + 1), str(row["Player_Name"]), f"{int(row['missed_days_after_22'])}",
               f"{int(row['injuries_after_22'])}")
console.print(t2)

t3 = Table(title="Top 3 — Ironmen (Fewest Injuries After 22)", box=box.SIMPLE_HEAVY, header_style="bold")
t3.add_column("#", justify="right", style="dim")
t3.add_column("Player", style="bold")
t3.add_column("Injuries After 22", justify="right")
t3.add_column("Missed Days After 22", justify="right")
for i, row in top_3_ironmen.reset_index(drop=True).iterrows():
    t3.add_row(str(i + 1), str(row["Player_Name"]), f"{int(row['injuries_after_22'])}",
               f"{int(row['missed_days_after_22'])}")
console.print(t3)

console.print(Panel.fit("📁 Visuals saved in [bold]/visuals[/] folder.", border_style="green"))
