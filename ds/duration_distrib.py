import polars as pl
import matplotlib.pyplot as plt

df = pl.read_csv("./ds/DCLDE_w_Buzzes.csv", null_values=["NA"], ignore_errors=True)
groups = df.group_by("Labels").agg(pl.col("Duration"))
labs = groups["Labels"]
durs = groups["Duration"]

fig, ax = plt.subplots()
ax.set_ylabel('Durations (s)')
ax.set_ylim(-0.5, 5)
ax.set_xticks(range(1, len(labs)+1))
ax.set_xticklabels(labs, rotation=45, ha='right')

plot = ax.violinplot([d.to_list() for d in durs], showmedians=True, quantiles=[[0.75]]*len(durs))
plt.tight_layout()
plt.savefig("./ds/duration_medians_by_label.png")
plt.show()

