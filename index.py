# 必要なライブラリ
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm

# ✅ CSV の読み込み
# 例: "data.csv"
data = pd.read_csv("data.csv")

# ✅ 列名を統一 (Excel の列名に合わせて変更)
# Excel側の列名 → Python内で統一的に使うラベルへ変更
data = data.rename(columns={
    "参加者ID": "s",
    "独立変数": "A",
    "従属変数": "y"
})

# A(条件) と s(参加者ID) をカテゴリ扱いにする
data["s"] = data["s"].astype("category")
data["A"] = data["A"].astype("category")

# ソート
data = data.sort_values(by=["A", "s"])

# ✅ 記述統計量（Rの summarize）
summary = data.groupby("A")["y"].agg(
    N='count',
    Mean='mean',
    SD='std',
    Min='min',
    Median='median',
    Max='max'
)
summary["SE"] = summary["SD"] / summary["N"] ** 0.5

print("\n▼ 記述統計")
print(summary)

# ✅ 箱ひげ図 + 平均値
plt.figure(figsize=(8, 6))
data.boxplot(column="y", by="A")
plt.scatter(range(1, len(summary)+1), summary["Mean"], marker="x")
plt.title("Boxplot with Mean")
plt.suptitle("")
plt.xlabel("Condition (A)")
plt.ylabel("Measurement (y)")
plt.show()

# ✅ Wilcoxon（対応あり）
# 条件ごとに値を並べ替えてペアにする
pivot = data.pivot(index="s", columns="A", values="y")

stat, p = wilcoxon(pivot.iloc[:, 0], pivot.iloc[:, 1])
print("\n▼ Wilcoxon 検定結果")
print("statistic =", stat)
print("p-value =", p)

# ✅ 効果量（Cohen's r）
z = norm.ppf(1 - p/2)  # 両側検定なので p/2
r = z / (len(data["y"]) ** 0.5)
print("\n▼ 効果量 Cohen's r")
print("r =", r)
