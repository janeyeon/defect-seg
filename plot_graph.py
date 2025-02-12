import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 로그 파일 경로
log_file_path = "./log_FsSS_IVI_VISION_V1_ND_SOFS_54/DATASET_VISION_V1_ND_METHOD_SOFS_RNG_SEED_54_TIME_2025_02_12_19_50.log"  # 적절한 경로로 변경

# 파일 읽기
with open(log_file_path, "r") as file:
    log_data = file.read()

# 정규 표현식을 사용하여 값 추출
pattern = r"mask_size : ([\d\.]+), similarity: ([\d\.]+), .* foreground_iou : ([\d\.]+)"

matches = re.findall(pattern, log_data)

# 리스트로 변환
mask_size_list = [float(m[0]) for m in matches]
similarity_list = [float(m[1]) for m in matches]
foreground_iou_list = [float(m[2]) for m in matches]

# 데이터프레임 생성
df = pd.DataFrame({
    "mask_size": mask_size_list,
    "similarity": similarity_list,
    "foreground_iou": foreground_iou_list
})

# 상관관계 계산 (피어슨, 스피어만, 켄달)
corr_pearson_mask = df["mask_size"].corr(df["foreground_iou"], method="pearson")
corr_spearman_mask = df["mask_size"].corr(df["foreground_iou"], method="spearman")
corr_kendall_mask = df["mask_size"].corr(df["foreground_iou"], method="kendall")

corr_pearson_sim = df["similarity"].corr(df["foreground_iou"], method="pearson")
corr_spearman_sim = df["similarity"].corr(df["foreground_iou"], method="spearman")
corr_kendall_sim = df["similarity"].corr(df["foreground_iou"], method="kendall")

# 산점도 그래프 그리기
plt.figure(figsize=(12, 5))

# mask_size vs foreground_iou
plt.subplot(1, 2, 1)
sns.regplot(x=df["mask_size"], y=df["foreground_iou"], scatter_kws={'alpha':0.7}, line_kws={'color': 'blue'})
plt.xlabel("Mask Size")
plt.ylabel("Foreground IoU")
plt.title(f"Mask Size vs Foreground IoU\nPearson: {corr_pearson_mask:.3f}, Spearman: {corr_spearman_mask:.3f}, Kendall: {corr_kendall_mask:.3f}")

# similarity vs foreground_iou
plt.subplot(1, 2, 2)
sns.regplot(x=df["similarity"], y=df["foreground_iou"], scatter_kws={'alpha':0.7, 'color': 'red'}, line_kws={'color': 'black'})
plt.xlabel("Similarity")
plt.ylabel("Foreground IoU")
plt.title(f"Similarity vs Foreground IoU\nPearson: {corr_pearson_sim:.3f}, Spearman: {corr_spearman_sim:.3f}, Kendall: {corr_kendall_sim:.3f}")

plt.tight_layout()

# 그래프 저장
plt.savefig("correlation_plots.png")

# 상관관계 값 출력
print(f"Mask Size ↔ Foreground IoU Correlations - Pearson: {corr_pearson_mask:.3f}, Spearman: {corr_spearman_mask:.3f}, Kendall: {corr_kendall_mask:.3f}")
print(f"Similarity ↔ Foreground IoU Correlations - Pearson: {corr_pearson_sim:.3f}, Spearman: {corr_spearman_sim:.3f}, Kendall: {corr_kendall_sim:.3f}")
