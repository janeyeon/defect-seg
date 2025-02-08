import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN

def visualize_patch_clusters(support_feat, img_shape, patch_size=(32, 32), eps=0.5, min_samples=5):
    """
    DBSCAN으로 클러스터링한 패치를 이미지에 매핑하여 시각화

    Args:
        support_feat (torch.Tensor): (S, C) 크기의 feature 벡터 (e.g., (1369, 768))
        img_shape (tuple): 원본 이미지 크기 (H, W, C) e.g., (224, 224, 3)
        patch_size (tuple): 패치 크기 e.g., (32, 32)
        eps (float): DBSCAN의 eps 파라미터
        min_samples (int): DBSCAN의 min_samples 파라미터
    """

    S, C = support_feat.shape
    features = support_feat.cpu().numpy()  # NumPy 변환

    # DBSCAN Clustering 수행
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(features)  # (S,)

    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels[unique_labels != -1])  # Noise 제외

    # 원본 이미지 크기
    H, W, _ = img_shape
    ph, pw = patch_size  # 패치 크기

    # 패치 개수 계산
    grid_h = H // ph
    grid_w = W // pw

    if grid_h * grid_w != S:
        raise ValueError(f"패치 수 불일치: 예상 ({grid_h * grid_w}) != 입력 ({S})")

    # 클러스터 색상 지정 (Noise는 회색)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_clusters)]  # 10개 색 반복 사용
    colors.insert(0, (0.5, 0.5, 0.5, 0.5))  # Noise (-1) → 회색

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # 좌표계 반전

    for i, label in enumerate(cluster_labels):
        row = i // grid_w
        col = i % grid_w

        x = col * pw
        y = row * ph

        rect = patches.Rectangle((x, y), pw, ph, linewidth=2, edgecolor=colors[label], facecolor=colors[label])
        ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"DBSCAN Patch Clustering (Clusters: {num_clusters})")
    plt.show()


