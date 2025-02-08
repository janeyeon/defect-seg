import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import DBSCAN, KMeans
import numpy as np

def conv_down_sample_vit(mask, patch_size=14):
    conv_param = torch.ones(patch_size, patch_size).cuda()
    down_sample_mask_vit = F.conv2d(
        mask,
        conv_param.unsqueeze(0).unsqueeze(0),
        stride=patch_size
    )
    down_sample_mask_vit = down_sample_mask_vit / (patch_size * patch_size)
    return down_sample_mask_vit


def cluster_prototypes_Kmeans(support_feat, N_clusters=10):
    S, C = support_feat.shape
    prototypes = []

    features = support_feat.cpu().numpy()  # (1369, 768)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=N_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)  # 각 feature의 클러스터 ID

    cluster_centers = []
    for cluster_id in range(N_clusters):
        cluster_points = features[cluster_labels == cluster_id]  # 해당 클러스터에 속한 feature들

        if len(cluster_points) == 0:
            # 만약 특정 클러스터에 속한 데이터가 없으면 KMeans 중심값을 사용
            cluster_centers.append(kmeans.cluster_centers_[cluster_id])
        else:
            # Weighted GAP 스타일: 클러스터 내의 feature들의 평균
            cluster_centers.append(cluster_points.mean(axis=0))  

    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32).to(support_feat.device)

    prototypes.append(cluster_centers)  # (N_clusters, 768)

    return torch.stack(prototypes)  # (B, N_clusters, 768)

def cluster_prototypes_dbscan(support_feat, eps=0.5, min_samples=5):
    S, C = support_feat.shape
    # support_feat = support_feat.permute(0,2,3,1).reshape(B,S,C)
    prototypes = []

    features = support_feat.cpu().numpy()  # (1369, 768)

    # DBSCAN Clustering 수행
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(features)  # (1369,)

    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]  # Noise(-1) 제거

    cluster_centers = []
    for label in unique_labels:
        cluster_points = features[cluster_labels == label]
        cluster_centers.append(cluster_points.mean(axis=0))  # 클러스터의 평균 벡터를 centroid로 사용

    # Noise 데이터를 고려하여, 클러스터가 너무 적으면 random feature 추가
    while len(cluster_centers) < 3:  # 최소 3개 prototype 보장
        random_idx = np.random.choice(S, 1)[0]
        cluster_centers.append(features[random_idx])

    cluster_centers = torch.tensor(np.stack(cluster_centers), dtype=torch.float32).to(support_feat.device)

    prototypes.append(cluster_centers)  # (N_clusters, 768)

    return torch.stack(prototypes)  # List of (N_clusters, 768)

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask, patch_size=14, conv_vit_down_sampling=False):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    if conv_vit_down_sampling:
        mask = conv_down_sample_vit(mask, patch_size=patch_size)
    else:
        mask = F.interpolate(
            (mask == 1).float(), q.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0]
    similarity = similarity.view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_normal_similarity(tmp_q, tmp_s, mask, shot, patch_size=14, conv_vit_down_sampling=False):
    tmp_s = rearrange(tmp_s, "(b n) c h w -> b n c h w", n=shot)
    bs, shot, d, h, w = tmp_s.shape
    if conv_vit_down_sampling:
        tmp_mask = conv_down_sample_vit(mask, patch_size=patch_size)
    else:
        tmp_mask = F.interpolate(mask,
                                 size=(h, w),
                                 mode="bilinear",
                                 align_corners=False)
    tmp_mask = rearrange(tmp_mask, "(b n) 1 h w -> b n 1 h w", n=shot)

    # b h*w c
    tmp_q = tmp_q.reshape(bs, d, -1).permute(0, 2, 1)

    tmp_s = tmp_s.reshape(bs, shot, d, -1).permute(0, 2, 1, 3).reshape(bs, d, -1).permute(
        0, 2, 1)
    tmp_mask = tmp_mask.reshape(bs, shot, 1, -1).permute(0, 2, 1, 3).reshape(bs, 1, -1)

    l2_normalize_s = F.normalize(tmp_s, dim=2) #0-1사이로 만듦
    l2_normalize_q = F.normalize(tmp_q, dim=2)
    #! 뭔가 코드가 이상함 ㅇㅇ 
    # # b hw (n*hw)
    # similarity = torch.bmm(l2_normalize_q, l2_normalize_s.permute(0, 2, 1))

    # # for abnormal segmentation
    # normal_similarity = similarity * (1 - tmp_mask)
    
    
     # b hw (n*hw)
    # breakpoint()
    # l2_normalize_q : 4, 1369, 768
    # l2_normalize_s : 4, 5476, 768
    # similarity: 4, 1369, 5476
    # normal_similarity.max(2)[0] : 4, 1369
    # tmp_mask : 4, 1, 5476
    similarity = torch.bmm(l2_normalize_q, l2_normalize_s.permute(0, 2, 1))

    # for abnormal segmentation
    normal_similarity = similarity 
    
    
    normal_cos_dis = 1 - normal_similarity.max(2)[0]

    min_max_abnormal_dis = normal_cos_dis.view(bs, 1, h, w)
    return min_max_abnormal_dis


def get_similarity_loss(tmp_q, tmp_s, mask, shot, patch_size=14, conv_vit_down_sampling=False):
    tmp_s = rearrange(tmp_s, "(b n) c h w -> b n c h w", n=shot)
    bs, shot, d, h, w = tmp_s.shape
    if conv_vit_down_sampling:
        tmp_mask = conv_down_sample_vit(mask, patch_size=patch_size)
    else:
        tmp_mask = F.interpolate(mask,
                                 size=(h, w),
                                 mode="bilinear",
                                 align_corners=False)
    tmp_mask = rearrange(tmp_mask, "(b n) 1 h w -> b n 1 h w", n=shot)

    # b h*w c
    tmp_q = tmp_q.reshape(bs, d, -1).permute(0, 2, 1)

    tmp_s = tmp_s.reshape(bs, shot, d, -1).permute(0, 2, 1, 3).reshape(bs, d, -1).permute(
        0, 2, 1)
    tmp_mask = tmp_mask.reshape(bs, shot, 1, -1).permute(0, 2, 1, 3).reshape(bs, 1, -1)

    l2_normalize_s = F.normalize(tmp_s, dim=2)
    l2_normalize_q = F.normalize(tmp_q, dim=2)

    # b hw (n*hw)
    similarity = torch.bmm(l2_normalize_q, l2_normalize_s.permute(0, 2, 1))

    # for abnormal segmentation
    normal_similarity = similarity * (1 - tmp_mask)
    normal_cos_dis = 1 - normal_similarity.max(2)[0]

    min_max_abnormal_dis = normal_cos_dis.view(bs, 1, h, w)
    return min_max_abnormal_dis



#! Add loss for Chamfer distance


import torch
import torch.nn.functional as F
import math

##############################################
# 1) batch 마스크에서 bbox 일괄 추출
##############################################
# def get_bboxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
#     """
#     masks: [B, 1, H, W]  (배치 단위 이진마스크)
#     반환: [B, 4] => 각 배치별 (ymin, ymax, xmin, xmax)
    
#     PyTorch 2.0의 scatter_reduce_ (amin/amax)를 이용한 벡터 연산 예시.
#     (masks > 0).nonzero() -> shape [N, 4], columns = [batch_idx, channel_idx, y, x].
#     """
#     # (masks>0) 위치만 추출 => coords.shape [N, 4]
#     coords = (masks > 0).nonzero(as_tuple=False)
#     # coords[:, 0] = b, coords[:, 2] = y, coords[:, 3] = x
#     b = coords[:, 0]
#     y = coords[:, 2]
#     x = coords[:, 3]

#     B = masks.size(0)
#     device = masks.device
#     dtype = masks.dtype

#     # 초기값
#     y_min = torch.full((B,), math.inf, device=device, dtype=y.dtype)
#     y_max = torch.full((B,), -math.inf, device=device, dtype=y.dtype)
#     x_min = torch.full((B,), math.inf, device=device, dtype=x.dtype)
#     x_max = torch.full((B,), -math.inf, device=device, dtype=x.dtype)

#     # amin/amax로 scatter
#     # PyTorch 2.0+ scatter_reduce_ 사용 (in-place)
#     y_min.scatter_reduce_(0, b, y, reduce='amin')
#     y_max.scatter_reduce_(0, b, y, reduce='amax')
#     x_min.scatter_reduce_(0, b, x, reduce='amin')
#     x_max.scatter_reduce_(0, b, x, reduce='amax')

#     # 스택하여 [B, 4]
#     bboxes = torch.stack([y_min, y_max, x_min, x_max], dim=1)
#     return bboxes.long()  # int로 변환


# def get_bboxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
#     """
#     masks: [B, 1, H, W]  (이진 마스크)
#     반환: [B, 4] => (ymin, ymax, xmin, xmax)  (int형)
    
#     - 마스크가 전부 0인 샘플은 scatter_reduce_ 결과가 inf/-inf가 되어버림
#       => 이를 (0,0,0,0)으로 처리하거나 원하는 기본값으로 처리.
#     """
#     coords = (masks > 0).nonzero(as_tuple=False)
#     b = coords[:, 0]
#     y = coords[:, 2]
#     x = coords[:, 3]

#     B = masks.size(0)
#     device = masks.device

#     # 초기값
#     y_min = torch.full((B,), math.inf, device=device, dtype=torch.float)
#     y_max = torch.full((B,), -math.inf, device=device, dtype=torch.float)
#     x_min = torch.full((B,), math.inf, device=device, dtype=torch.float)
#     x_max = torch.full((B,), -math.inf, device=device, dtype=torch.float)

#     # scatter_reduce_ (PyTorch 2.0+)
#     y_min.scatter_reduce_(0, b, y.float(), reduce='amin')
#     y_max.scatter_reduce_(0, b, y.float(), reduce='amax')
#     x_min.scatter_reduce_(0, b, x.float(), reduce='amin')
#     x_max.scatter_reduce_(0, b, x.float(), reduce='amax')

#     # 이제 inf / -inf로 남아있는 곳 처리
#     # => 마스크가 전부 0인 배치
#     inf_mask = torch.isinf(y_min) | torch.isinf(y_max) | torch.isinf(x_min) | torch.isinf(x_max)
#     if inf_mask.any():
#         # 원하는 대로 처리: 예) 전부 0으로 세팅
#         y_min[inf_mask] = 0
#         y_max[inf_mask] = 0
#         x_min[inf_mask] = 0
#         x_max[inf_mask] = 0

#     # long으로 변환
#     bboxes = torch.stack([y_min, y_max, x_min, x_max], dim=1).long()

#     return bboxes

# ##############################################
# # 2) bbox로부터 grid를 만들고 grid_sample
# ##############################################
# def build_grid_from_bboxes(bboxes: torch.Tensor, 
#                            src_h: int, src_w: int, 
#                            out_size=(64, 64)) -> torch.Tensor:
#     """
#     bboxes: [B, 4] => (ymin, ymax, xmin, xmax)
#     src_h, src_w: 원본 이미지(또는 feature)의 H, W
#     out_size: (H_out, W_out)
    
#     반환: grid: [B, H_out, W_out, 2]
#       => 각 배치별 bounding box 영역을 -1..1로 정규화한 sampling 좌표
#     """
#     B = bboxes.size(0)
#     out_h, out_w = out_size

#     # outH,outW 좌표계를 0.5 ~ (outH-0.5), 0.5 ~ (outW-0.5)로 잡아 미세하게 가운데 샘플링
#     rows = torch.arange(out_h, device=bboxes.device).float() + 0.5
#     cols = torch.arange(out_w, device=bboxes.device).float() + 0.5
#     # meshgrid => [out_h, out_w]
#     grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')  
#     # 배치 차원을 맞추기 위해 [B, out_h, out_w]
#     grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, out_h, out_w]
#     grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, out_h, out_w]

#     # bbox: [B, 4]
#     #   y_min, y_max, x_min, x_max
#     y_min, y_max, x_min, x_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
#     # 각 크기
#     h_range = (y_max - y_min + 1).clamp(min=1).unsqueeze(-1).unsqueeze(-1).float()
#     w_range = (x_max - x_min + 1).clamp(min=1).unsqueeze(-1).unsqueeze(-1).float()
#     y_min_  = y_min.unsqueeze(-1).unsqueeze(-1).float()
#     x_min_  = x_min.unsqueeze(-1).unsqueeze(-1).float()

#     # 실제 (y, x) 좌표 = y_min + grid_y*(h_range/out_h)
#     out_y = y_min_ + grid_y * (h_range / out_h)
#     out_x = x_min_ + grid_x * (w_range / out_w)

#     # grid_sample 은 입력이미지의 (0,0)~(H-1,W-1)를 -1..1로 정규화
#     norm_y = (out_y / (src_h - 1)) * 2 - 1  # [B, outH, outW]
#     norm_x = (out_x / (src_w - 1)) * 2 - 1  # [B, outH, outW]

#     # 최종 grid: [B, out_h, out_w, 2], 순서 (x, y)
#     grid = torch.stack([norm_x, norm_y], dim=-1)
#     return grid

# ##############################################
# # 3) Chamfer distance (DT 활용) & SSIM
# ##############################################
# def chamfer_distance_batch(y_bin: torch.Tensor, s_y_bin: torch.Tensor) -> torch.Tensor:
#     """
#     y_bin, s_y_bin: [B, 1, H_out, W_out], 이미 threshold된 (0/1) 마스크
#     => 배치 단위로 distance transform을 구한 뒤, 
#        Chamfer = mean_over_batch( sum(dt(s_y)*y / sum(y)) + sum(dt(y)*s_y / sum(s_y)) ).
#     Kornia의 distance_transform을 이용 (L2).
#     """
#     import kornia
#     # distance transform
#     dist_y  = kornia.contrib.distance_transform(y_bin)   # [B, 1, H_out, W_out]
#     dist_sy = kornia.contrib.distance_transform(s_y_bin) # [B, 1, H_out, W_out]

#     eps = 1e-8
#     # A->B 항: dt_sy * y_bin
#     c1 = (dist_sy * y_bin).sum(dim=(1,2,3)) / (y_bin.sum(dim=(1,2,3)) + eps)
#     # B->A 항: dist_y * s_y_bin
#     c2 = (dist_y * s_y_bin).sum(dim=(1,2,3)) / (s_y_bin.sum(dim=(1,2,3)) + eps)
#     chamfer = c1 + c2  # [B]
#     return chamfer.mean()  # 스칼라

# def ssim_batch(x_cropped: torch.Tensor, s_x_cropped: torch.Tensor,
#                max_val=1.0) -> torch.Tensor:
#     """
#     x_cropped, s_x_cropped: [B, C, H_out, W_out]
#     Kornia를 사용한 배치 SSIM 계산 (0~1 범위 가정)
#     """
#     import kornia
#     # ssim_map: [B, 1, H_out, W_out]
#     ssim_map = kornia.metrics.ssim(x_cropped, s_x_cropped, window_size=11, max_val=max_val)
#     # 배치별 SSIM 맵의 평균 => 그 다음 batch 차원 평균
#     ssim_val = ssim_map.mean(dim=(1,2,3)).mean(dim=0)  # scalar
#     return ssim_val

# ##############################################
# # 4) 최종 "for문 없는" chamfer + ssim 함수
# ##############################################
# def chamfer_and_ssim_loss(
#     x: torch.Tensor,   # [B, C, H, W]
#     s_x: torch.Tensor, # [B, C, H, W]
#     y: torch.Tensor,   # [B, 1, H, W]
#     s_y: torch.Tensor, # [B, 1, H, W]
#     out_size=(64, 64)
# ):
#     """
#     - (1) y, s_y에서 batch-wise bbox 추출 (scatter_reduce_)
#     - (2) batch-wise grid 만들고 F.grid_sample 한 번씩 호출
#     - (3) DT로 Chamfer distance를 구하고, SSIM도 한 번에 계산
#     - (4) {'chamfer': ..., 'ssim': ...} 형태로 반환
#     """
#     B, C, H_, W_ = x.shape

#     # --------------------------
#     # a) 마스크로부터 bbox
#     # --------------------------
#     bboxes_y  = get_bboxes_from_masks(y)   # [B,4]
#     bboxes_sy = get_bboxes_from_masks(s_y) # [B,4]

#     # --------------------------
#     # b) Grid 생성 + crop & resize
#     # --------------------------
#     grid_y  = build_grid_from_bboxes(bboxes_y,  H_, W_, out_size)  # [B, outH, outW, 2]
#     grid_sy = build_grid_from_bboxes(bboxes_sy, H_, W_, out_size)

#     # 이미지/마스크 각각 grid_sample
#     x_cropped   = F.grid_sample(x,  grid_y,  align_corners=False)  # [B, C, outH, outW]
#     s_x_cropped = F.grid_sample(s_x, grid_sy, align_corners=False)
    
#     breakpoint()

#     # y_cropped   = F.grid_sample(y,  grid_y,  align_corners=False)  # [B, 1, outH, outW]
#     # s_y_cropped = F.grid_sample(s_y, grid_sy, align_corners=False)

#     # # 마스크 0/1 이진화 (bilinear로 조금 번졌을 수 있으므로 0.5 임계)
#     # y_bin   = (y_cropped  > 0.5).float()
#     # s_y_bin = (s_y_cropped> 0.5).float()

#     # --------------------------
#     # c) Chamfer distance
#     # --------------------------
#     # chamfer_val = chamfer_distance_batch(y_bin, s_y_bin)

#     # --------------------------
#     # d) SSIM
#     # --------------------------
#     # 이미지가 이미 [0..1] 범위라 가정
#     ssim_val = ssim_batch(x_cropped, s_x_cropped, max_val=1.0)

#     return {
#         # 'chamfer': chamfer_val,
#         'ssim': ssim_val
#     }

# # ##############################################
# # # 간단 테스트
# # ##############################################
# # if __name__ == "__main__":
# #     import torch

# #     B, C, H, W = 4, 1, 128, 128
# #     x   = torch.zeros(B, C, H, W)
# #     s_x = torch.zeros(B, C, H, W)
# #     y   = torch.zeros(B, 1, H, W)
# #     s_y = torch.zeros(B, 1, H, W)

# #     # 예: 첫 번째 배치에만 임의 결함
# #     # x:  (30:60, 30:60)에 0.8
# #     # s_x:(40:70, 60:90)에 0.8
# #     x[0, 0, 30:60, 30:60]   = 0.8
# #     s_x[0, 0, 40:70, 60:90] = 0.8

# #     y[0, 0, 30:60, 30:60]   = 1.0
# #     s_y[0, 0, 40:70, 60:90] = 1.0

# #     result = chamfer_and_ssim_loss(x, s_x, y, s_y, out_size=(64,64))
# #     print("Chamfer:", result['chamfer'].item())
# #     print("SSIM:   ", result['ssim'].item())
from typing import Optional, Tuple
# import torch
# import torch.nn.functional as F

########################################
# 1. 마스크에서 bbox 얻기
########################################
def get_bbox_from_mask_single(mask_2d: torch.Tensor) -> Optional[Tuple[int,int,int,int]]:
    """
    mask_2d: [H, W] (0/1 이진)
    반환: (ymin, ymax, xmin, xmax) 또는 None(마스크 전부 0)
    """
    coords = (mask_2d > 0).nonzero(as_tuple=False)  # [N, 2], (y, x)
    if coords.numel() == 0:  # 전부 0이면
        return None
    y_min = coords[:, 0].min().item()
    y_max = coords[:, 0].max().item()
    x_min = coords[:, 1].min().item()
    x_max = coords[:, 1].max().item()
    return (y_min, y_max, x_min, x_max)

########################################
# 2. bbox 클램핑 (이미지 밖 넘어가면 잘라내기)
########################################
def clamp_bbox(
    bbox: Optional[Tuple[int,int,int,int]],
    H: int, W: int
) -> Optional[Tuple[int,int,int,int]]:
    """
    bbox: (y_min, y_max, x_min, x_max) 또는 None
    H, W: 이미지 크기
    반환: clamp된 bbox, 없으면 None
    """
    if bbox is None:
        return None
    y_min, y_max, x_min, x_max = bbox

    # 범위 벗어나면 clamp
    y_min_c = max(0, y_min)
    x_min_c = max(0, x_min)
    y_max_c = min(H - 1, y_max)
    x_max_c = min(W - 1, x_max)

    # 뒤집힘 체크
    if y_min_c > y_max_c or x_min_c > x_max_c:
        return None

    return (y_min_c, y_max_c, x_min_c, x_max_c)

########################################
# 3. 두 bbox의 교집합(intersection)
########################################
def intersect_bboxes(
    bboxA: Optional[Tuple[int,int,int,int]],
    bboxB: Optional[Tuple[int,int,int,int]]
) -> Optional[Tuple[int,int,int,int]]:
    """
    bboxA, bboxB: (y_min, y_max, x_min, x_max) or None
    반환: 두 bbox 교집합 (더 작은 공통영역) 또는 None
    """
    if bboxA is None or bboxB is None:
        return None
    (Aymin, Aymax, Axmin, Axmax) = bboxA
    (Bymin, Bymax, Bxmin, Bxmax) = bboxB

    # 교집합
    y_min = max(Aymin, Bymin)
    y_max = min(Aymax, Bymax)
    x_min = max(Axmin, Bxmin)
    x_max = min(Axmax, Bxmax)

    # 뒤집힘 체크
    if y_min > y_max or x_min > x_max:
        return None

    return (y_min, y_max, x_min, x_max)

########################################
# 4. safe_crop_with_bbox
########################################
def safe_crop_with_bbox(
    img_3d: torch.Tensor,
    bbox: Optional[Tuple[int,int,int,int]]
) -> torch.Tensor:
    """
    img_3d: [C, H, W]
    bbox: (y_min, y_max, x_min, x_max) or None
    => None이면 [C, 0, 0] 반환
    """
    C, H, W = img_3d.shape
    if bbox is None:
        return torch.empty((C, 0, 0), device=img_3d.device, dtype=img_3d.dtype)

    y_min, y_max, x_min, x_max = bbox
    cropped = img_3d[:, y_min:y_max+1, x_min:x_max+1]
    return cropped

########################################
# 5. SSIM 계산 (단일 샘플)
#    - 여기서는 항상 groups=1 (단일 채널 처리)
########################################
def gaussian_window(window_size, sigma):
    coords = torch.arange(window_size).float()
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0)  # [1, w]

def create_window(window_size, sigma=1.5):
    """
    channel=1만 지원 (groups=1)
    => weight shape: [1, 1, w, w]
    """
    _1D_window = gaussian_window(window_size, sigma)       # [1, window_size]
    _2D_window = _1D_window.transpose(0,1).mm(_1D_window)  # [window_size, window_size]
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)      # [1, 1, w, w]
    return _2D_window

def ssim_single(
    img1: torch.Tensor,  # [C, H, W], 0~1 범위
    img2: torch.Tensor,  # [C, H, W], 0~1 범위
    window_size=11,
    sigma=1.5,
    C1=0.01**2,
    C2=0.03**2
) -> torch.Tensor:
    """
    - 여러 채널이라도 일단 '한꺼번에' 처리(= 'C'축을 batch라고 보고, groups=1).
    - 크기가 다르면 SSIM=0.
    """
    sigma = 3.0
    device = img1.device
    dtype = img1.dtype

    if img1.dim() == 2:
        img1 = img1.unsqueeze(0)  # => [1, H, W]
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0)

    # 지금 구조상, img1.shape= [C, H, W], img2.shape=[C, H, W]
    # => 크기가 다르면 0
    C1_, H1, W1 = img1.shape
    C2_, H2, W2 = img2.shape
    if (C1_ != C2_) or (H1 != H2) or (W1 != W2):
        return torch.tensor(0.0, device=device, dtype=dtype)

    # conv2d 입력으로: [N, 1, H, W] (여기서 N=C)로 변환
    # => depthwise 가 아님. 그냥 'batch'=C로 보고 'channel'=1
    img1_4d = img1.unsqueeze(1)  # => [C, 1, H, W]
    img2_4d = img2.unsqueeze(1)

    # window shape = [1,1, w,w], groups=1
    window = create_window(window_size, sigma=sigma).to(device).to(dtype)
    pad = window_size // 2

    # mu1, mu2 => [C, 1, H, W]
    mu1 = F.conv2d(img1_4d, window, padding=pad, groups=1)
    mu2 = F.conv2d(img2_4d, window, padding=pad, groups=1)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_4d*img1_4d, window, padding=pad, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2_4d*img2_4d, window, padding=pad, groups=1) - mu2_sq
    sigma12   = F.conv2d(img1_4d*img2_4d, window, padding=pad, groups=1) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 최종 [C,1,H,W] => 평균
    return ssim_map.mean()

########################################
# 6. 배치 전체에 대해
#    (1) bboxA, bboxB
#    (2) clamp
#    (3) intersection
#    (4) crop
#    (5) ssim
#    (6) 평균
########################################
def ssim_intersect_bbox_batch(
    x: torch.Tensor,   # [B, C, H, W]
    y: torch.Tensor,   # [B, 1, H, W]
    s_x: torch.Tensor, # [B, C, H, W]
    s_y: torch.Tensor, # [B, 1, H, W]
) -> torch.Tensor:
    """
    각 배치 b 에 대해:
      1) bboxA <- y[b], bboxB <- s_y[b]
      2) clamp_bbox로 이미지 범위 벗어난 부분 자름
      3) intersect_bboxes로 교집합 영역
      4) x[b], s_x[b]에서 safe_crop
      5) ssim_single
    없거나 빈 영역이면 SSIM=0
    최종적으로 batch 평균
    """
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    ssim_vals = []

    for b in range(B):
        # ----------- bbox A,B -----------
        mask_b = y[b, 0]        # [H, W]
        s_mask_b = s_y[b, 0]    # [H, W]

        bboxA = get_bbox_from_mask_single(mask_b)     # Optional[...]
        bboxB = get_bbox_from_mask_single(s_mask_b)   # Optional[...]
        if bboxA is None or bboxB is None:
            ssim_vals.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue

        # ----------- clamp -----------
        bboxA_c = clamp_bbox(bboxA, H, W)
        bboxB_c = clamp_bbox(bboxB, H, W)
        if bboxA_c is None or bboxB_c is None:
            ssim_vals.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue

        # ----------- intersection -----------
        bbox_int = intersect_bboxes(bboxA_c, bboxB_c)
        if bbox_int is None:
            ssim_vals.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue

        # ----------- crop -----------
        x_b   = x[b]   # [C, H, W]
        s_x_b = s_x[b] # [C, H, W]

        cropA = safe_crop_with_bbox(x_b,   bbox_int)  # [C,cropH,cropW] or empty
        cropB = safe_crop_with_bbox(s_x_b, bbox_int)
        # 빈 텐서 => size(1)==0 or size(2)==0 => SSIM=0
        if cropA.size(1) == 0 or cropA.size(2) == 0 or cropB.size(1) == 0 or cropB.size(2) == 0:
            ssim_vals.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue

        # ----------- SSIM -----------
        # val = ssim_single(cropA, cropB)
        # ssim_vals.append(val)

    # ----------- batch 평균 -----------
    ssim_mean = torch.stack(ssim_vals).mean()
    return ssim_mean
