import math
import torch
from torch import nn
import torch.nn.functional as F
from utils.common import freeze_paras, ForwardHook, dice_ce_loss_sum
from utils import load_backbones
from einops import rearrange
from sklearn.cluster import DBSCAN, KMeans
from model.SOFS.Feature_Recorrect import Feature_Recorrect_Module
from model.SOFS.utils import Weighted_GAP, get_similarity, get_normal_similarity, conv_down_sample_vit
import matplotlib.pyplot as plt
import numpy as np
from .lora import LoRALinearLayer, LoRACompatibleLinear
# from model.SOFS.utils import cluster_prototypes_dbscan, cluster_prototypes_Kmeans
#! Add new loss 
from model.SOFS.utils import ssim_intersect_bbox_batch

 
import time
import cudf
import cupy as cp
from cuml.cluster import KMeans
from main import set_seed

import torch.jit
from concurrent.futures import ThreadPoolExecutor


def clustering_cuml_kmeans(support_features, N_clusters):
    normal_supp_feat_cudf = cp.asarray(support_features.detach().cpu().numpy())
    kmeans = KMeans(n_clusters=N_clusters, random_state=2025)

    kmeans.fit(normal_supp_feat_cudf)
    
    return torch.tensor(kmeans.cluster_centers_.get(), dtype=torch.float32).to(support_features.device)


def parallel_cluster_prototypes_Kmeans(support_features, N_clusters):
    """
    Multi-threaded execution of cluster_prototypes_Kmeans for each batch.
    """
    B = len(support_features)
    prototype_r = []
    
    # Use torch.jit.fork for parallel processing on GPU
    for b in range(B):
        prototype_r.append(torch.jit.fork(clustering_cuml_kmeans, support_features[b], N_clusters))
    
    # Collect results
    results = [torch.jit.wait(f) for f in prototype_r]
    
    return torch.stack(results, dim=0)  # 16, N_clusters, 768 (batch_size * shot = B)


class SOFS(nn.Module):
    def __init__(self, cfg):
        super(SOFS, self).__init__()

        if cfg.DATASET.name in ["VISION_V1_ND", "DS_Spectrum_DS_ND", "opendomain_test_dataset_ND", 'ECCV_Contest_ND', "ECCV_Contest_Test_ND"]:
            shot = cfg.DATASET.shot * cfg.DATASET.s_in_shot
        else:
            shot = cfg.DATASET.shot

        prior_layer_pointer = cfg.TRAIN.SOFS.prior_layer_pointer


        set_seed(seed=42)
        backbone = load_backbones(cfg.TRAIN.backbone)

        if cfg.TRAIN.backbone_load_state_dict:
            # for vit
            state_dict = torch.load(cfg.TRAIN.backbone_checkpoint, map_location=torch.device("cpu"))
            backbone.load_state_dict(state_dict, strict=False)
        freeze_paras(backbone)
        set_seed(seed=42)

        self.outputs = {}
        for tmp_layer in prior_layer_pointer:
            forward_hook = ForwardHook(
                self.outputs, tmp_layer
            )
            if cfg.TRAIN.backbone in ['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14_reg' ]:
                network_layer = backbone.__dict__["_modules"]["blocks"][tmp_layer]
            elif cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
                network_layer = backbone.__dict__["_modules"]["layer" + str(tmp_layer)]
            else:
                raise NotImplementedError

            if isinstance(network_layer, torch.nn.Sequential):
                network_layer[-1].register_forward_hook(forward_hook)
            else:
                network_layer.register_forward_hook(forward_hook)
        self.backbone = backbone
        self.shot = shot
        self.prior_layer_pointer = prior_layer_pointer
        self.target_semantic_temperature = cfg.TRAIN.SOFS.target_semantic_temperature
        self.ce_weight = cfg.TRAIN.LOSS.ce_weight
        self.dice_weight = cfg.TRAIN.LOSS.dice_weight

        if cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
            from utils.common import PatchMaker
            self.patch_maker = PatchMaker(3, stride=1)
            self.preprocessing_dim = [1024, 1024]
        elif cfg.TRAIN.backbone in ['dinov2_vitb14']:
            self.preprocessing_dim = [768] * len(prior_layer_pointer)
        elif cfg.TRAIN.backbone in ['dinov2_vitl14']:
            self.preprocessing_dim = [1024] * len(prior_layer_pointer)
        elif cfg.TRAIN.backbone in ['dinov2_vitg14_reg']:
            self.preprocessing_dim = [1536] * len(prior_layer_pointer)
            

        self.cfg = cfg

        reduce_dim = cfg.TRAIN.SOFS.reduce_dim
        fea_dim = sum(self.preprocessing_dim)
        transformer_embed_dim = cfg.TRAIN.SOFS.transformer_embed_dim
        transformer_num_stages = cfg.TRAIN.SOFS.transformer_num_stages
        transformer_nums_heads = cfg.TRAIN.SOFS.transformer_nums_heads

        self.feature_recorrect = Feature_Recorrect_Module(
            shot=shot,
            fea_dim=fea_dim,
            reduce_dim=reduce_dim,
            transformer_embed_dim=transformer_embed_dim,
            prior_layer_pointer=prior_layer_pointer,
            transformer_num_stages=transformer_num_stages,
            transformer_nums_heads=transformer_nums_heads,
            cfg=cfg
        )



    def encode_feature(self, x):
        self.outputs.clear()
        with torch.no_grad():
            _ = self.backbone(x) # DinoVisionTransformer
            
        multi_scale_features = [self.outputs[key] for key in self.prior_layer_pointer]
        
        return multi_scale_features

    @torch.no_grad()
    
    def feature_processing_vit(self, features, mask=None):
        #! 이거 주의
        diff = 1374-1369 # 5
        B, L, C = features[0][:, diff:, :].shape
        h = w = int(math.sqrt(L))
        # breakpoint()
        multi_scale_features = [each_feature[:, diff:, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
                                for each_feature in features]
        if mask is not None:
            # due to the missing mask, we do not use the mask in this stage for HDM and PFE
            multi_scale_features_ = []
            for each_feature in multi_scale_features:
                tmp_mask = F.interpolate(mask,
                                         size=(each_feature.size(2),
                                               each_feature.size(3)),
                                         mode="bilinear",
                                         align_corners=False)
                multi_scale_features_.append(each_feature * tmp_mask)
            # multi scale feature를 mask와 각각 곱함 mask의 크기에 맞게
            
            return multi_scale_features_
        else:
            return multi_scale_features

    def feature_processing_cnn(self, features):
        bs, _, h, w = features[0].shape
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]  # [((bs, h*w, c, 3, 3), [h, w]), ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )  # (bs, h, w, c, 3, 3)
            _features = _features.permute(0, -3, -2, -1, 1, 2)  # (bs, c, 3, 3, h, w)
            perm_base_shape = _features.shape  # (bs, c, 3, 3, h, w)
            _features = _features.reshape(-1, *_features.shape[-2:])  # (bs * c * 3 * 3, h, w)
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )  # (bs, c, 3, 3, h*2, w*2)
            _features = _features.permute(0, -2, -1, 1, 2, 3)  # (bs, h*2, w*2, c, 3, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])  # (bs, h*w, c, 3, 3)
            features[i] = _features
        # [[bs * h * w, c, 3, 3], [bs * (h1 * 2)=h * (w1 * 2)=w, c*2, 3, 3]]
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]  # (bs * h * w, c, 3, 3)

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = [F.adaptive_avg_pool1d(x.reshape(len(x), 1, -1), self.preprocessing_dim[idx]) for idx, x in
                    enumerate(features)]
        features = torch.concat(features, dim=1)
        features = features.reshape(len(features), 1, -1)  # bs * h * w, 1, 2 * self.preprocessing_dim
        features = F.adaptive_avg_pool1d(features, self.preprocessing_dim[-1])
        features = [features.reshape(-1, h, w, self.preprocessing_dim[-1]).permute(0, 3, 1, 2)] * len(
            self.prior_layer_pointer)
        return features

    def generate_query_label(self, x, s_x, s_y):
        x_size = x.size()
        # [4,3,518,518]
        bs_q, _, img_ori_h, img_ori_w = x_size
        # for b in range(bs_q):
        #     image =  x[b].permute(1,2,0).detach().cpu().numpy()
        #     image = (image - image.min()) / (image.max() - image.min())  # 0~1 정규화
        #     plt.imsave(f"./x_original_{b}.png",image)
        # breakpoint()
        patch_size = self.cfg.TRAIN.SOFS.vit_patch_size
        conv_vit_down_sampling = self.cfg.TRAIN.SOFS.conv_vit_down_sampling

        with torch.no_grad():
            
            # [4, 1370, 768] x 6
            query_multi_scale_features = self.encode_feature(x)
            query_features_list = []
            
            if self.cfg.TRAIN.backbone in ['dinov2_vitb14', "dinov2_vitl14", "dinov2_vitg14_reg"]:
                query_features = self.feature_processing_vit(query_multi_scale_features)

            elif self.cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
                query_features = self.feature_processing_cnn(query_multi_scale_features)
            for idx, layer_pointer in enumerate(self.prior_layer_pointer):
                exec("query_feat_{}=query_features[{}]".format(layer_pointer, idx))
                query_features_list.append(eval('query_feat_' + str(layer_pointer)))

            #   Support Feature
            # s_y 는 [4, 4, 1, 518, 518] : [batch size, shot 수 , _ , org_h, org_w]
            mask = rearrange(s_y, "b n 1 h w -> (b n) 1 h w")
            mask = (mask == 1.).float()
            s_x = rearrange(s_x, "b n c h w -> (b n) c h w")
            
            #  [16, 1370, 768] x 6
            support_multi_scale_features = self.encode_feature(s_x)
            support_features_list = []
            if self.cfg.TRAIN.backbone in ['dinov2_vitb14', "dinov2_vitl14", "dinov2_vitg14_reg"]:
                support_features = self.feature_processing_vit(support_multi_scale_features)
            elif self.cfg.TRAIN.backbone in ["resnet50", "wideresnet50", 'antialiased_wide_resnet50_2']:
                support_features = self.feature_processing_cnn(support_multi_scale_features)
            
            for idx, layer_pointer in enumerate(self.prior_layer_pointer):
                exec("supp_feat_{}=support_features[{}]".format(layer_pointer, idx))
                support_features_list.append(eval('supp_feat_' + str(layer_pointer)))
            # support_features_list는 길이가 6 (6개의 layer), 각각의 shape은 [16, 768, 37, 37]
            # supp_feat_bin_list = []
            feature_masks = []
            for i, each_layer_supp_feat in enumerate(support_features_list):
                masks = []
                if conv_vit_down_sampling: # True
                    tmp_mask = conv_down_sample_vit(mask, patch_size=patch_size)
                else:
                    tmp_mask = F.interpolate(
                        mask,
                        size=(each_layer_supp_feat.size(2),
                            each_layer_supp_feat.size(3)),
                        mode="bilinear",
                        align_corners=False
                    )
                #  [16, 768, 1, 1] 임 (prototype)
                supp_feat_bin = Weighted_GAP(
                    each_layer_supp_feat,
                    tmp_mask
                )
                B, C, H, W = each_layer_supp_feat.shape
                batch_size = B//self.shot
                cos_sim_per_shot = [[] for _ in range(batch_size)]
                each_layer_supp_feat_reshape = each_layer_supp_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, 1369, 768)
                mask_flat = tmp_mask.permute(0, 2, 3, 1).reshape(B, H*W).unsqueeze(-1).expand_as(each_layer_supp_feat_reshape)
                normal_supp_feat = [each_layer_supp_feat_reshape[b][mask_flat[b] == 0] for b in range(B)] 
                normal_supp_feat = [n.view(-1, C) for n in normal_supp_feat]

                #! 이 부분 갯수를 support mask 크기에 따라서 바꾸기
                N_clusters = 70
                # 프로토타입 계산 (벡터화)
                
                # thread or GPU
                # prototype_r = torch.stack([cluster_prototypes_Kmeans(normal_supp_feat[b], N_clusters) for b in range(B)], dim=0)  # 16, N_clusters, 768 (batch_size * shot = B)
                
                prototype_r = parallel_cluster_prototypes_Kmeans(normal_supp_feat, N_clusters)
                                
                ##############################################
                ############ clustering by cuml ##############
               
                # prototype_r = []
                # for b in range(len(normal_supp_feat)):

                #     normal_supp_feat_cudf = cp.asarray(normal_supp_feat[b].detach().cpu().numpy())
                #     kmeans = KMeans(n_clusters=N_clusters, random_state=2025)

                #     kmeans.fit(normal_supp_feat_cudf)

                #     prototype_r.append(torch.tensor(kmeans.cluster_centers_.get(), dtype=torch.float32).to(normal_supp_feat[b].device))

                # prototype_r = torch.stack(prototype_r, dim=0)

                # prototype_r = torch.stack([cluster_prototypes_Kmeans(normal_supp_feat[b], N_clusters) for b in range(B)], dim=0)  # 16, N_clusters, 768

                ############ clustering by cuml ##############
                ##############################################

                
                
                prototype_r = prototype_r.unsqueeze(-1).unsqueeze(-1).view(batch_size, -1, C, 1, 1) # 16, N_clusters, 768, 1, 1
                
                # Support feature 변환 (벡터 연산 사용)                
                supp_feat_b = supp_feat_bin.unsqueeze(1).view(batch_size, self.shot, C, 1, 1) # 16, 1, 768, 1, 1 -> 16, 10, 768
                
                # 프로토타입과 Support feature 결합
                all_proto_b = torch.cat([supp_feat_b, prototype_r], dim=1)  # (batch, shot * C', C, 1, 1), C' = 1 + N_clusters
                all_prototype = all_proto_b.expand(-1, -1, -1, H, W).reshape(batch_size, -1, C, H * W)  # (batch, shot * C', C, H*W)
                query_feat = query_features[i].view(batch_size, C, H*W) 
                
                

                # ##########################################################
                # ####################### apply cdam #######################

                ## options for cdam ##
                CDAM = False
                if CDAM: 
                    do_cdam = True
                    
                    minmax1 = True
                    minmax2 = False
                    multi_scale = False

                    T = 0.1 # 0.1 # 0.2 for voc
                    P = 0.1 #[0.1,0.1,0.1,0.1] # 0.1
                    cdam_coef = 1.0

                    device = query_feat.device
                    
                    bs_s, d, h, w = each_layer_supp_feat.shape
                    bs, _, _ = query_feat.shape
                    hw_shape = (h, w)

                    q_x_norm = query_feat / query_feat.norm(dim=-1, keepdim=True)
                    q_x_norm = q_x_norm.permute(0,2,1)
                    q_x_norm = q_x_norm.unsqueeze(1)


                    _,ts,_,_,_ = all_proto_b.size()
                    s_x = all_proto_b.reshape(bs,1,ts,d)
                    s_x_norm = s_x / s_x.norm(dim=-1, keepdim=True)
                    shots = s_x_norm.size(1)

                    ### from cdam ###
                    logits = q_x_norm @ s_x_norm.permute(0,1,3,2) # bs x (bs_s//bs) x t1 x t2
                    out_dim = logits.shape[-1]
                    
                    logits = logits.permute(0, 1, 3, 2).reshape(bs, shots, out_dim, h, w)

                    logits_no_inter = logits.clone()
                
                    logits_sm = F.softmax(logits_no_inter/T, dim=2)
                    
                    bs, _, c, w_attn, h_attn = logits_sm.shape
                
                    logits_flatten = logits_sm.reshape(bs, shots, c,-1) # 50 196
                    ######

                    #original size
                    softmax_temp = nn.Softmax(dim=3)
                    p_temp = logits_flatten.unsqueeze(4).expand(bs, shots, c, w_attn * h_attn, w_attn * h_attn)
                    q_temp = logits_flatten.unsqueeze(3).expand(bs, shots, c, w_attn * h_attn, w_attn * h_attn)
                    
                    M = (p_temp+q_temp) * 0.5

                    # #jhonson
                    kl_temp =0.5 * (torch.sum(logits_flatten.unsqueeze(4) *  (torch.log(p_temp + 1e-8) - torch.log(M + 1e-8)), dim=2) + \
                                    torch.sum(logits_flatten.unsqueeze(3) *  (torch.log(q_temp + 1e-8) - torch.log(M + 1e-8)), dim=2))
                    
                    kl_temp = 1 - kl_temp # bs x t x t

                    

                    ### minmax_norm 1 ###
                    if minmax1:
                        min_vals = kl_temp.min(dim=-1, keepdim=True)[0].expand(bs, shots, w_attn * h_attn, w_attn * h_attn)  # Find minimum values along dim=2, keep dimensions
                        max_vals = kl_temp.max(dim=-1, keepdim=True)[0].expand(bs, shots, w_attn * h_attn, w_attn * h_attn)  # Find maximum values along dim=2, keep dimensions
                        kl_temp = (kl_temp - min_vals) / (max_vals - min_vals + 1e-8)

                    kl_temp_ori_soft = softmax_temp(kl_temp/P) 
                    
                    ### minmax_norm 2 ###
                    if minmax2:
                        min_vals = kl_temp_ori_soft.min(dim=-1, keepdim=True)[0].expand(bs, shots, w_attn * h_attn, w_attn * h_attn)  # Find minimum values along dim=2, keep dimensions
                        max_vals = kl_temp_ori_soft.max(dim=-1, keepdim=True)[0].expand(bs, shots, w_attn * h_attn, w_attn * h_attn)  # Find maximum values along dim=2, keep dimensions
                        kl_temp_ori = (kl_temp_ori_soft - min_vals) / (max_vals - min_vals + 1e-8)
                    else:
                        kl_temp_ori = kl_temp_ori_soft

                    ### multi-scale version ###
                    
                    # print(logits_sm.shape)
                    # print(q_x.shape)

                    if not multi_scale:
                        js_attn_map = kl_temp_ori.mean(dim=1)
                
                    else:
                        kl_temp_weight = torch.zeros(bs, shots, 12, w*h, w*h, dtype=torch.float16).to(device) 
                        
                        # logits_no_inter_clone = logits_no_inter.clone()

                        step = 0
                        attn_list = []
                        
                        bs, shots, c, w_attn, h_attn = logits_sm.shape

                        kl_sizes = [0.5, 0.65, 0.8]    ############## reduce scales    
                        kl_size_w = [int(w_attn*size) for size in kl_sizes]
                        kl_size_h = [int(h_attn*size) for size in kl_sizes]
                        

                        reshaped_image_features = q_x.permute(0, 2, 1).reshape(1, -1, w, h) #[1, 256, 28, 28]
                    
                        for k_w, k_h in zip(kl_size_w, kl_size_h):
                            # logits_no_inter = nn.functional.interpolate(logits_no_inter_clone, size=(k_w,k_h), mode='bilinear')
                            
                            image_features_no_inter = nn.functional.interpolate(reshaped_image_features, size=(k_w,k_h), mode='bilinear') #[1, 256, k_w, k_h]

                            image_features_no_inter = image_features_no_inter.reshape(1, -1, k_w*k_h).permute(0,2,1)

                            q_x_norm = image_features_no_inter / image_features_no_inter.norm(dim=-1, keepdim=True)
                            q_x_norm = q_x_norm.unsqueeze(1)
                            

                            ### from cdam ###
                            logits = q_x_norm @ s_x_norm.permute(0,1,3,2) # bs x (bs_s//bs) x t1 x t2
                            out_dim = logits.shape[-1]
                            
                            logits = logits.permute(0, 1, 3, 2).reshape(bs, shots, out_dim, k_h, k_w)

                            logits_no_inter = logits.clone()
                            logits_sm = F.softmax(logits_no_inter/T, dim=2)
                            
                            logits_flatten = logits_sm.reshape(bs, shots, c,-1) # 50 196
                            ######

                            #original size
                            softmax_temp = nn.Softmax(dim=3)
                            p_temp = logits_flatten.unsqueeze(4).expand(bs, shots, c, k_h * k_w, k_h * k_w)
                            q_temp = logits_flatten.unsqueeze(3).expand(bs, shots, c, k_h * k_w, k_h * k_w)
                            
                            
                            M = (p_temp+q_temp) * 0.5
                            

                            # #jhonson
                            kl_temp =0.5 * (torch.sum(logits_flatten.unsqueeze(4) *  (torch.log(p_temp + 1e-8) - torch.log(M + 1e-8)), dim=2) + \
                                            torch.sum(logits_flatten.unsqueeze(3) *  (torch.log(q_temp + 1e-8) - torch.log(M + 1e-8)), dim=2))
                            
                            kl_temp = 1 - kl_temp # bs x t x t

                            kl_temp_up = torch.zeros(bs, shots, w_attn*h_attn, w_attn*h_attn, dtype=torch.float16).to(device) # (1,1,37^2, 37^2)

                            kl_temp = nn.functional.interpolate(kl_temp.reshape(bs*shots, k_h*k_w, k_h, k_w), size=(w,h), mode='bilinear') #1 100 14 14
                            for shot in range(shots):
                                for i in range(bs):
                                    kl_temp_samp = kl_temp[(shot+1)*i].permute(1,2,0).reshape(w,h,k_w,k_h) # 100 14 14 -> 14 14 10 10
                                    kl_temp_samp = nn.functional.interpolate(kl_temp_samp, size=(w,h), mode='bilinear') # 14 14 14 14
                                    kl_temp_samp = kl_temp_samp.permute(2,3,0,1).reshape(w*h,w , h).reshape(w*h,w*h)

                                    kl_temp_up[i, shot] = kl_temp_samp
                            kl_temp = kl_temp_up


                            ### minmax_norm 1 ###
                            if minmax1:
                                min_vals = kl_temp.min(dim=-1, keepdim=True)[0].expand(bs, shots, w*h, w*h)  # Find minimum values along dim=2, keep dimensions
                                max_vals = kl_temp.max(dim=-1, keepdim=True)[0].expand(bs, shots, w*h, w*h)  # Find maximum values along dim=2, keep dimensions
                                kl_temp = (kl_temp - min_vals) / (max_vals - min_vals + 1e-8)

                            kl_temp_1_soft = softmax_temp(kl_temp/P) 
                            
                            ### minmax_norm 2 ###
                            if minmax2:
                                min_vals = kl_temp_1_soft.min(dim=-1, keepdim=True)[0].expand(bs, shots, w*h, w*h)  # Find minimum values along dim=2, keep dimensions
                                max_vals = kl_temp_1_soft.max(dim=-1, keepdim=True)[0].expand(bs, shots, w*h, w*h) # Find maximum values along dim=2, keep dimensions
                                kl_temp_1 = (kl_temp_1_soft - min_vals) / (max_vals - min_vals + 1e-8)
                            else:
                                kl_temp_1 = kl_temp_1_soft

                            attn_list.append(kl_temp_1)

                        js_attn_map = ((torch.stack(attn_list, dim=0).sum(0) +1.0*kl_temp_ori ) / (len(kl_size_w) + 1.0)).mean(dim=1) 

                    q_js_attn = torch.bmm(js_attn_map, query_feat.permute(0,2,1)).permute(0,2,1)
                    query_feat = query_feat + cdam_coef * q_js_attn
                    # query_feat =  q_js_attn * 0.5 + q_js_attn * 0.5

                # ####################### apply cdam #######################
                # ##########################################################

                
                
                # 코사인 유사도 연산 최적화
                norm_each_layer = query_feat.norm(dim=1, keepdim=True).view(batch_size, 1, H * W).repeat(1, self.shot * (1+N_clusters), 1)  # 정규화 , batch, self.shot * C', 1, H*W
                norm_proto = all_prototype.norm(dim=2)  # 정규화  B, shot * C',  H*W                
                # each_layer_supp_feat: B, C, H*W 
                # all_prototype: # (B, C', C, H*W)
                # dot_product = torch.einsum('bchw, bcpw -> bphw', each_layer_supp_feat, all_prototype)  # 내적 연산 최적화
                dot_product = torch.einsum('bch, bpch -> bph', query_feat, all_prototype) # batch, shot * c', h*w
                cos_sim = dot_product / (norm_each_layer * norm_proto + 1e-8)  # 유사도 계산

                cos_sim_per_shot = [[cos_sim[b]] for b in range(batch_size)]

                # 평균 유사도 계산 (벡터화)
                cos_sim_avg = torch.stack([torch.stack(cos_list, dim=0).mean(dim=0) for cos_list in cos_sim_per_shot], dim=0)

                # 최적 채널 인덱스 찾기
                # shot +  shot * (N_clsuter) = 총 prototype 갯수 
                best_channel_idx = cos_sim_avg.argmax(dim=1)

                # 마스크 생성
                #! 찾아낸 argmax값이 0-4 사이로 들어올때 진행하기 
                mask_result = ( (best_channel_idx >= 0) & (best_channel_idx < self.shot) ).float() # batch, H*W
                feature_masks.append(mask_result)
            
            feature_masks = [fm.view(batch_size, H, W) for fm in feature_masks]
            semantic_similarity = torch.stack(feature_masks, dim=1)  # (4, 6, 37, 37)            
        
        
        return semantic_similarity
    
    def normalize_mask(self, mask):
        # mask : [4, 6, 37, 37]
        max_val = mask.max()
        b, c, h, w = mask.shape
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = mask.reshape(b, c, -1)
        # mask = F.softmax(mask / 0.1, dim=-1)
        # mask = F.softmax(mask / 0.5 , dim=-1)
        # mask = F.softmax(mask, dim=-1)
        # mask = F.softmax(mask / 0.8, dim=-1)
        # mask = F.softmax(mask / 1.5, dim=-1)
        # mask = F.softmax(mask / 0.5, dim=-1)
        mask = F.softmax(mask / 0.2, dim=-1)
        
        mask = mask.reshape(b, c, h, w)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        # mask *= max_val
        return mask 
    
    def apply_binary_mask(self, x, final_out, alpha=0.5):
        
        # final_out의 shape을 (B, 1, H, W) -> (B, 3, H, W)로 확장 (broadcasting)
        mask = final_out.expand(-1, x.shape[1], -1, -1)

        # 마스크가 1인 부분은 원본 유지, 0인 부분은 alpha만큼 어둡게
        masked_x = x * mask + (1 - mask) * x * alpha  # 0일 때 alpha만큼 줄이기

        return masked_x    
    
    def save_image(self, input, name=None):
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a 2D array
        data = input
        # Display the array as an image
        plt.imshow(data.detach().cpu().numpy(), cmap='gray')
        if name:
            plt.savefig(name)
        else:
            plt.savefig("./output_image.png")
        # plt.colorbar()
        # plt.show()

    def forward(self, x, s_x, s_y, y=None):
        """_summary_
        Args:
            x (Tensor): torch.Size([4, 3, 518, 518])
                이게 query image 이지 않을까? 
                [batch, channel, height, width] 인 것 같음 
            s_x (Tensor): torch.Size([4, 4, 3, 518, 518])
                이게 4개의 support image 인 것 같음
                [batch, shot 갯수, channel, height, width] 인 것 같음 
            s_y (Tensor): torch.Size([4, 4, 1, 518, 518])
                이게 4개의 support mask 인 것 같음
                [batch, index-mask image 갯수, 1, height, width] 인 것 같음
            y (Tensor, optional): torch.Size([4, 1, 518, 518]). Defaults to None.
                이건 query image의 mask 인듯함, 있을때도 있고 없을 때도 있는듯 
        Returns:
            final_out (Tensor): _description_
        """
        x_size = x.size()
        bs_q, _, img_ori_h, img_ori_w = x_size
        patch_size = self.cfg.TRAIN.SOFS.vit_patch_size
        conv_vit_down_sampling = self.cfg.TRAIN.SOFS.conv_vit_down_sampling
        # final_out, mask_weight, each_normal_similarity, query_multi_scale_features, support_multi_scale_features, mask, sim_loss = self.generate_query_label(x, s_x, s_y)
        
       # final_out = self.generate_query_label(x, s_x, s_y)[:, 0, ...] # 4, 37, 37
        final_out = self.generate_query_label(x, s_x, s_y).mean(1)
        # masked_x = self.apply_binary_mask(x, final_out)
        
        # mask_weight_ = mask_weight.unsqueeze(1).unsqueeze(1)
        # normal_out = F.interpolate(each_normal_similarity, size=(img_ori_h, img_ori_w), mode='bilinear', align_corners=False).squeeze(1)
        # each_normal_similarity_
        if self.cfg.TRAIN.SOFS.meta_cls: # true
            final_out = F.interpolate(final_out.unsqueeze(1), size=(img_ori_h, img_ori_w), mode='bilinear', align_corners=False).squeeze(1)  # 4, 512, 512
        # final_out_prob = torch.sigmoid(final_out).contiguous()
        #     final_out_prob = final_out.contiguous()
        # else:
        #     final_out = F.interpolate(final_out, size=(img_ori_h, img_ori_w), mode='bilinear', align_corners=False)
        #     final_out_prob = torch.softmax(final_out, dim=1)[:, 1, ...].contiguous()

        # final_out_prob = mask_weight_ * final_out_prob + (1 - mask_weight_) * normal_out
        # final_out = torch.cat([1 - final_out_prob.unsqueeze(1), final_out_prob.unsqueeze(1)], dim=1) # 4, 2, 512, 512
        # background = torch.where(final_out_prob.unsqueeze(1) < threshold, 1, 0) 
        # foreground = torch.where(final_out_prob.unsqueeze(1) >= threshold, 1, 0) 0[background, foreground], dim=1) # 4, 2, 512, 512
        final_out = final_out.unsqueeze(1)
        foreground = final_out
        background = 1-final_out
        final_out = torch.cat([background, foreground], dim=1) # 4, 2, 512, 512


        return final_out
