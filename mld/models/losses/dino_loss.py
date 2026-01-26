import torch
import torch.nn as nn
import omegaconf
import typing
from mld.models.dino import MotionDINO

torch.serialization.add_safe_globals([
    typing.Any, 
    omegaconf.dictconfig.DictConfig, 
    omegaconf.listconfig.ListConfig
])

class DinoVAELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.lambda_perceptual = cfg.LOSS.LAMBDA_PERCEPTUAL 
        self.dino_ckpt = cfg.LOSS.DINO_CKPT

        self.mse_loss = nn.MSELoss()
        self.dino_model = None
        
        if self.lambda_perceptual > 0:
            print(f"🦖 [MLD-DINO] Loading MotionDINO from {self.dino_ckpt}...")
            
            try:
                checkpoint = torch.load(self.dino_ckpt, map_location='cpu', weights_only=False)
                
                hparams = checkpoint.get("hyper_parameters", {})
                
                self.dino_model = MotionDINO(**hparams) 
                self.dino_model.load_state_dict(checkpoint["state_dict"], strict=False)
                
            except Exception as e:
                print(f"❌ Manual load failed: {e}")
                raise e

            self.dino_model.eval()
            self.dino_model.freeze()
            print("✅ [MLD-DINO] MotionDINO loaded! Perceptual loss enabled.")

    def forward(self, rs_set):
        loss_logs = {}
        perceptual_loss = 0.0

        if self.lambda_perceptual > 0 and self.dino_model is not None:
            # 提取特征
            feat_ref = self.dino_model.student_backbone(rs_set['m_ref']).detach()
            feat_rst = self.dino_model.student_backbone(rs_set['m_rst'])
            
            loss_val = self.mse_loss(feat_rst, feat_ref)
            perceptual_loss = self.lambda_perceptual * loss_val
            loss_logs['loss_perceptual'] = perceptual_loss.detach()
        
        return perceptual_loss, loss_logs