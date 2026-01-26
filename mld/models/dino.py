import math
import copy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed1D(nn.Module):
    def __init__(self, max_len=200, patch_size=4, in_chans=263, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

class MotionTransformer(nn.Module):
    def __init__(self, input_dim=263, embed_dim=256, depth=6, num_heads=4, patch_size=4, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), max_len=300, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed1D(max_len, patch_size, input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        num_patches_max = max_len // patch_size + 10
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_max + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        x = self.patch_embed(x)
        B, N, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :N+1, :]
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch] if epoch < len(self.teacher_temp_schedule) else self.teacher_temp_schedule[-1]
        
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * torch.distributed.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MotionDINO(pl.LightningModule):
    def __init__(self, cfg, datamodule=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['datamodule'])
        
        params = cfg.model.params
        embed_dim = params.embed_dim
        dino_out_dim = 65536 
        
        # 1. Student Network
        self.student_backbone = MotionTransformer(
            input_dim=params.input_dim, embed_dim=embed_dim, 
            depth=params.depth, num_heads=params.num_heads, patch_size=params.patch_size,
            drop_path_rate=0.1 
        )
        self.student_head = DINOHead(embed_dim, dino_out_dim, norm_last_layer=True)

        self.teacher_backbone = MotionTransformer(
            input_dim=params.input_dim, embed_dim=embed_dim, 
            depth=params.depth, num_heads=params.num_heads, patch_size=params.patch_size,
            drop_path_rate=0.0 
        )
        self.teacher_head = DINOHead(embed_dim, dino_out_dim, norm_last_layer=True)
        
        # 手动加载权重
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Freeze Teacher
        for p in self.teacher_backbone.parameters(): p.requires_grad = False
        for p in self.teacher_head.parameters(): p.requires_grad = False

        # 3. Loss
        nepochs = cfg.TRAIN.END_EPOCH if 'TRAIN' in cfg else 100
        warmup_temp_epochs = 10 

        self.dino_loss = DINOLoss(
            out_dim=dino_out_dim,
            ncrops=2 + 4, 
            warmup_teacher_temp=0.04,
            teacher_temp=0.04,
            warmup_teacher_temp_epochs=warmup_temp_epochs,
            nepochs=nepochs,
            student_temp=0.1
        )
        
        self.teacher_momentum = 0.996

    def forward(self, x):
        return self.student_backbone(x)

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.student_head.last_layer.requires_grad_(False)
        else:
            self.student_head.last_layer.requires_grad_(True)

    def training_step(self, batch, batch_idx):
        # Teacher Forward (Global views only)
        with torch.no_grad():
            teacher_outputs = []
            for x in batch[:2]:
                feat = self.teacher_backbone(x)
                out = self.teacher_head(feat)
                teacher_outputs.append(out)
            teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # Student Forward (All views)
        student_outputs = []
        for x in batch:
            feat = self.student_backbone(x)
            out = self.student_head(feat)
            student_outputs.append(out)
        student_outputs = torch.cat(student_outputs, dim=0)

        loss = self.dino_loss(student_outputs, teacher_outputs, self.current_epoch)
        self.log('train/dino_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_after_backward(self):
        self.update_teacher()

    @torch.no_grad()
    def update_teacher(self):
        m = self.teacher_momentum
        for param_q, param_k in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.OPTIM.params.lr, weight_decay=0.04)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.cfg.TRAIN.END_EPOCH, eta_min=1e-6
        )
        return [optim], [scheduler]