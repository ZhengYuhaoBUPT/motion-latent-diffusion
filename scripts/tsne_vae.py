import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# MLD imports
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
import random

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

TARGET_ACTIONS = {
    "walk": ["walk", "walking"],
    "run": ["run", "running"],
    "jump": ["jump", "jumping"],
    "kick": ["kick", "kicking"],
    "punch": ["punch", "punching"],
    "sit": ["sit", "sitting"],
    "dance": ["dance", "dancing"],
}

LABEL_ORDER = list(TARGET_ACTIONS.keys())

def get_action_label(text):
    text = text.lower()
    for label, keywords in TARGET_ACTIONS.items():
        for word in keywords:
            if word in text:
                return label
    return None

def extract_vae_features(cfg, model, dataloader, device, max_samples=2000):
    """提取 VAE Latent 特征"""
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Batches"):
            motions = batch["motion"].to(device)
            lengths = batch["length"]
            raw_texts = batch["text"] 

            if cfg.model.vae:
                if hasattr(model, 'vae'):
                    _, dist = model.vae.encode(motions, lengths)
                else:
                    _, dist = model.encode(motions, lengths)
                
                # 确保压平成 2D [Batch, Latent]
                batch_latents = dist.loc.permute(1, 0, 2).reshape(motions.shape[0], -1).cpu().numpy()
                
            else:
                print("❌ Config says 'vae: false', cannot extract VAE features.")
                return [], []

            for i, text in enumerate(raw_texts):
                label = get_action_label(text)
                if label is not None:
                    latents.append(batch_latents[i])
                    labels.append(label)
            
            if len(latents) >= max_samples:
                break
    
    latents = np.array(latents)
    return latents, np.array(labels)

def draw_tsne(output_dir, latents, labels, title_suffix="", xlim=None, ylim=None):
    """
    绘制并保存 t-SNE 图
    参数:
        xlim: tuple (min, max), 例如 (-100, 100)
        ylim: tuple (min, max), 例如 (-100, 100)
    """
    if len(latents) == 0:
        print("⚠️ No samples extracted! Check your dataset or keywords.")
        return

    print(f"🎨 Computing t-SNE for {len(latents)} samples...")
    tsne = TSNE(n_components=2, verbose=1, random_state=42, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(latents)

    df = pd.DataFrame()
    df["x"] = z_embedded[:, 0]
    df["y"] = z_embedded[:, 1]
    df["label"] = labels

    plt.figure(figsize=(10, 8))
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette="bright",
        hue_order=LABEL_ORDER,
        s=60,
        alpha=0.7,
        legend="full"
    )
    
    # 👇【新增功能】设置坐标轴范围
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
        
    # 可选：如果你希望去掉坐标轴刻度（t-SNE 的绝对数值通常没有意义，去掉更美观）
    # plt.xticks([])
    # plt.yticks([])
    
    title = f"VAE Latent Space t-SNE\n{title_suffix}"
    plt.title(title, fontsize=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    filename = f'tsne_vae_{title_suffix.strip().replace(" ", "_")}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"✅ Figure saved to: {save_path}")

def main():
    set_deterministic(42)
    cfg = parse_args(phase="test")
        
    logger = create_logger(cfg, phase="test")
    
    output_dir = Path(os.path.join(cfg.FOLDER, "tsne_visualization"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pl.seed_everything(cfg.SEED_VALUE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading Datasets...")
    datamodule_list = get_datasets(cfg, logger=logger, phase="test")
    datamodule = datamodule_list[0]
    
    datamodule.setup(stage="test") 
    dataloader = datamodule.test_dataloader()

    logger.info(f"Loading Model from Checkpoint: {cfg.TEST.CHECKPOINTS}")
    model = get_model(cfg, datamodule)
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu", weights_only=False)["state_dict"]
        
    # 加载参数到模型
    load_res = model.load_state_dict(state_dict, strict=False)
    print(f"Load result: {load_res}")
    
    model.to(device)
    model.eval()

    is_dino = cfg.TRAIN.MARK == 'dino'
    exp_name = "DINO_VAE" if is_dino else "Baseline_VAE"
    
    latents, labels = extract_vae_features(cfg, model, dataloader, device, max_samples=1500)
    
    if len(latents) > 0:
        draw_tsne(output_dir, latents, labels, title_suffix=exp_name)

if __name__ == "__main__":
    main()