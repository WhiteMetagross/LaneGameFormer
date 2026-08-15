import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import random
import sys
import time
import shutil
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import config as cfg
from src.model.lanegcn import get_model
from src.data_loader import CustomDataset, collate_fn
from utils import Logger, load_pretrain

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    import glob
    
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by epoch number (extracted from filename)
    def extract_epoch(path):
        filename = os.path.basename(path)
        try:
            return float(filename.replace('.ckpt', ''))
        except:
            return 0
    
    checkpoints.sort(key=extract_epoch)
    return checkpoints[-1]  # Return latest

def load_checkpoint(checkpoint_path, net, opt):
    """Load model and optimizer state from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    net.load_state_dict(checkpoint["state_dict"])
    
    # Load optimizer state
    if "opt_state" in checkpoint:
        opt.opt.load_state_dict(checkpoint["opt_state"])
    
    epoch = checkpoint.get("epoch", 0)
    print(f"Resumed from epoch: {epoch}")
    
    return epoch

def main(resume_from_checkpoint=True, specific_checkpoint=None):
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    config = cfg.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    config, net, loss, post_process, opt = get_model(config)
    net.to(device)
    loss.to(device)
    post_process.to(device)

    save_dir = config["save_dir"]
    
    # Check for existing checkpoint to resume from
    start_epoch = config["epoch"]
    if resume_from_checkpoint:
        checkpoint_to_load = None
        
        if specific_checkpoint:
            if os.path.exists(specific_checkpoint):
                checkpoint_to_load = specific_checkpoint
                print(f"Using specific checkpoint: {specific_checkpoint}")
            else:
                print(f"Specific checkpoint not found: {specific_checkpoint}")
                print("Falling back to latest checkpoint search")
        
        if not checkpoint_to_load:
            checkpoint_to_load = find_latest_checkpoint(save_dir)
        
        if checkpoint_to_load:
            try:
                start_epoch = load_checkpoint(checkpoint_to_load, net, opt)
                print(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint {checkpoint_to_load}: {e}")
                print("Starting training from scratch")
                start_epoch = config["epoch"]
        else:
            print("No checkpoint found, starting training from scratch")
    else:
        print("Starting training from scratch (resume disabled)")
    
    log = os.path.join(save_dir, "log")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    src_dirs = [os.path.join(root_path, "src")]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    train_dataset = CustomDataset(config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = CustomDataset(config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    epoch = start_epoch
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    print(f"Training from epoch {epoch} to {config['num_epochs']} ({remaining_epochs} epochs remaining)")
    
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader, device)

def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader, device):
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = config["display_iters"]
    val_iters = config["val_iters"]

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(tqdm(train_loader)):
        epoch += epoch_per_batch
        data = dict(data)
        
        output = net(data, device)
        loss_out = loss(output, data, device)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)
        
        # Clear GPU cache periodically to avoid memory fragmentation
        if i % 50 == 0:
            torch.cuda.empty_cache()

        num_iters = int(np.round(epoch * num_batches))
        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()
        
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch, device)
        
        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch, device)
            return

def val(config, data_loader, net, loss, post_process, epoch, device):
    net.eval()
    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(tqdm(data_loader)):
        data = dict(data)
        with torch.no_grad():
            output = net(data, device)
            loss_out = loss(output, data, device)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)
    net.train()

def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LaneGCN Training')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from latest checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Start training from scratch')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Load specific checkpoint
        main(resume_from_checkpoint=True, specific_checkpoint=args.checkpoint)
    else:
        # Use resume flag
        main(resume_from_checkpoint=args.resume)