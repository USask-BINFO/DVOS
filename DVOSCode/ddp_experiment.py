import argparse
import glob
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

from datetime import datetime
from icecream import ic
from functools import partial
from tqdm import tqdm
from typing import Tuple, Optional, Union, Dict, List, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import utilities as utils
from data import get_dataloader
from model import get_model
from diffusion import SimpleDiffusion

warnings.filterwarnings("ignore")

# Restrict the number of threads used by PyTorch DataLoader
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
torch.set_num_interop_threads(2)


def setup(rank, world_size):
    if isinstance(rank, int): 
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prediction_visualizer(inverse_normalize: Callable, 
                          out_dir: str,
                          ground_images: torch.Tensor, 
                          ground_masks: torch.Tensor,
                          ground_refers: torch.Tensor,
                          pred_images: torch.Tensor,
                          pred_masks: torch.Tensor,
                          epoch: int, 
                          phase: str,
) -> None: 
    ground_img = inverse_normalize(ground_images[:1].detach().cpu())[0]
    ground_ref = inverse_normalize(ground_refers[:1].detach().cpu())[0]
    ground_msk = ground_masks[0].cpu()
    pred_img = inverse_normalize(pred_images[:1].detach().cpu())[0]
    pred_msk = torch.ge(torch.sigmoid(pred_masks[0].detach().cpu()), 0.5).float()
    
    fix, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    ground_img_msk = utils.overlay_mask(ground_img, ground_msk)
    axes[0, 0].imshow(ground_img_msk.numpy().transpose(1, 2, 0))
    axes[0, 0].set_title("GIGM")
    axes[0, 0].axis("off")
    
    ground_img_msk = utils.overlay_mask(ground_img, pred_msk)
    axes[0, 1].imshow(ground_img_msk.numpy().transpose(1, 2, 0))
    axes[0, 1].set_title("GIPM")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(ground_ref.numpy().transpose(1, 2, 0))
    axes[1, 0].set_title("RI")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(pred_img.numpy().transpose(1, 2, 0))
    axes[1, 1].set_title("PI")
    axes[1, 1].axis("off")
    
    plt.savefig(f"{out_dir}/{phase}Pred_{epoch:0>3}.png", bbox_inches='tight', dpi=300) 
    
def loss_visualizer(out_dir: str,
                    epochs: int,
                    train_rec_losses: List, 
                    train_seg_losses: List, 
                    valid_rec_losses: List, 
                    valid_seg_losses: List,                   
) -> None: 
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_rec_losses, color="red", label="RecTrain", linestyle='-', marker='o', linewidth=2.0)
    plt.plot(range(epochs), train_seg_losses, color="orange", label="SegTrain", linestyle='--', marker='s', linewidth=2.0)
    plt.plot(range(epochs), valid_rec_losses, color="blue", label="RecValid", linestyle='-', marker='o', linewidth=2.0)
    plt.plot(range(epochs), valid_seg_losses, color="green", label="SegValid", linestyle='--', marker='s', linewidth=2.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_facecolor('#f5f5f5')
    plt.title("Training and Validation Losses", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/losses.png", bbox_inches='tight')

def save_records(out_dir: str, 
                 records_df: pd.DataFrame
) -> None:
    records_df.to_csv(os.path.join(out_dir, "records.csv"), index=False)

def save_model(model: nn.Module,
               out_dir: str
) -> None:
    checkpoint_dir = os.path.join(out_dir, "weights")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if hasattr(model, "module"): 
        if hasattr(model, "_orig_mod"):
            model_stats = model._orig_mod.module.state_dict()
        else:
            model_stats = model.module.state_dict() 
    elif hasattr(model, "_orig_mod"):
        model_stats = model._orig_mod.state_dict()
    else: 
        model_stats = model.state_dict()
    
    torch.save(
        model_stats,
        os.path.join(checkpoint_dir, f"ckpt_best.tar")
    )
    print(f"Model saved in {checkpoint_dir} directory.")

def step_train(model: nn.Module, 
               optimizer: Callable,
               loss_fn: Dict[str, Callable], 
               configs: Dict,
               rloss_weight: float,
               sloss_weight: float,
               ground_query_images: torch.Tensor,
               ground_query_masks: torch.Tensor,
               ground_reference_images: torch.Tensor      
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.train()
    optimizer.zero_grad()
    with torch.autocast(device_type=ground_query_images.device.type, dtype=torch.bfloat16):
        predicted_query_images, predicted_query_masks = model(ground_reference_images)

        rloss, sloss = torch.tensor(0.0), torch.tensor(0.0)
        if not configs.ModelConfig.FREEZE_REC_HEAD:
            rloss = loss_fn["RECONSTRUCTION"](output=predicted_query_images, target=ground_query_images)
        if not configs.ModelConfig.FREEZE_SEG_HEAD:
            sloss = loss_fn["SEGMENTATION"](output=predicted_query_masks, target=ground_query_masks)
        gloss = rloss_weight * rloss + sloss_weight * sloss
    
    gloss.backward()
    optimizer.step()
        
    return rloss.item(), sloss.item(), predicted_query_images, predicted_query_masks

@torch.no_grad()
def step_evaluate(model: nn.Module,
                  loss_fn: Dict[str, Callable], 
                  configs: Dict,
                  ground_query_images: torch.Tensor,
                  ground_query_masks: torch.Tensor,
                  ground_reference_images: torch.Tensor
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.autocast(device_type=ground_query_images.device.type, dtype=torch.bfloat16):
        predicted_query_images, predicted_query_masks = model(ground_reference_images)
                
        rloss, sloss = torch.tensor(0.0), torch.tensor(0.0)
        if not configs.ModelConfig.FREEZE_REC_HEAD:
            rloss = loss_fn["RECONSTRUCTION"](output=predicted_query_images, target=ground_query_images)
        if not configs.ModelConfig.FREEZE_SEG_HEAD:
            sloss = loss_fn["SEGMENTATION"](output=predicted_query_masks, target=ground_query_masks)
    
    return rloss.item(), sloss.item(), predicted_query_images, predicted_query_masks

def run_train(step_train_fn: Callable, 
              step_eval_fn: Callable, 
              train_loader: DataLoader, 
              valid_loader: DataLoader, 
              diffuser: SimpleDiffusion,
              score_fn: Dict[str, Callable], 
              scheduler: Callable, 
              epochs: int, 
              device: Union[int, str, torch.device], 
              is_ddp: bool,
              visualize_predictions: Callable, 
              visualize_losses: Callable, 
              log_records: Callable, 
              save_best_model: Callable
) -> None: 
    is_main_process = (not is_ddp) or (device == 0) 
    disable_tqdm = is_ddp and device != 0
    
    loss_recorder = utils.MeanMetric(scalar=False)
    score_recorder = utils.MeanMetric(scalar=False)
    records_df = pd.DataFrame({
        "Epoch": [], 
        "TrainLoss": [], 
        "ValidLoss": [], 
        "TrainDice": [], 
        "ValidDice": [], 
        "TrainIoU": [],
        "ValidIoU": []
    })
    
    train_rec_losses, train_seg_losses = [], []
    valid_rec_losses, valid_seg_losses = [], []    
    best_dice = 0.0
    for epoch in range(epochs):
        if is_main_process: 
            print(f"{epoch + 1}/{epochs}".center(50, '-'))
            loss_recorder.reset()
            score_recorder.reset()         
        if isinstance(device, int): dist.barrier()
        for (sq, sr) in tqdm(train_loader, total=len(train_loader), ncols=100, desc="Train", disable=disable_tqdm):
            ground_query_images = sq[2].to(device)
            ground_query_masks = sq[3].to(device)
            ground_reference_images = [diffuser.diffuse(ref.to(device)) for ref in sr[2]]
            
            rloss, sloss, predicted_query_images, predicted_query_masks = step_train_fn(
                ground_query_images, ground_query_masks, ground_reference_images
            )
            
            loss_recorder.update({"RLoss": rloss, "SLoss": sloss})
            score_recorder.update(score_fn["SEGMENTATION"](output=predicted_query_masks, target=ground_query_masks))
        
        if is_main_process:
            if scheduler: scheduler.step()
            visualize_predictions(ground_query_images, ground_query_masks, 
                                  ground_reference_images[0], 
                                  predicted_query_images, predicted_query_masks, 
                                  epoch + 1, "Train")
            losses = loss_recorder.compute()
            train_epoch_scores = score_recorder.compute()
            train_rec_losses.append(losses["RLoss"])
            train_seg_losses.append(losses["SLoss"])
            loss_recorder.reset()
            score_recorder.reset()
        
        if isinstance(device, int): dist.barrier()
        for (sq, sr) in tqdm(valid_loader, total=len(valid_loader), ncols=100, desc="Valid", disable=disable_tqdm):
            ground_query_images = sq[2].to(device)
            ground_query_masks = sq[3].to(device) 
            ground_reference_images = [ref.to(device) for ref in sr[2]]

            rloss, sloss, predicted_query_images, predicted_query_masks = step_eval_fn(
                ground_query_images, ground_query_masks, ground_reference_images
            ) 
            
            loss_recorder.update({"RLoss": rloss, "SLoss": sloss})
            score_recorder.update(score_fn["SEGMENTATION"](output=predicted_query_masks, target=ground_query_masks))  
            
        if is_main_process: 
            losses = loss_recorder.compute()
            valid_epoch_scores = score_recorder.compute()
            valid_rec_losses.append(losses["RLoss"])
            valid_seg_losses.append(losses["SLoss"])
        
            print(f"Train:: RecLoss: {train_rec_losses[-1]:0.3f}, SegLoss: {train_seg_losses[-1]:0.3f}, Dice: {train_epoch_scores['dice']:0.2f}, IoU: {train_epoch_scores['iou']:0.2f}")
            print(f"Valid:: RecLoss: {valid_rec_losses[-1]:0.3f}, SegLoss: {valid_seg_losses[-1]:0.3f}, Dice: {valid_epoch_scores['dice']:0.2f}, IoU: {valid_epoch_scores['iou']:0.2f}")

            records_df.loc[epoch + 1] = [
                epoch, 
                train_rec_losses[-1] + train_seg_losses[-1], valid_rec_losses[-1] + valid_seg_losses[-1], 
                train_epoch_scores['dice'], valid_epoch_scores['dice'],
                train_epoch_scores['iou'], valid_epoch_scores['iou']
            ]
            visualize_predictions(ground_query_images, ground_query_masks, 
                                  ground_reference_images[-1], 
                                  predicted_query_images, predicted_query_masks, 
                                  epoch + 1, "Valid")
            
            if valid_epoch_scores["dice"] >= best_dice: 
                best_dice = valid_epoch_scores["dice"]
                save_best_model()
        
    if is_main_process:
        log_records(records_df)
        visualize_losses(train_rec_losses, train_seg_losses, valid_rec_losses, valid_seg_losses)

def run_test(step_eval_fn: Callable,  
             test_loader: DataLoader, 
             score_fn: Dict[str, Callable],  
             device: torch.device, 
             visualize_predictions: Callable, 
             log_records: Callable
) -> None:
    loss_recorder = utils.MeanMetric(scalar=False)
    score_recorder = utils.MeanMetric(scalar=False)
    step_index = 1     
    for (sq, sr) in tqdm(test_loader, total=len(test_loader), ncols=100, desc="Test"):
        ground_query_images = sq[2].to(device)
        ground_query_masks = sq[3].to(device)
        ground_reference_images = [ref.to(device) for ref in sr[2]]
        
        rloss, sloss, predicted_query_images, predicted_query_masks = step_eval_fn(ground_query_images, ground_query_masks, ground_reference_images)
        
        loss_recorder.update({"RLoss": rloss, "SLoss": sloss})
        score_recorder.update(score_fn["SEGMENTATION"](output=predicted_query_masks, target=ground_query_masks))
        
        visualize_predictions(
            ground_query_images, ground_query_masks, ground_reference_images[-1], 
            predicted_query_images, predicted_query_masks, step_index, "Test"
        )
        step_index += 1
        
    losses = loss_recorder.compute()
    scores = score_recorder.compute()
    
    log_records(
        pd.DataFrame({
            "RLoss": [losses['RLoss']], "SLoss": [losses['SLoss']],
            "Dice":  [scores['dice']],  "IoU":   [scores['iou']]
        })
    )
        
    print(f"Test:: RecLoss: {losses['RLoss']:0.3f}, SegLoss: {losses['SLoss']:0.3f}, "
          f"Dice: {scores['dice']:0.2f}, IoU: {scores['iou']:0.2f}"
    )

def get_metrics(rank: Union[int, str, torch.device], 
                configs: Dict
) -> Tuple[Dict[str, Callable], Dict[str, Callable]]: 
    # Criteria and Metrics.
    loss_fn = {
        "RECONSTRUCTION": utils.ReconstructionLoss(
            loss_names=configs.TrainConfig.Losses.RECONSTRUCTION_LOSSES,
            loss_weights=configs.TrainConfig.Losses.RECONSTRUCTION_LOSSES_WEIGHTS,
            alpha=configs.TrainConfig.Losses.RECONSTRUCTION_GDL_ALPHA,
            reduction=configs.TrainConfig.Losses.RECONSTRUCTION_GDL_REDUCTION,
            device=rank
        ),
        "SEGMENTATION": utils.SegmentationLoss(
            loss_names=configs.TrainConfig.Losses.SEGMENTATION_LOSSES,
            loss_weights=configs.TrainConfig.Losses.SEGMENTATION_LOSSES_WEIGHTS,
            sigmoid=configs.TrainConfig.Losses.SEGMENTATION_SIGMOID,
            threshold=configs.TrainConfig.Losses.SEGMENTATION_THRESHOLD,
            alpha=configs.TrainConfig.Losses.SEGMENTATION_ALPHA,
            beta=configs.TrainConfig.Losses.SEGMENTATION_BETA,
            gamma=configs.TrainConfig.Losses.SEGMENTATION_GAMMA,
            smooth=configs.TrainConfig.Losses.SEGMENTATION_SMOOTH,
            device=rank
        )
    }
    score_fn = {
        "RECONSTRUCTION": lambda p, g: 0.0,
        "SEGMENTATION": utils.SegmentationMetrics()
    }
    return loss_fn, score_fn

def training_setup(rank: int, 
                   world_size: int, 
                   configs: Dict
) -> None:
    print(f"Running {'DDP' if isinstance(rank, int) else 'MAIN'} process on {rank}.")
    if configs.TrainConfig.DISTRIBUTED:
        setup(rank, world_size)
    
    diffuser = SimpleDiffusion(
        num_steps=configs.TrainConfig.InputDiffusion.NUMSTEPS,
        time_steps=configs.TrainConfig.InputDiffusion.TIMESTEPS,
        beta_range=configs.TrainConfig.InputDiffusion.BETA_RANGE,
        patch_size=configs.TrainConfig.InputDiffusion.PATCH_SIZE,
        mp=configs.TrainConfig.InputDiffusion.MP,
        dp=configs.TrainConfig.InputDiffusion.DP, 
        apply=configs.TrainConfig.InputDiffusion.APPLY,
        device=rank
    )
    
    train_loader = get_dataloader(
        metadata_paths=configs.TrainConfig.TRAIN_METADATA_PATH.SEGMENTATION,
        segmentation=True,
        phase='TRAIN',
        configs=configs, 
        rank=rank
    )
    valid_loader = get_dataloader(
        metadata_paths=configs.TrainConfig.VALID_METADATA_PATH.SEGMENTATION,
        segmentation=True,
        phase='VALID',
        configs=configs, 
        rank=rank
    )

    model = get_model(rank, configs)
    if configs.TrainConfig.COMPILE:
        model = torch.compile(model)

    optimizer = getattr(torch.optim, configs.TrainConfig.Optimizer.NAME)(
        model.parameters(),
        **configs.TrainConfig.Optimizer.PARAMS
    )
    scheduler = None
    if configs.TrainConfig.Optimizer.SCHEDULER.APPLY: 
        scheduler = getattr(torch.optim.lr_scheduler, configs.TrainConfig.Optimizer.SCHEDULER.NAME)(
            optimizer, **configs.TrainConfig.Optimizer.SCHEDULER.PARAMS
        )
        
    # Get metrics 
    loss_fn, score_fn = get_metrics(rank, configs)
    
    inverse_normalize = utils.InverseNormalization(
        mean=configs.TrainConfig.TRANSFORM_MEAN,
        std=configs.TrainConfig.TRANSFORM_STD,
        dtype=torch.uint8
    )
    out_dir = os.path.join(configs.BasicConfig.ROOT_LOG_DIR, configs.BasicConfig.EXPERIMENT_NAME)
    vis_dir = os.path.join(out_dir, "Visualizations")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    visualize_predictions = partial(prediction_visualizer, inverse_normalize, vis_dir)
    visualize_losses = partial(loss_visualizer, out_dir, configs.TrainConfig.NUM_EPOCHS)
    log_records = partial(save_records, out_dir)
    save_best_model = partial(save_model, model, out_dir)
        
    step_train_fn = partial(step_train, model, optimizer, loss_fn, configs, 1.0, 1.0)
    step_eval_fn = partial(step_evaluate, model, loss_fn, configs)
    run_train(
        step_train_fn, step_eval_fn, 
        train_loader, valid_loader, diffuser, 
        score_fn, scheduler, configs.TrainConfig.NUM_EPOCHS, rank, 
        configs.TrainConfig.DISTRIBUTED,
        visualize_predictions, visualize_losses, log_records, save_best_model
    )
    
    if configs.TrainConfig.DISTRIBUTED:
        cleanup()
    print(f"Finished running the process on {rank}.")

def testing_setup(device: torch.device, 
                  configs: Dict                 
) -> None:
    print(f"Testing process on {device}.")
    
    test_loader = get_dataloader(
        metadata_paths=configs.TrainConfig.TEST_METADATA_PATH.SEGMENTATION,
        segmentation=True,
        phase='TEST',
        configs=configs, 
        rank=device
    )

    model = get_model(device, configs)
    
    loss_fn, score_fn = get_metrics(device, configs)
    
    inverse_normalize = utils.InverseNormalization(
        mean=configs.TrainConfig.TRANSFORM_MEAN,
        std=configs.TrainConfig.TRANSFORM_STD,
        dtype=torch.uint8
    )
    out_dir = os.path.join(configs.BasicConfig.ROOT_LOG_DIR, configs.BasicConfig.EXPERIMENT_NAME)
    vis_dir = os.path.join(out_dir, "Visualizations")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    visualize_predictions = partial(prediction_visualizer, inverse_normalize, vis_dir)
    log_records = partial(save_records, out_dir)
        
    step_eval_fn = partial(step_evaluate, model, loss_fn, configs)
    run_test(step_eval_fn, test_loader, score_fn, device,
             visualize_predictions, log_records)   
      
     
if __name__ == "__main__": 
    # Environmental Variables 
    torch.set_float32_matmul_precision("high")
       
    parser = argparse.ArgumentParser(
        description='Initializers parameters for running the experiments.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        default='configs/configs.yaml',
                        help='The string path of the config file.'
    )
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    configs = utils.ConfigLoader()(config_path=args.config_path)
    
    configs.BasicConfig.ROOT_LOG_DIR = os.path.join(configs.BasicConfig.ROOT_LOG_DIR, configs.BasicConfig.EXPERIMENT_NAME)
    if configs.BasicConfig.DEVELOPMENT_PHASE == "TRAIN":
        configs.BasicConfig.EXPERIMENT_NAME = f"{configs.BasicConfig.DEVELOPMENT_PHASE}_{timestamp}"
    elif configs.BasicConfig.DEVELOPMENT_PHASE == "TEST": 
        configs.BasicConfig.EXPERIMENT_NAME = f"{configs.BasicConfig.DEVELOPMENT_PHASE}_{configs.BasicConfig.TEST_EXPERIMENT_DIR}_{timestamp}"
    os.makedirs(os.path.join(configs.BasicConfig.ROOT_LOG_DIR, configs.BasicConfig.EXPERIMENT_NAME), exist_ok=True)
    
    # Backup code and config
    if configs.BasicConfig.DEVELOPMENT_PHASE == "TRAIN":
        codebase_dir = os.path.join(configs.BasicConfig.ROOT_LOG_DIR, configs.BasicConfig.EXPERIMENT_NAME, "CodeSnapshot/")
        os.makedirs(codebase_dir, exist_ok=True)
        items_to_copy = glob.glob("*.py") + [args.config_path]
        for item in items_to_copy: 
            shutil.copy(item, codebase_dir)
        shutil.copytree("model/", os.path.join(codebase_dir, "model/"), dirs_exist_ok=True)
        shutil.copytree("utilities/", os.path.join(codebase_dir, "utilities/"), dirs_exist_ok=True)
        
    # Set the seeds
    utils.set_seeds(
        configs.BasicConfig.SEED,
        configs.BasicConfig.DETERMINISTIC_CUDANN,
        configs.BasicConfig.BENCHMARK_CUDANN, 
        configs.BasicConfig.DETERMINISTIC_ALGORITHM
    )    
    
    start_time = datetime.now()
    
    if configs.BasicConfig.DEVELOPMENT_PHASE == "TEST":
        configs.TrainConfig.DISTRIBUTED = False
        testing_setup(
            device=torch.device(f"cuda:{configs.BasicConfig.DEFAULT_DEVICE_IDs[0]}"),
            configs=configs               
        )
    elif configs.TrainConfig.DISTRIBUTED:
        if isinstance(configs.BasicConfig.DEFAULT_DEVICE_IDs, List):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, configs.BasicConfig.DEFAULT_DEVICE_IDs)
            )
        world_size = len(configs.BasicConfig.DEFAULT_DEVICE_IDs)
        mp.spawn(
            training_setup,
            args=(world_size, configs),
            nprocs=world_size,
            join=True
        )
    else: 
        training_setup(
            rank=torch.device(f"cuda:{configs.BasicConfig.DEFAULT_DEVICE_IDs[0]}"), 
            world_size=1, 
            configs=configs
        )
    
    # Log the total duration of the training process 
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_seconds = int(elapsed_time.total_seconds())
    elapsed_minutes = elapsed_seconds // 60
    elapsed_hours = elapsed_seconds // 3600

    timing_info = f"""
    ==================================================
    Process Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
    Process End Time  : {end_time.strftime('%Y-%m-%d %H:%M:%S')}
    ----------------------------------------
    Elapsed Time: {elapsed_hours:2d}:{elapsed_minutes:2d}:{elapsed_seconds:2d}
    ==================================================
    """
    if configs.BasicConfig.DEVELOPMENT_PHASE == "TRAIN":
        with open(os.path.join(configs.BasicConfig.ROOT_LOG_DIR, configs.BasicConfig.EXPERIMENT_NAME, "timing.txt"), "w") as file:
            file.write(timing_info)
    print(timing_info)