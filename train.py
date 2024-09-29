import tyro
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from typing import Sequence, Literal
import time
from pathlib import Path

from dataset import ShapeNetDataset
from point_e import (
    PointDiffusionTransformer, 
    GaussianDiffusion,
    get_named_beta_schedule,
    PointCloudSampler,
    ShuffledKeyTransformer,
)
import viser

@dataclass
class Args:
    num_epochs: int
    model: Literal['point-e', 'sk',]
    batch_size: int = 32
    num_points: int = 1024
    num_epochs_per_vis: int = 5
    num_epochs_per_loss_vis: int = 500
    window_size: int | None = None
    category: str = 'chair'
    lr: float = 1e-4
    cuda: int = 0
    port: int = 7777
    data_parallel: bool = True
    num_layers: int = 16
    width: int = 384

@torch.no_grad()
def visualize(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    args: Args,
    name: str,
    server: viser.ViserServer,
    color: tuple = (0, 0.7, 0),
    positions: np.ndarray | None = None,
    label: int | None = None,
):
    if positions is None:
        model.eval()
        device = torch.device(args.cuda)
        sampler = PointCloudSampler(
            device=device, 
            models=[model], 
            diffusions=[diffusion], 
            num_points=[args.num_points],
            aux_channels=[],
            guidance_scale=(2.0,),
        )
        if label is not None:
            label = torch.tensor([label], device=torch.device(args.cuda))
        positions = sampler.sample_batch(
            batch_size=1, 
            model_kwargs=dict(labels=label)
        ).transpose(1, 2).contiguous().cpu().numpy()[0]
    return server.scene.add_point_cloud(
        name=name,
        points=positions,
        colors=color,
        point_shape='circle',
        point_size=0.02,
    )

def train(args: Args):
    device = torch.device(args.cuda)
    categories = args.category.split(',')
    dataset = ShapeNetDataset(categories=categories, num_points=args.num_points, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.model == 'point-e':
        model = PointDiffusionTransformer(
            device=device,
            dtype=torch.float32,
            n_ctx=args.num_points,
        )
    elif args.model == 'sk':
        model = ShuffledKeyTransformer(
            device=device,
            dtype=torch.float32,
            n_ctx=args.num_points,
            window_size=args.window_size,
            layers=args.num_layers,
            width=args.width,
            num_classes=len(categories),
        )
    else:
        raise NotImplementedError(args.model)
    
    if args.data_parallel:
        num_cudas = torch.cuda.device_count()
        model = torch.nn.DataParallel(model, device_ids=list(range(num_cudas)))
        print(f'We use {num_cudas} GPUs.')
    
    diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule('cosine', 1024),
        model_mean_type='epsilon',
        model_var_type='learned_range',
        loss_type='mse',
        # channel_scales=np.array([2.0, 2.0, 2.0]),
        # channel_biases=np.array([0.0, 0.0, 0.0]),
        channel_scales=None,
        channel_biases=None,
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    server = viser.ViserServer(port=args.port, verbose=False)
    vis_lst = []

    for epoch in range(1, args.num_epochs + 1):
        losses = []
        with tqdm(total=len(dataloader), ascii=True, desc=f'Epoch {epoch:>5d}', leave=False) as tbar:
            for labels, gt_points in dataloader:
                model.train()
                loss = diffusion.training_losses(
                    model=model,
                    x_start=gt_points,
                    t=torch.randint(low=0, high=diffusion.num_timesteps, size=labels.shape, device=device),
                    model_kwargs={'labels': labels},
                )['loss'].mean()
                losses.append(loss.detach().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.update(1)
                tbar.set_description(f'Epoch {epoch:>5d}: Loss = {losses[-1]:.3f}')

        if epoch % args.num_epochs_per_loss_vis == 0:
            print(f'Epoch {epoch:>5d}: Loss = {np.mean(losses):.3f}')

        if epoch % 50 == 0:
            visualize(
                model, 
                diffusion, 
                args, 
                name=f'/history/{epoch}', 
                server=server, 
                color=(0, 0, 0.7),
                label=(epoch // 50 - 1) % len(categories)
            )
        elif epoch % args.num_epochs_per_vis == 0:
            vis_lst.append(visualize(
                model=model, 
                diffusion=diffusion, 
                args=args, 
                name=f'/gt/{epoch}', 
                server=server, 
                color=(0.7, 0, 0), 
                positions=gt_points[0].transpose(0, 1).contiguous().cpu().numpy()
            ))
            vis_lst.append(visualize(
                model, 
                diffusion, 
                args, 
                name=f'{epoch}', 
                server=server,
                label=(epoch // args.num_epochs_per_vis) % len(categories)
            ))
        if len(vis_lst) > 20:
            vis_lst.pop(0).remove()
            vis_lst.pop(0).remove()
    while(True):
        time.sleep(1)
            

if __name__ == '__main__':
    train(tyro.cli(Args))