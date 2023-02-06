"""Multi-View Pixel2Mesh network training loop"""
"""Based on exercises training loops"""

from pathlib import Path

import torch
import numpy as np
import pickle
torch.cuda.empty_cache()
from src.utils.tools import construct_feed_dict
from src.model.multi_view_p2m import MVP2M
from src.model.vgg16 import VGG16P2M
from src.data.shapenet import ShapeNetRenderings
from src.utils.loss.losses import P2MLoss
from src.utils.logger import Logger

def train(model, perceptual_network, optimizer, train_dataloader, val_dataloader, device, config, feed_dict, last_epoch):
    if config["send_telegram_message"]:
        logger = Logger()
        logger.start()

    # Load the initial ellipsoid used for P2M
    ellipsoid = feed_dict['ellipsoid_feature_X'].to(device)
    lape_idx = feed_dict['lape_idx']
    edges = feed_dict['edges']
    faces_cpu = feed_dict['faces_triangle']

    lape_idx = [torch.as_tensor(l, dtype=torch.long).to(device) for l in lape_idx]
    edges = [torch.as_tensor(e, dtype=torch.long).to(device) for e in edges]
    faces = [torch.as_tensor(f, dtype=torch.long).to(device) for f in faces_cpu]
    
    # Initialize the loss function
    loss_func = P2MLoss(lape_idx, edges, faces).to(device)

    model.train()
    #perceptual_network.train()

    best_loss_val = np.inf

    # Keep track of running average of train loss for printing
    train_loss_running = 0.

    if config["send_telegram_message"]:
        logger.send_userly("MVP2M training starts.")
        logger.send_timely("...")
    for epoch in range(last_epoch, config['max_epochs'] + last_epoch):
        for batch_idx, batch in enumerate(train_dataloader):
            batch["cameras"] = batch["cameras"].to(device)
            batch["images"] = batch["images"].to(device)
            batch["points"] = batch["points"].to(device)
            batch["normals"] = batch["normals"].to(device)

            optimizer.zero_grad()
            
            with torch.no_grad():
                img_feat = perceptual_network(batch['images'])["semantic_feature"]
            
            out = model(ellipsoid, img_feat, batch['cameras'])

            loss = loss_func(out, batch)
            loss.backward()
            optimizer.step()
            
            # Logging
            # out = model(ellipsoid, img_feat, batch['cameras'])

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx
            print(iteration)

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
                train_loss_running = 0.

            # Validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                model.eval()
                # perceptual_network.eval()

                # Evaluation on entire validation set
                loss_val = 0.
                for batch_val in val_dataloader:
                    batch_val["cameras"] = batch_val["cameras"].to(device)
                    batch_val["images"] = batch_val["images"].to(device)
                    batch_val["points"] = batch_val["points"].to(device)
                    batch_val["normals"] = batch_val["normals"].to(device)

                    with torch.no_grad():
                        img_feat = perceptual_network(batch_val['images'])["semantic_feature"]
                        out = model(ellipsoid, img_feat, batch_val['cameras'])
                    loss_val += loss_func(out, batch_val)

                loss_val /= len(val_dataloader)
                if loss_val < best_loss_val:
                    # write output mesh result
                    output_mesh = out["pred_coord"][2].squeeze().cpu()
                    vert = np.hstack((np.full([output_mesh.shape[0], 1], 'v'), output_mesh))
                    face = np.hstack((np.full([faces_cpu[2].shape[0], 1], 'f'), faces_cpu[2] + 1))

                    mesh = np.vstack((vert, face))

                    # output_mesh = out["pred_coord"][2].squeeze().cpu()
                    # vert = np.hstack((np.full([output_mesh.shape[0], 1], 'v'), output_mesh))
                    # face = np.loadtxt('src/data/face3.obj', dtype='|S32')
                    # mesh = np.vstack((vert, face))

                    pred_path = 'src/data/mvp2m_prediction.obj'
                    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')
                    np.savetxt('src/data/coarse.xyz', output_mesh, fmt='%s', delimiter=' ')

                    print('=> save to {}'.format(pred_path))

                    torch.save({
                        'mvp2m' : model.state_dict(),
                        'perceptual_network': perceptual_network.state_dict(),
                        'adam': optimizer.state_dict(),
                        'epoch': epoch
                    }, f'src/runs/{config["experiment_name"]}/checkpoint.ckpt')
                    
                    best_loss_val = loss_val
                    if config["send_telegram_message"]:
                        logger.update_latest_timely(f"Checkpoint saved. [epoch: {epoch:03d}/batch: {batch_idx:05d}]\n  - val_loss: {loss_val:.6f}\n  - best_loss_val: {best_loss_val:.6f}")

                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')
                torch.cuda.empty_cache()
                model.train()
                #perceptual_network.train()


def main(config):
    """
    Function for training MultiViewDeformationNetwork on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    if config['is_overfit']:
        train_dataloader = ShapeNetRenderings("overfit")
        val_dataloader = ShapeNetRenderings("overfit")
    else:
        train_dataloader = ShapeNetRenderings("train")
        val_dataloader = ShapeNetRenderings("validation")

    pkl = pickle.load(open('src/data/iccv_p2mpp.dat', 'rb'))
    feed_dict=construct_feed_dict(pkl)
    perceptual_network = VGG16P2M(pretrained=True).requires_grad_(False)
    # print("AAAAAAAAAAAAAA", len(list(perceptual_network.parameters())))
    model = MVP2M(feed_dict['supports'],feed_dict['pool_idx'],2883,192,3)
    # for param in perceptual_network.parameters():
    #     param.requires_grad = False
    # params =list(model.parameters())
    params =list(model.parameters()) #+ list(perceptual_network.parameters())
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'], weight_decay=config["weight_decay"])
    last_epoch = 0

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        checkpoint = torch.load(config['resume_ckpt'])              
        #perceptual_network.load_state_dict(checkpoint['perceptual_network'])
        model.load_state_dict(checkpoint['mvp2m'])
        # optimizer.load_state_dict(checkpoint['adam'])
        last_epoch = checkpoint['epoch']

    model.to(device)
    perceptual_network.to(device)

    print("============================================")
    print("MVP2M #params:")
    print_number_of_params(model)
    print("============================================")

    print("============================================")
    print("perceptual_network #params:")
    print_number_of_params(perceptual_network)
    print("============================================")

    # Create folder for saving checkpoints
    Path(f'src/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, perceptual_network, optimizer, train_dataloader, val_dataloader, device, config, feed_dict, last_epoch)

def print_number_of_params(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()  if p.requires_grad)
    print("total: ", total_params, " trainable:", trainable_params)
