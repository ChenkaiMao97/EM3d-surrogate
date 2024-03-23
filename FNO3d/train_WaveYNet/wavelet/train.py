import os
import csv
import torch
import numpy as np
import argparse
from torch.utils.data import random_split, DataLoader
from dataset import SimulationDataset
from model import Model
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# log_dir = "./logs/3_block_3_layer"
# log_dir = "./logs/4_block_12_layer"
log_dir = "./logs/6_block_32_layer"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

argparser = argparse.ArgumentParser()
plt.rcParams.update({"font.size": 18, "font.family": "serif"})

# general training args
argparser.add_argument("--epoch", type=int, help="epoch number", default=100)
argparser.add_argument("--batch_size", type=int, help="batch size", default=24)
argparser.add_argument(
    "--data_folder", type=str, help="folder for the data", default="/scratch/groups/jonfan/UNet/data/data_generation_52_thick_8bar_Si/30k_new_wmin625"
)
argparser.add_argument("--start_lr", type=float, help="initial learning rate", default=5e-4)
argparser.add_argument("--end_lr", type=float, help="final learning rate", default=1e-4)
argparser.add_argument("--weight_decay", type=float, help="l2 regularization coeff", default=0.0)


def MAE_loss(a, b):
    return torch.mean(torch.abs(a - b)) / torch.mean(torch.abs(b))


def MSE_loss(a, b):
    return torch.mean((a - b) ** 2)


def loss_fn(a, b):
    return MSE_loss(a, b) + MAE_loss(a, b)


def train(args, train_ds, test_ds):
    # model = Model([3, 3, 3], [16, 32, 64])
    # model = Model([3, 3, 3, 3], [16, 32, 64, 32])
    model = Model([4, 6, 6, 6, 6, 4], [32, 64, 128, 128, 128, 64], [8, 16, 16, 16, 16, 16])

    # Specify the GPU devices to use
    device_ids = [3, 4, 5, 6]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.cuda(device_ids[0])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    train_info = []

    for step in range(args.epoch):
        model.train()
        train_loss = 0.0

        for idx, sample_batched in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{step+1}/{args.epoch}]"):
            optimizer.zero_grad(set_to_none=True)

            y_batch_train, yeex_batch_train, yeey_batch_train, yeez_batch_train = (
                sample_batched["field"].cuda(device_ids[0]),
                sample_batched["yeex"],
                sample_batched["yeey"],
                sample_batched["yeez"],
            )

            logits = model(yeex_batch_train, yeey_batch_train, yeez_batch_train)
            logits = logits.reshape(y_batch_train.shape)

            data_loss = loss_fn(logits, y_batch_train)
            train_loss += data_loss.item()

            data_loss.backward()
            optimizer.step()

        lr_scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        test_loss_mae = 0.0

        with torch.no_grad():
            for idx, sample_batched in tqdm(enumerate(test_loader), total=len(test_loader), desc=f" Testing Epoch [{step+1}/{args.epoch}]"):
                y_batch_test, yeex_batch_test, yeey_batch_test, yeez_batch_test = (
                    sample_batched["field"].cuda(device_ids[0]),
                    sample_batched["yeex"],
                    sample_batched["yeey"],
                    sample_batched["yeez"],
                )

                logits = model(yeex_batch_test, yeey_batch_test, yeez_batch_test)
                logits = logits.reshape(y_batch_test.shape)

                data_loss = loss_fn(logits, y_batch_test)
                data_loss_mae = MAE_loss(logits, y_batch_test)
                test_loss += data_loss.item()
                test_loss_mae += data_loss_mae.item()

                if idx == 0:
                    # Get the first data from the first test batch
                    y_data = y_batch_test[0].cpu().numpy()
                    logits_data = logits[0].cpu().numpy()

                    # Create a figure with subplots for 3D plots
                    fig = plt.figure(figsize=(16, 12))

                    # Define component labels
                    component_labels = ["Re[Ex]", "Im[Ex]", "Re[Ey]", "Im[Ey]", "Re[Ez]", "Im[Ez]"]
                    indices = [0, 1, 2, 3, 4, 5]

                    # Iterate over components
                    for i in range(6):
                        # Plot 3D visualization for ground truth
                        ax1 = fig.add_subplot(3, 4, i * 2 + 1, projection="3d")
                        ax1.set_facecolor("white")  # Set background color to white
                        j = indices[i]
                        x, y, z = np.meshgrid(np.arange(y_data.shape[0]), np.arange(y_data.shape[1]), np.arange(y_data.shape[2]))
                        ax1.scatter(
                            x, y, z, c=y_data[:, :, :, j].flatten(), cmap="coolwarm", marker="o", alpha=0.05, vmin=-2, vmax=2, edgecolors="none"
                        )
                        ax1.set_title(f"Ground Truth ({component_labels[j]})")
                        ax1.set_xlabel("x")
                        ax1.set_ylabel("y")
                        ax1.set_zlabel("z")
                        ax1.xaxis.pane.fill = False
                        ax1.yaxis.pane.fill = False
                        ax1.zaxis.pane.fill = False
                        ax1.xaxis.pane.set_edgecolor("w")
                        ax1.yaxis.pane.set_edgecolor("w")
                        ax1.zaxis.pane.set_edgecolor("w")

                        # Plot 3D visualization for model output
                        ax2 = fig.add_subplot(3, 4, i * 2 + 2, projection="3d")
                        x, y, z = np.meshgrid(np.arange(logits_data.shape[0]), np.arange(logits_data.shape[1]), np.arange(logits_data.shape[2]))
                        ax2.scatter(
                            x, y, z, c=logits_data[:, :, :, j].flatten(), cmap="coolwarm", marker="o", alpha=0.05, vmin=-2, vmax=2, edgecolors="none"
                        )
                        ax2.set_title(f"Model Output ({component_labels[j]})")
                        ax2.set_xlabel("x")
                        ax2.set_ylabel("y")
                        ax2.set_zlabel("z")
                        ax2.xaxis.pane.fill = False
                        ax2.yaxis.pane.fill = False
                        ax2.zaxis.pane.fill = False
                        ax2.xaxis.pane.set_edgecolor("w")
                        ax2.yaxis.pane.set_edgecolor("w")
                        ax2.zaxis.pane.set_edgecolor("w")

                    # Save the figure
                    plt.tight_layout()
                    plt.savefig(f"{log_dir}/epoch_{step+1}_test_data_visualization.png", dpi=200)
                    plt.close()

        test_loss /= len(test_loader)
        test_loss_mae /= len(test_loader)
        torch.save(model.state_dict(), f"{log_dir}/model_{step+1}.pt")

        print(f"Epoch [{step+1}/{args.epoch}], Train Loss: {train_loss:.4f}, Test Loss (MSE): {test_loss:.4f}, Test Loss (MAE): {test_loss_mae:.4f}")
        train_info.append([step + 1, train_loss, test_loss, test_loss_mae])

        with open(f"{log_dir}/train_info.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Test Loss (MSE)", "Test Loss (MAE)"])
            writer.writerows(train_info)


def main(args):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    ds = SimulationDataset("/media/lts0/chenkaim/3d_data/SR_aperiodic_TiO2_no_src_in_PML")
    means = ds.means
    train_ds, test_ds = random_split(ds, [int(0.9 * len(ds)), len(ds) - int(0.9 * len(ds))])

    train(args, train_ds, test_ds)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
