import os
import os.path
import numpy as np
import torch
import random
import csv
import pywt
from dataset import SimulationDataset
from dwt import DWTForward3d_Laplacian, DWTInverse3d_Laplacian
from tqdm import tqdm


wavelets = [
    "bior1.1",
    "bior1.3",
    "bior1.5",
    "bior2.2",
    "bior2.4",
    "bior2.6",
    "bior2.8",
    "bior3.1",
    "bior3.3",
    "bior3.5",
    "bior3.7",
    "bior3.9",
    "bior4.4",
    "bior5.5",
    "bior6.8",
    "coif1",
    "coif2",
    "coif3",
    "coif4",
    "coif5",
    "coif6",
    "coif7",
    "coif8",
    "coif9",
    "coif10",
    "coif11",
    "coif12",
    "coif13",
    "coif14",
    "coif15",
    "coif16",
    "coif17",
    "db1",
    "db2",
    "db3",
    "db4",
    "db5",
    "db6",
    "db7",
    "db8",
    "db9",
    "db10",
    "db11",
    "db12",
    "db13",
    "db14",
    "db15",
    "db16",
    "db17",
    "db18",
    "db19",
    "db20",
    "db21",
    "db22",
    "db23",
    "db24",
    "db25",
    "db26",
    "db27",
    "db28",
    "db29",
    "db30",
    "db31",
    "db32",
    "db33",
    "db34",
    "db35",
    "db36",
    "db37",
    "db38",
    "dmey",
    "haar",
    "rbio1.1",
    "rbio1.3",
    "rbio1.5",
    "rbio2.2",
    "rbio2.4",
    "rbio2.6",
    "rbio2.8",
    "rbio3.1",
    "rbio3.3",
    "rbio3.5",
    "rbio3.7",
    "rbio3.9",
    "rbio4.4",
    "rbio5.5",
    "rbio6.8",
    "sym2",
    "sym3",
    "sym4",
    "sym5",
    "sym6",
    "sym7",
    "sym8",
    "sym9",
    "sym10",
    "sym11",
    "sym12",
    "sym13",
    "sym14",
    "sym15",
    "sym16",
    "sym17",
    "sym18",
    "sym19",
    "sym20",
]


data_folder = "/media/lts0/chenkaim/3d_data/SR_aperiodic_TiO2_no_src_in_PML"
ds = SimulationDataset(data_folder)

mae_sums = {wavelet_type: 0.0 for wavelet_type in wavelets}
compression_rate_sums = {wavelet_type: 0.0 for wavelet_type in wavelets}
mae_sums_single_term = {wavelet_type: 0.0 for wavelet_type in wavelets}
compression_rate_sums_single_term = {wavelet_type: 0.0 for wavelet_type in wavelets}

for i in tqdm(range(len(ds))):
    sample = ds[i]
    field_data = sample["field"].cuda().permute(3, 0, 1, 2).unsqueeze(0)
    total_elements_original = field_data.numel()

    for wavelet_type in wavelets:
        wavelet = pywt.Wavelet(wavelet_type)
        max_depth = 4
        dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=max_depth, wave=wavelet, mode="zero").to("cuda")
        dwt_forward_3d_lap = DWTForward3d_Laplacian(J=max_depth, wave=wavelet, mode="zero").to("cuda")

        transformed = dwt_forward_3d_lap(field_data)

        # Calculate MAE and compression rate for the case when both transformed[1][0] and transformed[1][1] are set to None
        transformed_copy = [
            coeff.clone() if isinstance(coeff, torch.Tensor) else [c.clone() if c is not None else None for c in coeff] for coeff in transformed
        ]
        transformed_copy[1][0] = torch.zeros_like(transformed_copy[1][0])
        transformed_copy[1][1] = torch.zeros_like(transformed_copy[1][1])

        total_elements = 0
        for j in range(len(transformed_copy)):
            if isinstance(transformed_copy[j], torch.Tensor):
                total_elements += transformed_copy[j].numel()
            if isinstance(transformed_copy[j], list):
                for k in range(len(transformed_copy[j])):
                    if transformed_copy[j][k] is not None:
                        total_elements += transformed_copy[j][k].numel()

        reconstructed = dwt_inverse_3d_lap(transformed_copy)
        reconstructed_np = reconstructed

        mae = torch.mean(torch.abs(field_data - reconstructed_np)) / torch.mean(torch.abs(field_data))
        compression_rate = (total_elements - transformed_copy[1][0].numel() - transformed_copy[1][1].numel()) / total_elements_original

        mae_sums[wavelet_type] += mae.item()
        compression_rate_sums[wavelet_type] += compression_rate

        # Calculate MAE and compression rate for the case when only transformed[1][0] is set to None
        transformed_copy = [
            coeff.clone() if isinstance(coeff, torch.Tensor) else [c.clone() if c is not None else None for c in coeff] for coeff in transformed
        ]
        transformed_copy[1][0] = torch.zeros_like(transformed_copy[1][0])

        total_elements_single_term = 0
        for j in range(len(transformed_copy)):
            if isinstance(transformed_copy[j], torch.Tensor):
                total_elements_single_term += transformed_copy[j].numel()
            if isinstance(transformed_copy[j], list):
                for k in range(len(transformed_copy[j])):
                    if transformed_copy[j][k] is not None:
                        total_elements_single_term += transformed_copy[j][k].numel()

        reconstructed_single_term = dwt_inverse_3d_lap(transformed_copy)
        reconstructed_np_single_term = reconstructed_single_term

        mae_single_term = torch.mean(torch.abs(field_data - reconstructed_np_single_term)) / torch.mean(torch.abs(field_data))
        compression_rate_single_term = (total_elements_single_term - transformed_copy[1][0].numel()) / total_elements_original

        mae_sums_single_term[wavelet_type] += mae_single_term.item()
        compression_rate_sums_single_term[wavelet_type] += compression_rate_single_term

num_samples = len(ds)

mae_averages = {wavelet_type: mae_sum / num_samples for wavelet_type, mae_sum in mae_sums.items()}
compression_rate_averages = {wavelet_type: compression_rate_sum / num_samples for wavelet_type, compression_rate_sum in compression_rate_sums.items()}
mae_averages_single_term = {wavelet_type: mae_sum / num_samples for wavelet_type, mae_sum in mae_sums_single_term.items()}
compression_rate_averages_single_term = {
    wavelet_type: compression_rate_sum / num_samples for wavelet_type, compression_rate_sum in compression_rate_sums_single_term.items()
}

results = [
    (
        wavelet_type,
        mae_averages[wavelet_type],
        compression_rate_averages[wavelet_type],
        mae_averages_single_term[wavelet_type],
        compression_rate_averages_single_term[wavelet_type],
    )
    for wavelet_type in wavelets
]

sorted_results = sorted(results, key=lambda x: x[1])

csv_filename = "wavelet_results.csv"

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Wavelet", "Average MAE", "Average Compression Rate", "Average MAE (Single Term)", "Average Compression Rate (Single Term)"])
    writer.writerows(sorted_results)

print(f"Results saved to {csv_filename}")
