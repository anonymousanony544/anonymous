import sys
import os
import random
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader, Sampler
from ldm_predic import *
from encoder_batch import *
from ddpm import DiffusionModel


def loaddata(data_name):
    path = data_name
    data = pd.read_csv(path, header=None)
    return data.values.reshape(-1, 63, 10, 10)


def normalize(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    if max_val - min_val == 0:
        return torch.zeros_like(data)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data


def process_d(speed, demand, inflow):
    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)
    speed_np = torch.clamp(speed, max=140)
    demand_np = torch.clamp(demand, max=demand_threshold)
    inflow_np = torch.clamp(inflow, max=inflow_threshold)
    x1 = normalize(demand_np)
    y1 = normalize(inflow_np)
    z1 = normalize(speed_np)
    temp2 = y1.unsqueeze(1)
    temp1 = x1.unsqueeze(1)
    temp3 = z1.unsqueeze(1)
    res_speed = temp3.reshape(-1, 63, 100).float()
    res_demand = temp1.reshape(-1, 63, 100).float()
    res_inflow = temp2.reshape(-1, 63, 100).float()
    return res_speed, res_demand, res_inflow


class MyDataset(Dataset):
    def __init__(self, data, edge):
        self.data = data
        self.edge = edge

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.edge[index]


def process_edges(edge):
    binary_edge = (edge != 0).float()
    return binary_edge

def load_pretrained_weights(diff_model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = diff_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    diff_model.load_state_dict(model_dict)

def generate_mask(batch_size, num_hours, mask_num):
    mask = torch.ones(batch_size, num_hours)
    zero_indices = torch.randperm(num_hours)[:mask_num]
    mask[:, zero_indices] = 0
    return mask


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # update the data path
    speed = loaddata('')
    inflow = loaddata('')
    demand = loaddata('')
    edge = np.load('')
    speed = torch.tensor(speed).to(device)
    inflow = torch.tensor(inflow).to(device)
    demand = torch.tensor(demand).to(device)
    speed, demand, inflow, min_d, max_d = process_d1(speed, demand, inflow)
    speed = inflow.reshape(-1, 12, 63, 100, 1)
    speed = speed[:, :, :50, :, :]
    edge = torch.tensor(edge).float()
    edge = process_edges(edge)
    edge_expanded = edge[np.newaxis, np.newaxis, :, :, :]
    edge_tiled = np.tile(edge_expanded, (162, 1, 1, 1))
    edge_tiled = torch.tensor(edge_tiled).to(device)
    edge_tiled = edge_tiled.reshape(162, 63, 100, 100)
    edge_tiled = edge_tiled[:, :50, :, :]
    batch = 32
    mask_num = 2
    dataset = MyDataset(speed, edge_tiled)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    in_channels = 1
    hidden_channels = 16
    out_channels = 1
    num_timesteps = 1000
    num_features = 1
    model = GCNEDModel(in_channels, hidden_channels, out_channels).to(device)
    model.eval()
    model.encoder.load_state_dict(torch.load(""))
    model.decoder.load_state_dict(torch.load(""))
    diff_model = DiffusionModelWithPredicDecoder(in_channels=1, num_timesteps=1000, device=device, num_features=1,
                                                 hidden_dim=64).to(device)
    load_pretrained_weights(diff_model, '')
    optimizer = optim.Adam(diff_model.parameters(), lr=0.0001)
    criterion_recon = nn.MSELoss()
    for epoch in range(1000):
        model.eval()
        diff_model.train()
        total_loss = 0
        for batch_idx, (data, edge) in enumerate(dataloader):
            batch_loss = 0
            data = data.to(device)
            edge = edge.to(device)
            optimizer.zero_grad()
            batch_size, num_hours, num_regions, num_nodes, num_feature = data.shape
            region_list = np.arange(num_regions)
            np.random.shuffle(region_list)
            for i in region_list:
                data_region = data[:, :, i, :, :]
                data_edge = edge[0, i, :, :]
                mask = generate_mask(batch_size, num_hours, mask_num).to(device)
                data_regionx = data_region * mask.unsqueeze(-1).unsqueeze(-1)
                edge_indices = data_edge.nonzero(as_tuple=False).t().contiguous().to(device)
                embed_data = model.encoder(data_regionx, edge_indices)
                recover_data = diff_model(embed_data, data_edge, mask)
                prediction = model.decoder(recover_data, edge_indices)

                loss = criterion_recon(masked_prediction, masked_data_region)
                loss.backward(retain_graph=True)
                batch_loss += loss.item()
            optimizer.step()
            total_loss += batch_loss
        loss = total_loss / (len(dataloader) * num_regions)
        if (epoch + 1) % 100 == 0:
            save_model(epoch + 1, diff_model, mask_num, num_timesteps, path="")
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')


def save_model(epoch, model, mask_num, time_step, path="model_checkpoint_mask{}_ts{}_epoch{}.pth"):
    torch.save(model.state_dict(), path.format(mask_num, time_step, epoch))


if __name__ == '__main__':
    main()
