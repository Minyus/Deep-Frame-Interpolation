from src.utils import VideoInterpTripletsDataset
from src.train import trainGAN
import torch
from torch.utils.data import DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = VideoInterpTripletsDataset('datasets')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
trainGAN(10, dataloader)