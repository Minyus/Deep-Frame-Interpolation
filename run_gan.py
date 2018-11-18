from src.utils import VideoInterpTripletsDataset
from src.train import trainGAN
from torch.utils.data import DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = VideoInterpTripletsDataset('datasets/wow', read_frames=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
trainGAN(10, dataloader)