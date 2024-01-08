"""
Program to load a processed dataset, train a model,
and save the trained model.

python3 mlops_yf/train_model.py
"""
import torch
import hydra
from models.model import MyNeuralNet
from torch import optim, nn


@hydra.main(config_path='config', config_name='train_conf.yaml')
def train(cfg):
    """Train a model on MNIST."""
    print('Training model...')

    # Get hyperparameters from config file
    lr = cfg.learning_rate
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    print('cfg: ', cfg)

    # Create model, loss function, and optimizer
    model = MyNeuralNet()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Load training dataset into dataloader
    data_folder = hydra.utils.get_original_cwd()+'/data/processed/'
    data_name = 'corruptmnist'
    train_images = torch.load(
        data_folder + data_name + '/processed_train_images.pt',
    )
    train_labels = torch.load(
        data_folder + data_name + '/processed_train_labels.pt',
    )
    trainset = torch.utils.data.TensorDataset(train_images, train_labels)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False,
    )


if __name__ == '__main__':
    train()
