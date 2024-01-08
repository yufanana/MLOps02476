"""
Program to load a processed dataset, train a model,
and save the trained model.

python3 mlops_yf/train_model.py
"""
import argparse
import torch
from models.model import MyNeuralNet
from torch import optim, nn


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="learning rate to use for training")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size to use for training")
    parser.add_argument("--epochs", default=20, type=int,
                        help="number of epochs to use for training")

    args = parser.parse_args(argv)
    return args


def train(lr, batch_size, epochs):
    """Train a model on MNIST."""
    print("Training model...")
    print(f"lr: {lr}")

    # Create model, loss function, and optimizer
    model = MyNeuralNet()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Load training dataset into dataloader
    data_folder = "data/processed/"
    data_name = "corruptmnist"
    train_images = torch.load(
        data_folder + data_name + "/processed_train_images.pt")
    train_labels = torch.load(
        data_folder + data_name + "/processed_train_labels.pt")
    trainset = torch.utils.data.TensorDataset(train_images, train_labels)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False)

    # Training loop
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        running_loss = 0
        for batch in trainloader:
            images, labels = batch
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            # Forward pass, calculate loss, backward pass, update weights
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            "Epoch: {}/{}".format(e + 1, epochs),
            "Training Loss: {:.3f}".format(loss.item()),
        )
    # Save the model
    save_dir = "models/"
    save_name = "trained_model.pt"
    torch.save(model, save_dir + save_name)


def main():
    args = parse_args()
    train(args.lr, args.batch_size, args.epochs)


if __name__ == "__main__":
    main()
