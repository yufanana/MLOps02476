"""
Program to load a processed dataset, train a model,
and save the trained model.

python3 mlops_yf/train_model.py
"""
import os
import hydra
import torch
# import wandb
from models.model import MyNeuralNet
from torch import optim, nn
import matplotlib as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


@hydra.main(config_path='config', config_name='train_conf.yaml')
def train(cfg):
    """Train a model on MNIST."""
    print('Training model...')

    # Initialize wandb
    # wandb_cfg = {
    #     'learning_rate': cfg.learning_rate,
    #     'batch_size': cfg.batch_size,
    #     'epochs': cfg.epochs,
    # }
    # wandb.init(project='mlops_yf', config=wandb_cfg)

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

    # Training loop
    preds, target = [],[]
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        running_loss = 0
        for batch in trainloader:
            images, labels = batch
            # Flatten each image in the batch into a 784 long vector
            images.resize_(images.size()[0], 784)   # batch_size, 784
            # Forward pass, calculate loss, backward pass, update weights
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds.append(log_ps.argmax(dim=1).numpy())
            target.append(labels.detach())

        print(
            'Epoch: {}/{}'.format(e + 1, epochs),
            'Training Loss: {:.3f}'.format(loss.item()),
        )
        # wandb.log({'loss': loss})

    # CML report
    report = classification_report(target, preds)
    with open('classification_report.txt', 'w') as out_file:
        out_file.write(report)
    cm = confusion_matrix(target, preds)
    cm_disp = ConfusionMatrixDisplay(cm)
    plt.savefig('confusion_matrix.png')

    # Save the model
    save_dir = hydra.utils.get_original_cwd()+'/models/'
    save_name = 'trained_model.pt'
    torch.save(model, save_dir + save_name)


if __name__ == '__main__':
    train()
