"""
Take the raw data e.g. the corrupted MNIST files located in a data/raw folder
and process them into a single tensor, normalize the tensor and save this
intermediate representation to the data/processed folder.

python3 mlops_yf/data/make_dataset.py
"""
import os
import torch
import os
import torch.nn.functional as F


def process_data(dataname):
    """
    Get the data from the raw data folder and process it into a single tensor.
    """
    raw_folder = 'data/raw/' + dataname + '/'
    proc_folder = 'data/processed/' + dataname + '/'

    # Load pt files
    dataset = []
    labels = []
    for i in range(5):
        print('Loading file: ', raw_folder+f'train_images_{i}.pt')
        print('Loading file: ', raw_folder+f'train_target_{i}.pt')
        dataset.append(torch.load(raw_folder+f'train_images_{i}.pt'))
        labels.append(torch.load(raw_folder+f'train_target_{i}.pt'))
    dataset = torch.cat(dataset, dim=0)
    dataset = F.normalize(dataset)
    labels = torch.cat(labels, dim=0)

    # Create directory if it does not exist
    os.makedirs(proc_folder, exist_ok=True)

    # Save tensor to processed folder
    torch.save(dataset, proc_folder + 'processed_train_images.pt')
    torch.save(labels, proc_folder + 'processed_train_labels.pt')
    print('Saved processed data to: ', proc_folder)


if __name__ == '__main__':
    # Get the data and process it
    dataname = 'corruptmnist'
    process_data(dataname)
