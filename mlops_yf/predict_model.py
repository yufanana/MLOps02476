"""
Takes a pre-trained model file and creates prediction for some data.
Users provides either a folder with raw images that gets loaded in or
a numpy or pickle file with already loaded images

python3 mlops_yf/predict_model.py models/trained_model.pt data/example_images.npy
"""
import argparse
import torch


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict using trained model')

    parser.add_argument('model', type=str, help='Path to trained model')
    parser.add_argument('data', type=str, help='Path to data to predict on')

    args = parser.parse_args(argv)
    return args


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples
        and d is the output dimension of the model
    """

    print('Predicting using trained model...')
    print(model)

    model = torch.load(model)

    # Make predictions
    y_preds = []
    with torch.no_grad():
        for batch in dataloader:
            y_pred = model.forward(batch)
            y_preds.append(y_pred)
    y_preds = torch.cat(y_preds, 0)

    # Alternative:
    # torch.cat([model(batch) for batch in dataloader], 0)

    return y_preds


def main():
    args = parse_args()
    dataset = args.data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    y_preds = predict(args.model, dataloader)


if __name__ == '__main__':
    main()
