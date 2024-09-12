import h5py
import numpy as np
import json
import yaml
import torch
import torch.nn as nn
import lightning as L
import argparse
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# __________ DEFINE DATASET __________

class H5Dataset(Dataset):
    def __init__(self, event_config, options_config):
        self.h5_file = None
        try:
            # Extract dataset paths and inputs from the YAML file (event.yaml)
            dataset_path = event_config['DATASETS']['train']
            inputs_config = event_config['INPUTS']
            labels_key = event_config['LABELS']

            # Try to open the HDF5 file
            self.h5_file = h5py.File(dataset_path, 'r')

            # Extract and preprocess features
            self.features = []
            for group, variables in inputs_config.items():
                for var, preprocess in variables.items():
                    data = self.h5_file[f'INPUTS/{group}/{var}'][:]
                    # Apply preprocessing
                    if preprocess == "log_normalize":
                        data = np.log1p(data)
                        data = (data - data.min()) / (data.max() - data.min())
                    elif preprocess == "normalize":
                        data = (data - data.min()) / (data.max() - data.min())
                    # Append processed data
                    self.features.append(data)

            # Stack features into a single array
            self.features = np.hstack([f.reshape(len(f), -1) for f in self.features])

            # Extract labels
            self.labels = self.h5_file['LABELS/' + labels_key][:]
        
        except Exception as e:
            print(f"Error during data loading or preprocessing: {e}")
            self.close()  # Ensure the file is closed if an error occurs
            raise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def close(self):
        # Ensure the file is properly closed, even if an error occurred earlier
        if self.h5_file:
            try:
                self.h5_file.close()
            except Exception as e:
                print(f"Error closing HDF5 file: {e}")
    

# __________ DEFINE MODEL __________

class DNNClassifier(L.LightningModule):
    def __init__(self, dataset, options_config):
        super(DNNClassifier, self).__init__()

        # Extract model architecture from options_config
        hidden_dims = options_config['model']['hidden_dims']
        activation = options_config['model']['activation']
        dropout_rate = options_config['model']['dropout']

        # Optimizer settings from options_config
        self.learning_rate = options_config['training']['learning_rate']
        self.optimizer_type = options_config['training']['optimizer'] 

        # Calculate input and output dimensions from the dataset
        input_dim = dataset.features.shape[1]  # Number of features in the dataset
        output_dim = len(set(dataset.labels))  # Number of unique classes in the labels

        # Create the layers based on the architecture
        layers = []
        dims = [input_dim] + hidden_dims

        # Add hidden layers with activations and dropout
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(self.get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        if output_dim == 1:
            layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        else:
            layers.append(nn.Softmax(dim=1))  # Softmax for multi-class classification

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Choose the optimizer based on options_config
        if self.optimizer_type == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def get_activation(self, activation):
        # Return the appropriate activation function
        if activation == 'ReLU':
            return nn.ReLU()
        elif activation == 'Tanh':
            return nn.Tanh()
        elif activation == 'LeakyReLU':
            return nn.LeakyReLU()
        elif activation == 'GeLU':
            return nn.GeLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


# __________ TRAINING __________

def get_latest_version(base_dir='outputs/DNN/version_'):
    version = 0
    while os.path.exists(f'{base_dir}{version}'):
        version += 1
    return version    

def main(options_file, gpus=None):
    # Load the options.json file from the provided path
    with open(options_file, 'r') as json_file:
        options_config = json.load(json_file)

    # Load the event.yaml file from the path specified in options.json
    yaml_path = options_config['yaml_path']  # Get the path to event.yaml from options.json
    with open(yaml_path, 'r') as yaml_file:
        event_config = yaml.safe_load(yaml_file)

    # Setup device-agnostic code:
    if gpus is None:
       gpus = options_config['training']['gpus']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
      num_cuda_devices = torch.cuda.device_count()
      if gpus > num_cuda_devices:
        raise ValueError(f'There are only {num_cuda_devices} GPUs available, but requested {gpus}.')
      else:
        gpu_indices = list(range(gpus))
    else:
      gpu_indices = None

    # Create dataset using the event and options configuration
    train_dataset = H5Dataset(event_config, options_config)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=options_config["training"]["batch_size"], 
        shuffle=options_config["training"]["shuffle"],
        num_workers=options_config["training"]["num_workers"],
    )

    # Initialize model using the dataset for input/output dimension calculations
    model = DNNClassifier(train_dataset, options_config)

    # Determine the latest version folder for this run
    version = get_latest_version()
    versioned_folder = f'outputs/DNN/version_{version}/'

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',  # Metric to monitor
        dirpath=f'{versioned_folder}checkpoints/',  # Directory to save checkpoints
        filename='best-checkpoint-{epoch:02d}-{train_loss:.2f}',  # Checkpoint filename
        save_top_k=1,  # Save only the best model
        mode='min',  # Save the checkpoint with the minimum loss
    )

    # Initialize PyTorch Lightning Trainer
    trainer = L.Trainer(max_epochs=options_config['training']['epochs'],
                         devices=gpu_indices,
                         accelerator="auto",
                         callbacks=[checkpoint_callback],  # Add the checkpoint callback
                         log_every_n_steps=10,  # Log every 10 steps to TensorBoard
                         logger=TensorBoardLogger('outputs/', name=f'DNN')  # TensorBoard logging directory
    )

    # Train the model
    trainer.fit(model, train_loader)

    # Close the dataset file after use
    train_dataset.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a DNN model for hZZ signal vs. background classification')
    parser.add_argument('-of', type=str, help='Path to options.json file')
    parser.add_argument('--gpus', type=int, help='Override the number of GPUs to use')

    args = parser.parse_args()

    # Call main with parsed arguments
    main(options_file=args.of, gpus=args.gpus)