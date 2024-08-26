# Cyber Attack Detection using PyTorch

![Cyber Security](https://www.dhl.com/content/dam/dhl/global/core/images/smart-grid-thought-leadership-1375x504/csi-ltr6-cyber-security-trend.jpg)

This project involves designing and implementing a deep learning model to detect cyber threats in network traffic logs using the BETH dataset. The model is trained to identify anomalies that could indicate potential cyber attacks, contributing to enhanced cybersecurity measures.

## Project Overview

As a cybersecurity analyst, identifying and mitigating cyber threats is crucial. In this project, a neural network model is built using PyTorch to detect anomalies in network traffic data, specifically focusing on identifying suspicious events. The BETH dataset simulates real-world logs, providing a rich source of information for training and testing the model.

## Dataset

The BETH dataset includes various features extracted from network logs, with a target label `sus_label` indicating whether an event is malicious (1) or benign (0).

### Features

- **processId:** The unique identifier for the process that generated the event.
- **threadId:** ID for the thread spawning the log.
- **parentProcessId:** Label for the process spawning this log.
- **userId:** ID of the user spawning the log.
- **mountNamespace:** Mounting restrictions the process log works within.
- **argsNum:** Number of arguments passed to the event.
- **returnValue:** Value returned from the event log (usually 0).
- **sus_label:** Binary label indicating a suspicious event (1 is suspicious, 0 is not).

## Model Architecture

The neural network model is a simple feed-forward neural network built using PyTorch's `nn.Sequential`. The architecture consists of:

- Input Layer: Corresponding to the number of features in the dataset.
- Hidden Layers:
  - 64 neurons, ReLU activation, 30% dropout.
  - 32 neurons, ReLU activation.
- Output Layer: 1 neuron with Sigmoid activation for binary classification.

## Training

The model was trained using Binary Cross Entropy loss (`nn.BCELoss`) and the `Adam` optimizer. The training loop ran for 20 epochs, with the goal of achieving a validation accuracy of at least 60%.

### Training and Validation Accuracy

The model was evaluated using the validation set after each epoch. The best validation accuracy achieved was saved as an integer value.

## Results

- **Best Validation Accuracy:** `val_accuracy%`

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cyber-attack-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd cyber-attack-detection
   ```

## Usage

My recommendation is to do all the machine learning work on **Google Colab** or **Jupyter Notebook** for ease of use and accessibility. You can upload the dataset and notebooks directly and run the code in an interactive environment.

## Dependencies

- Python 3.x
- PyTorch
- Pandas
- Scikit-learn
- Torchmetrics (or Scikit-learn for accuracy calculation)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/KunalParkhade/detect-cyber-security-threat/blob/main/LICENSE) file for more details.

## Acknowledgments

- The creators of the  dataset.
- The [PyTorch](https://github.com/pytorch) community for providing an amazing deep learning framework.c
