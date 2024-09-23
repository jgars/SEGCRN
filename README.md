# Self Explainable Graph Convolutional Recurrent Network for Spatio-Temporal Forecasting

This repository contains the implementation of *Self Explainable Graph Convolutional Recurrent Network (SEGCRN)* described in "Self Explainable Graph Convolutional Recurrent Network for Spatio-Temporal Forecasting".

## About

SEGCRN is a deep learning model for spatio-temporal forecasting that seeks to integrate explainability into its architecture. To this end, during its training it seeks to minimize the prediction error while minimizing the amount of information used, as well as indicating the relevance of the different values depending on the context. In this way, the analysis of the predictions of the model is facilitated, being able to find patterns that help to understand its behavior.

## Getting started

The code can be cloned by using the following command:

```bash
git clone https://github.com/jgars/SEGCRN.git
```

The repository includes the `.devcontainer` folder with the necessary configuration to be able to run in VSCode a development environment in Docker with all the necessary dependencies.

To run it outside the Docker environment, open a terminal in the folder where the source code is located and run:

```bash
export PYTHONPATH="${PYTHONPATH}:$PWD"
pip install -r requirements.txt
pip install torch==2.1.0
```

To perform the different actions, `client.py` is used, which is executed using the command:

```bash
python src/client.py
```

The different existing options can be viewed using `--help`:

```
Usage: client.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  evaluate  Evaluate the model on the indicated dataset.
  fidelity  Calculte the fidelity and infidelity of the model on the indicated
            dataset.
  train     Train the model on the indicated dataset.
```

It can also be used in the different commands:

```
Usage: client.py train [OPTIONS]

Options:
  --dataset TEXT        Sets the dataset to train on.
  --save-model BOOLEAN  Sets whether the weights of the model are saved.
  --help                Show this message and exit.
```

### Dataset

For training and validation of the model, the PeMSD3, PeMSD4, PeMSD7, PeMSD8 and PeMS-Bay datasets have been used, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1XAP5m3KzL22CemR8iX10vrMhIc3q5inW?usp=sharing).

The datasets must be placed in the path `data\traffic\`.

### Training

The model can be trained with the command:

```bash
python src/client.py train --dataset PEMSD4 --save-model True
```

The available datasets are *PEMSD3*, *PEMSD4*, *PEMSD7*, *PEMSD8* y *PEMS_BAY*.

### Validation

The accuracy of the models can be measured using the command:

```bash
python src/client.py evaluate --dataset PEMSD4
```

Fidelity and infidelity can be measured using the following command:

```bash
python src/client.py fidelity --dataset PEMSD4
```

The different savepoints used for the validation of the model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Wqi2Ycx0xnX_Q5aO5XR8FmEQZ7t4nhJp?usp=sharing).

## Citation

The paper is currently under review at [Machine Learning](https://link.springer.com/journal/10994).