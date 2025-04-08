# Chest X-Ray Report Generator

> This project is part of a task for the college where I study, so `task-parts` contains files that associated with that task, whishing that I would get the full mark ;). In general the base code doesn't have any special parts except that folder.

## Installation

After cloning the repository, install the required packages in a virtual environment.

Next, download the datasets and checkpoints, as describe below.

## Dataset

### IU X-Ray

1. Download the Chen et al. labels and the chest X-rays in png format for IU X-Ray from:

```
https://openi.nlm.nih.gov
```

2. Place the files into `dataset` folder, such that their paths are `dataset/reports` and `dataset/images`.

## Checkpoints

This approach uses `CheXNet`, and `DenseNet121` as a CNN Encoder model. By default the `CheXNet` pretrained weights are located in `weights` folder.

## Config

The model configurations for each task can be found in its `config.py` file.

## Training and Evaluation

### Training

Use the below command to train the model form a saved checkpoint or without a checkpoint.

```bash
python train.py
```

### Evaluation

The model performance measure is based of the `BLEU` metric.

> Feel free to change the performance measure metric in the `check_accuracy` method that is located in the `eval.py` file

Run the following command to calculate `BLEU` score.

```bash
python eval.py
```

