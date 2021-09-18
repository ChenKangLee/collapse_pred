# 陳有蘭崩塌地預測

## Project Organization
The project is organized into the following directories:
- `preprocessing/` contains the preprocessing pipeline.
- `model/` contains the model and loss definitions (Pytorch `nn/modules` class.)
- `trainer/` contains the classes that defines various batch training process.
- `run/` contains the entry point to the training-prediction process.
- `utils/` contains the self defined pytorch `Dataset` class and helper functions.
- `visualization/` contains the function that prints or plots the model output.

## Usage

This section walks through the whole process of preprocessing data and training the model.

### 1. Preprocessing
The rainfall, geo and collapse data should be groupped by year into separate files. See the code for the naming scheme.

Change the path to the input raw data files `root`, and the path to the outputted preprocessed files `dest` (both in the `run()` function in `preprocess/preprocessor.py`). Call the class to run the preprocessing pipeline:

```shell
$: python3 -m preprocess.preprocessor
```

The ouput files have the name `year_[end year].npy`, with raindata, geodata, collapse data each occupying separate folders.

### 2. Training
With the preprocessed data, we can begin the training process. `trainer/supervised.py` defines the batched forward passing and back-prop.

The entry point of the training process is in `run/rnn.py`. Set the hyperparameters defined in the funciton `train_rnn()` and call the module to run the training process.

```shell
$: python3 -m run.rnn
```

The paths `path_processed`, `path_model`, `path_fig` specifies where to look for the preprocessed files, where to store the model weight and where to output the figures, respectively.

More description for each hyperparameters can be seen in the comments.