# 陳有蘭崩塌地預測

## Usage

### To run the preprocessor
Modify the path that points to the raw data and the processed data in `preprocess/preprocessor.py`

In the outmost directory (same level as this README), run:

```shell
python3 -m preprocess.preprocessor
```

### To Train The Model
* On new machines, run the preprocessor first !!

Modify the paths to point to the preprcessed files.

Then, in the outmost directory (same level as this README), run the following command:

```shell
python3 -m run.rnn
```