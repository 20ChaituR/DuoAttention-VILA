# DuoAttention with VILA

1. Download and unzip the [dataset](https://drive.google.com/file/d/1KOUzy07viQzpmpcBqydUA043VQZ4nmRv/view). Add the unzipped folder to your Google Drive, and name the folder `VNBench`.

2. Download the [annotation json file](https://huggingface.co/datasets/videoniah/VNBench/tree/main). Add the json file to your Google Drive, and name it `VNBench-annotations.json`.

3. Copy this [colab notebook](https://colab.research.google.com/drive/1SCX6QiHwsIvnYsdxnRqF6EkLS7qAmVRf?usp=sharing) to your Google Drive.

4. Follow the instructions in the notebook to run the experiments.

# Running the Demos


## Running Baseline Demo:

Ensure that you have run: `./environment_setup.sh`

Then, in the VILA directory:
```
python -W ignore llava/eval/run_vila_demo.py
```

## Running DuoAttention Demo

Ensure that you have run: `./environment_setup_duo.sh`

Then,
```
python -W ignore llava/eval/run_vila_duo_demo.py
```