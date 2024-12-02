# DuoAttention with VILA

1. Download and unzip the dataset: https://drive.google.com/file/d/1KOUzy07viQzpmpcBqydUA043VQZ4nmRv/view

2. Download the annotation json file: https://huggingface.co/datasets/videoniah/VNBench/tree/main

3. Clone this repo and setup the environment:

```
git clone https://github.com/20ChaituR/DuoAttention-VILA.git
cd DuoAttention-VILA
./environment_setup.sh
```

4. Run the evaluation:

```
python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --output-path "data/VILA1.5-3b-VNBench-results.json" \
    --anno-path "data/VNBench-annotations.json" \
    --video-dir "data"
```
