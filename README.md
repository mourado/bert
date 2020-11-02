# BERT
BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding paper. NVIDIA's BERT is an optimized version of Google's official implementation, leveraging mixed precision arithmetic and Tensor Cores on A100, V100 and T4 GPUs for faster training times while maintaining target accuracy.

# Pretrain 
To pretrain or fine tune your model for Question Answering using mixed precision with Tensor Cores or using FP32/TF32, perform the following steps using the default parameters of the BERT model.

Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
```

# Build the BERT TensorFlow NGC container.
```
bash scripts/docker/build.sh
```

# Download and preprocess the dataset.
```
This repository provides scripts to download, verify and extract the SQuAD dataset, GLUE dataset and pretrained weights for fine tuning as well as Wikipedia and BookCorpus dataset for pre-training.
To download, verify, and extract the required datasets, run:
```
```
bash scripts/data_download.sh
```

```
The script launches a Docker container with the current directory mounted and downloads the datasets to a data/ folder on the host.

The launch.sh script assumes that the datasets are in the following locations by default after downloading the data.

SQuAD v1.1 - data/download/squad/v1.1

SQuAD v2.0 - data/download/squad/v2.0

GLUE The Corpus of Linguistic Acceptability (CoLA) - data/download/CoLA

GLUE Microsoft Research Paraphrase Corpus (MRPC) - data/download/MRPC

GLUE The Multi-Genre NLI Corpus (MNLI) - data/download/MNLI

BERT Large - data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16

BERT Base - data/download/google_pretrained_weights/uncased_L-12_H-768_A-12

BERT - data/download/google_pretrained_weights/

Wikipedia + BookCorpus TFRecords - data/tfrecords<config>/books_wiki_en_corpus

Start pre-training.
```

# Benchamrking 
```
The following section shows how to run benchmarks measuring the model performance in training and inference modes.

Both of these benchmarking scripts enable you to run a number of epochs, extract performance numbers, and run the BERT model for fine tuning.
```


# Training performance benchmark
Training benchmarking can be performed by running the script:
```
scripts/finetune_train_benchmark.sh <bert_model> <use_xla> <num_gpu> squad
```
## Example for 8 GPUs, Large Bert Model and True use of xla
```
scripts/finetune_train_benchmark.sh large true 8 squad
```

This script runs 2 epochs by default on the SQuAD v1.1 dataset and extracts performance numbers for various batch sizes and sequence lengths in both FP16 and FP32/TF32. These numbers are saved at
```
/results/squad_train_benchmark_bert_<bert_model>_gpu_<num_gpu>.log.
```

# Inference performance benchmark
Inference benchmarking can be performed by running the script:
```
scripts/finetune_inference_benchmark.sh squad
```
This script runs 1024 eval iterations by default on the SQuAD v1.1 dataset and extracts performance and latency numbers for various batch sizes and sequence lengths in both FP16 and FP32/TF32, for base and large models. These numbers are saved at /results/squad_inference_benchmark_bert_<bert_model>.log.

# Results
```
The following sections provide details on how we achieved our performance and accuracy in training and inference for pre-training using LAMB optimizer as well as fine tuning for Question Answering. All results are on BERT-large model unless otherwise mentioned. All fine tuning results are on SQuAD v1.1 using a sequence length of 384 unless otherwise mentioned.
```
