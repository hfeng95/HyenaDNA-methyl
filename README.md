# HyenaDNA-methyl

Original HyenaDNA repo with installation info here: https://github.com/HazyResearch/hyena-dna

HyenaDNA with code and config for methylation prediction.

Pretraining:
```
python -m train wandb=null experiment=plants_genomic dataset=plants_genomic dataset.max_length=6400 dataset.csv_file=/path/to/dataset.csv
```

Finetuning:
```
python -m train wandb=null experiment=methyl dataset=methyl train.pretrained_model_path=/path/to/model.ckpt dataset.csv_file=/path/to/dataset.csv
```

Evaluating:
```
python test.py wandb=null experiment=methyl dataset=methyl train.pretrained_model_path=/path/to/model.ckpt train.pretrained_model_state_hook._name_=null dataset.csv_file=/path/to/dataset.csv
```
