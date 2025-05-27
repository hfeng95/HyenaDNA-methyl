'''
adapted from hyenadna standalone
'''
import transformers
import torch
import hydra
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig,OmegaConf

import src.utils as utils
from huggingface_norun import HyenaDNAPreTrainedModel
from train import SequenceLightningModule,create_trainer
from src.dataloaders.datasets.methyl_dataset import MethylDataset
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

def train_loop(model,device,train_loader,optimizer,epoch,loss_fn,log_interval=10):
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output,target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx%log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()))

def infer_loop(model,device,test_loader):
    with torch.inference_mode():
        acc = 0.
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            n_match = torch.sum(pred==target.squeeze())
            acc += n_match.item()
        acc /= len(test_loader.dataset)
        print('Test Accuracy:',acc)

def train_hf():
    num_epochs = 5
    max_length = 3200
    use_padding = True
    padding = 'max_length'
    batch_size = 256
    learning_rate = 6e-4
    rc_aug = True
    add_eos = False
    weight_decay = 0.1
    
    model_name = 'amps_tomato_CG'
    model_dir = '/home/hfeng031/repos/hyena-dna/models'
    train_file = '/home/hfeng031/repos/hyena-dna/data/sample_train_tomato_CG.csv'
    test_file = '/home/hfeng031/repos/hyena-dna/data/sample_test_tomato_CG.csv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = HyenaDNAPreTrainedModel.from_pretrained(
            model_dir,
            model_name,
            download=False,
            device=device,
            use_head=True,
            n_classes=2
        )

    tokenizer = CharacterTokenizer(
            characters=['A','C','G','T','N'],
            model_max_length=max_length+2,
            add_special_tokens=False
        )

    ds_train = MethylDataset(
            train_file,
            max_sequence_length=max_length,
            tokenizer=tokenizer
        )

    ds_test = MethylDataset(
            test_file,
            max_sequence_length=max_length,
            tokenizer=tokenizer
        )

    train_loader = DataLoader(ds_train,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(ds_test,batch_size=batch_size,shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    model.to(device)
    model.train()

    # preliminary save
    torch.save({
            'epoch': -1,
            'model_state_dict': { 'model.'+k:v for k,v in model.state_dict().items() }
        },'./ft_prelim.pt')

    for epoch in range(num_epochs):
        model.train()
        train_loop(model,device,train_loader,optimizer,epoch,loss_fn,log_interval=100)
        model.eval()
        infer_loop(model,device,test_loader)
        optimizer.step()
        torch.save({
                'epoch': epoch,
                'state_dict': { 'model.'+k:v for k,v in model.state_dict().items() }
            },'./ft_out.pt')

def inference(config):
    
    model_name = 'amps_tomato_CG'
    max_length = 3200
    batch_size = 16
    padding = 'max_length'
    test_file = '/home/hfeng031/repos/hyena-dna/data/sample_test_tomato_CG.csv'
    model_dir = '/home/hfeng031/repos/hyena-dna/models'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load
        )

    tokenizer = CharacterTokenizer(
            characters=['A','C','G','T','N'],
            model_max_length=max_length+2,
            add_special_tokens=False
        )

    ds_test = MethylDataset(
            test_file,
            max_sequence_length=max_length,
            tokenizer=tokenizer
        )

    test_loader = DataLoader(ds_test,batch_size=batch_size,shuffle=False)

    trainer = create_trainer(config)
 
    model.to(device)
    model.eval()

    # infer_loop(model,device,test_loader)

    trainer.validate(model)

# standalone version
def inference_hf():

    model_name = 'amps_tomato_CG'
    max_length = 3200
    batch_size = 16
    padding = 'max_length'
    test_file = '/home/hfeng031/repos/hyena-dna/data/sample_test_tomato_CG.csv'
    model_dir = '/home/hfeng031/repos/hyena-dna/models'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = HyenaDNAPreTrainedModel.from_pretrained(
            model_dir,
            model_name,
            download=False,
            device=device,
            use_head=True,
            n_classes=2
        )

    tokenizer = CharacterTokenizer(
            characters=['A','C','G','T','N'],
            model_max_length=max_length+2,
            add_special_tokens=False
        )

    ds_test = MethylDataset(
            test_file,
            max_sequence_length=max_length,
            tokenizer=tokenizer
        )

    test_loader = DataLoader(ds_test,batch_size=batch_size,shuffle=False)

    model.to(device)
    model.eval()

    '''
    with torch.inference_mode():
        out_a = model(torch.LongTensor(tokenizer('ATCG')['input_ids']).to(device).unsqueeze(0))
        out_b = model(torch.LongTensor(tokenizer('ATCG')['input_ids']).to(device).unsqueeze(0))
        out_c = model(torch.LongTensor(tokenizer('ATCG')['input_ids']).to(device).unsqueeze(0))
        print(out_a.shape,out_b.shape,out_c.shape)
    exit()
    '''

    infer_loop(model,device,test_loader)

@hydra.main(config_path='configs',config_name='config.yaml')
def main(config: OmegaConf):
    config = utils.train.process_config(config)
    utils.train.print_config(config,resolve=True)
    inference(config)

if __name__ == '__main__':
    main()
    # inference_hf()      # huggingface
    # train_hf()

