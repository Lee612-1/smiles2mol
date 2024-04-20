import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    HfArgumentParser,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import re
from tqdm import tqdm
import pandas as pd
import pickle
import argparse
import sys
sys.path.append('/hpc2hdd/home/yli106/smiles2mol')
from conf3d import dataset
import yaml
from easydict import EasyDict
from torch.cuda import empty_cache

# python -u /hpc2hdd/home/yli106/smiles2mol/script/inference.py --config_path /hpc2hdd/home/yli106/smiles2mol/config/qm9_default.yml
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path of config', required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    config.training_arguments.learning_rate = float(config.training_arguments.learning_rate)
    
    
    # load the tokenizer
    model_path = os.path.join(config.model.base_path, '%s_%s' % (config.model.type, config.model.size))
    peft_path = config.model.peft_path #os.path.join(config.model.save_path, '%s_%s_%s' % (config.model.type, config.model.size, config.training_arguments.num_train_epochs))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token # Use the EOS token to pad shorter sequences
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # model set up
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map= "auto"
    )
    model = PeftModel.from_pretrained(model, peft_path)
    model = model.merge_and_unload()

    # load the dataset
    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)
    with open(os.path.join(load_path, config.data.test_set), 'rb') as f:
        test_data = pickle.load(f)
     
    for i in tqdm(range(len(test_data))):
        # encode input text
        num_conf = len(test_data[i][3])
        input_text = dataset.process_inst(test_data[i][0], test_data[i][1], test_data[i][2])
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to('cuda')
        
        # define StoppingCriteria
        class CustomStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores):
                # check if the end signiture is generated
                if 11056 in input_ids[0]:
                    return True 
                # check if the number of line breaks reaches certain length
                elif (input_ids[0] == 13).sum().item() >= test_data[0][1]+test_data[0][2]+10:
                    return True
                else:
                    return False

        generated_texts = []
        batch_size= 5
        for _ in range((config.inference.multiplier*num_conf) // batch_size):
            output_sequences = model.generate(
                input_ids=input_ids, 
                max_length=config.inference.max_length, 
                do_sample=config.inference.do_sample, 
                top_k=config.inference.top_k, 
                top_p=config.inference.top_p, 
                temperature=config.inference.temperature, 
                eos_token_id=tokenizer.eos_token_id, 
                stopping_criteria=[CustomStoppingCriteria()], 
                num_return_sequences=batch_size
            )
            generated_texts += [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output_sequences]
            empty_cache()  # Clear graphics memory cache

        # If num_conf is not an integer multiple of batch_size, process the remaining sequence
        remainder = (config.inference.multiplier*num_conf) // batch_size
        if remainder > 0:
            output_sequences = model.generate(
                input_ids=input_ids, 
                max_length=config.inference.max_length, 
                do_sample=config.inference.do_sample, 
                top_k=config.inference.top_k, 
                top_p=config.inference.top_p, 
                temperature=config.inference.temperature, 
                eos_token_id=tokenizer.eos_token_id, 
                stopping_criteria=[CustomStoppingCriteria()], 
                num_return_sequences=remainder
            )
            generated_texts += [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output_sequences]
        # generated_text = generated_text.replace(input_text+' ', '')
        test_data[i].append(generated_texts)

        
    # save as file
    with open(os.path.join(config.inference.save_path,'inference_%s_%s.pkl' % (config.model.type, config.model.size)), 'wb') as file:
        pickle.dump(test_data, file)