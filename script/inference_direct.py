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
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from rdkit import RDLogger

# python -u /hpc2hdd/home/yli106/smiles2mol/script/inference_direct.py --config_path /hpc2hdd/home/yli106/smiles2mol/config/qm9_default.yml --start 26 --end 50
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path of config', required=True)
    parser.add_argument('--start', type=int, default=1, help='start idx of test generation')
    parser.add_argument('--end', type=int, default=200, help='end idx of test generation')
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

    if config.model.type == 'llama2':
        end_id = 11056
        n_id = 13
    elif config.model.type == 'mistral':
        end_id = 21288
        n_id = 13
    elif config.model.type == 'llama3':
        end_id = 11424
        n_id = 198
        
    # load the dataset
    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)
    with open(os.path.join(load_path, config.data.test_set), 'rb') as f:
        raw_test_data = pickle.load(f)
    test_data = raw_test_data[args.start-1:args.end]
    print(f'generate {args.start} to {args.end}')   

    for i in tqdm(range(len(test_data))):
        # encode input text
        num_conf = len(test_data[i][3])
        if config.model.type == 'llama3':
            input_text = dataset.process_inst_llama3(test_data[i][0], test_data[i][1], test_data[i][2])
        else:
            input_text = dataset.process_inst(test_data[i][0], test_data[i][1], test_data[i][2])
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to('cuda')
        
        # define StoppingCriteria
        class CustomStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores):
                # check if the end signiture is generated
                if end_id in input_ids[0]:
                    return True 
                # check if the number of line breaks reaches certain length
                elif (input_ids[0] == n_id).sum().item() >= test_data[0][1]+test_data[0][2]+10:
                    return True
                else:
                    return False

        logger = RDLogger.logger()
        logger.setLevel(RDLogger.CRITICAL)
        generated_mol = []
        remain = config.inference.multiplier*num_conf
        batch_size= 5
        total_generate = 0
        all_fail = []
        while True:
            output_sequences = model.generate(
                input_ids=input_ids, 
                max_length=config.inference.max_length, 
                do_sample=config.inference.do_sample, 
                top_k=config.inference.top_k, 
                top_p=config.inference.top_p, 
                temperature=config.inference.temperature, 
                eos_token_id=tokenizer.eos_token_id,       
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=[CustomStoppingCriteria()], 
                num_return_sequences=batch_size
            )
            if  config.model.type == 'llama3':
                raw_generated_texts = [tokenizer.decode(sequence, skip_special_tokens=False) for sequence in output_sequences]
            else:
                raw_generated_texts = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output_sequences]
            
            total_generate+=batch_size
            fail = 0
            for j in range(batch_size):
                if config.model.type == 'llama3':
                 mol_block_text = dataset.get_mol_block_llama3(raw_generated_texts[j], test_data[i][0], test_data[i][1], test_data[i][2])
                else:
                    mol_block_text = dataset.get_mol_block(raw_generated_texts[j], test_data[i][0], test_data[i][1], test_data[i][2])
                
                with open('test.mol', 'w') as f:
                    f.write(mol_block_text)
                try:
                    gen_mol = Chem.MolFromMolFile('test.mol')
                    gen_mol = RemoveHs(gen_mol)
                    generated_mol.append(gen_mol)
                except:
                    fail+=1
            
            # print(mol_block_text)
            empty_cache()  # Clear graphics memory cache
            if fail==batch_size:
                all_fail.append(1)
            else:
                all_fail.append(0)
            last_ten = all_fail[-10:]
            if sum(last_ten) == 10:
                print('timeout')
                break
            remain = remain - batch_size + fail
            print(remain)
            if remain<=0:
                break

        print(f'valid:{100*len(generated_mol)/total_generate}%')
        test_data[i].append(generated_mol)

        
        # save as file
        with open(os.path.join(config.inference.save_path,'inference_d_%s_%s_%sto%s.pkl' % (config.model.type, config.model.size, args.start, args.end)), 'wb') as file:
            pickle.dump(test_data, file)