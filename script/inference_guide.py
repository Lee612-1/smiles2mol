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
import time
import pandas as pd
import pickle
import argparse
import sys
sys.path.append('/hpc2hdd/home/yli106/smiles2mol')
from conf3d import dataset,utils
import yaml
from easydict import EasyDict
from torch.cuda import empty_cache
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from rdkit import RDLogger

# python -u /hpc2hdd/home/yli106/smiles2mol/script/inference_guide.py --config_path /hpc2hdd/home/yli106/smiles2mol/config/qm9_default.yml --start 186 --end 200
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
    
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
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
        dot_id = 29889
        n_id = 13
        after_dot = 4
    elif config.model.type == 'mistral':
        dot_id = 28723
        n_id = 13
        after_dot = 4
    elif config.model.type == 'llama3':
        dot_id = 13
        n_id = 198
        after_dot = 2

    # load the dataset
    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)
    with open(os.path.join(load_path, config.data.test_set), 'rb') as f:
        raw_test_data = pickle.load(f)
    test_data = raw_test_data[args.start-1:args.end]
    print(f'generate {args.start} to {args.end}')

    for i in tqdm(range(len(test_data))):
        # encode input text
        
        bond_list = utils.get_diff_bond(test_data[i][3])
        atom_list = utils.get_atom(test_data[i][3][0])

        num_conf = len(test_data[i][3])
        # print(num_conf)
        if config.model.type == 'llama3':
            input_text = dataset.process_inst_llama3(test_data[i][0], test_data[i][1], test_data[i][2])
        else:
            input_text = dataset.process_inst(test_data[i][0], test_data[i][1], test_data[i][2])
        input_text += ('\n').join(atom_list[:4])
        # print(input_text)

        times = config.inference.multiplier*num_conf
        gen = 0
        raw_generated_texts = []
        pbar = tqdm(total=times, desc=f"Generating No.{args.start+i}")
        start_time = time.time()

        while True:

            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            input_ids = input_ids.to('cuda')


            for j in range(test_data[i][1]): 
                class CustomStoppingCriteria(StoppingCriteria):
                    def __init__(self):
                        self.stopped_by_second_condition = False
                    def __call__(self, input_ids, scores):
                        indices = (input_ids[0] == dot_id).nonzero(as_tuple=True)[0]
                        if len(indices) >= 5+3*j:
                            third_index = indices[4+3*j]
                            num_elements_after_third = input_ids[0].numel() - third_index.item() - 1
                            if num_elements_after_third >=after_dot:
                                return True      
                            else:
                                return False
                        elif (input_ids[0] == n_id).sum().item() >= test_data[i][1]+8:
                            self.stopped_by_second_condition = True  # Set the flag
                            return True
                        else:
                            return False
                stopping_criteria = CustomStoppingCriteria()
                output_sequences = model.generate(
                    input_ids=input_ids, 
                    max_new_tokens=config.inference.max_new_tokens, 
                    do_sample=config.inference.do_sample, 
                    top_k=config.inference.top_k, 
                    top_p=config.inference.top_p, 
                    temperature=config.inference.temperature, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=[stopping_criteria], 
                    num_return_sequences=1
                )   
                if config.model.type == 'llama3':
                    add_text = ' '+atom_list[4+j]+'\n'
                else:
                    add_text = atom_list[4+j]+'\n'
                add_ids = tokenizer.encode(add_text, return_tensors='pt')
                add_ids = add_ids.to('cuda')
                input_ids = torch.cat((output_sequences, add_ids), dim=1)

            if not stopping_criteria.stopped_by_second_condition:
                if  config.model.type == 'llama3':
                    raw_generated_texts.append(tokenizer.decode(input_ids[0], skip_special_tokens=False))
                else:
                    raw_generated_texts.append(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                gen+=1
                pbar.update(1)
            
            if gen>=times:
                pbar.close()
                break

            end_time = time.time()
            if end_time - start_time >= 100*times:
                pbar.close()
                break

        bond_text = ('\n').join(bond_list[0][:])
        generated_texts = [x + bond_text for x in raw_generated_texts]
        gen_mol_list = []
        for generated_text in generated_texts:
            if config.model.type == 'llama3':
                 mol_block_text = dataset.get_mol_block_llama3(generated_text, test_data[i][0], test_data[i][1], test_data[i][2])
            else:
                mol_block_text = dataset.get_mol_block(generated_text, test_data[i][0], test_data[i][1], test_data[i][2])
            with open('/hpc2hdd/home/yli106/smiles2mol/test.mol', 'w') as f:
                f.write(mol_block_text)
            try:
                gen_mol = Chem.MolFromMolFile('/hpc2hdd/home/yli106/smiles2mol/test.mol')
                gen_mol = RemoveHs(gen_mol)
                gen_mol_list.append(gen_mol)
            except:
                pass
        test_data[i].append(gen_mol_list)

        
        # save as file
        with open(os.path.join(config.inference.save_path,'inference_%s_%s_%sto%s.pkl' % (config.model.type, config.model.size, args.start, args.end)), 'wb') as file:
            pickle.dump(test_data, file)