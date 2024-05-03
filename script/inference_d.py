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
from conf3d import dataset,utils
import yaml
from easydict import EasyDict
from torch.cuda import empty_cache
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from rdkit import RDLogger

# python -u /hpc2hdd/home/yli106/smiles2mol/script/inference_d.py --config_path /hpc2hdd/home/yli106/smiles2mol/config/qm9_default.yml
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
        
        bond_list = utils.get_diff_bond(test_data[i][3])
        atom_list = utils.get_atom(test_data[i][3][0])

        num_conf = len(test_data[i][3])
        print(num_conf)
        input_text = dataset.process_inst(test_data[i][0], test_data[i][1], test_data[i][2])
        input_text += ('\n').join(atom_list[:4])
        print(input_text)

        time = config.inference.multiplier*num_conf
        gen = 0
        raw_generated_texts = []
        pbar = tqdm(total=time, desc="Generating")

        while True:

            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            input_ids = input_ids.to('cuda')


            for j in range(test_data[i][1]): 
                class CustomStoppingCriteria(StoppingCriteria):
                    def __init__(self):
                        self.stopped_by_second_condition = False
                    def __call__(self, input_ids, scores):
                        indices = (input_ids[0] == 29889).nonzero(as_tuple=True)[0]
                        if len(indices) >= 5+3*j:
                            third_index = indices[4+3*j]
                            num_elements_after_third = input_ids[0].numel() - third_index.item() - 1
                            if num_elements_after_third >=4:
                                return True      
                            else:
                                return False
                        elif (input_ids[0] == 13).sum().item() >= test_data[i][1]+8:
                            self.stopped_by_second_condition = True  # Set the flag
                            return True
                        else:
                            return False
                stopping_criteria = CustomStoppingCriteria()
                output_sequences = model.generate(
                    input_ids=input_ids, 
                    max_length=config.inference.max_length, 
                    do_sample=config.inference.do_sample, 
                    top_k=config.inference.top_k, 
                    top_p=config.inference.top_p, 
                    temperature=config.inference.temperature, 
                    eos_token_id=tokenizer.eos_token_id, 
                    stopping_criteria=[stopping_criteria], 
                    num_return_sequences=1
                )   
                add_text = atom_list[4+j]+'\n'
                add_ids = tokenizer.encode(add_text, return_tensors='pt')
                add_ids = add_ids.to('cuda')
                input_ids = torch.cat((output_sequences, add_ids), dim=1)

            if not stopping_criteria.stopped_by_second_condition:
                raw_generated_texts.append(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                gen+=1
                pbar.update(1)
            
            if gen>=time:
                pbar.close()
                break

        bond_text = ('\n').join(bond_list[0][:])
        generated_texts = [x + bond_text for x in raw_generated_texts]
        gen_mol_list = []
        for generated_text in generated_texts:
            mol_block_text = dataset.get_mol_block(generated_text, test_data[i][0], test_data[i][1], test_data[i][2])
            with open('test.mol', 'w') as f:
                f.write(mol_block_text)
            try:
                gen_mol = Chem.MolFromMolFile('/hpc2hdd/home/yli106/smiles2mol/test.mol')
                gen_mol = RemoveHs(gen_mol)
                gen_mol_list.append(gen_mol)
            except:
                pass
        test_data[i].append(gen_mol_list)

        
        # save as file
        with open(os.path.join(config.inference.save_path,'inference_%s_%s.pkl' % (config.model.type, config.model.size)), 'wb') as file:
            pickle.dump(test_data, file)