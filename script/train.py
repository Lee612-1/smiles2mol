import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import argparse
from easydict import EasyDict
import yaml

# python -u /hpc2hdd/home/yli106/smiles2mol/script/train.py --config_path /hpc2hdd/home/yli106/smiles2mol/config/qm9_default.yml
# python -u /hpc2hdd/home/yli106/smiles2mol/script/train.py --config_path /hpc2hdd/home/yli106/smiles2mol/config/qm9_test.yml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path of config', required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    config.training_arguments.learning_rate = float(config.training_arguments.learning_rate)

    # load the dataset
    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)
    dataset = load_dataset('csv', data_files={'train': os.path.join(load_path, config.data.train_set), 'test': os.path.join(load_path, config.data.val_set)})
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    print(train_dataset)
    print(test_dataset)

    # load the tokenizer
    model_path = os.path.join(config.model.base_path, '%s_%s' % (config.model.type, config.model.size))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token # Use the EOS token to pad shorter sequences
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    if config.finetune.bnb:
        bnb_config = BitsAndBytesConfig(**config.bnb_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map= "auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Set training parameters
    training_arguments = TrainingArguments(**config.training_arguments)

    # lora model setup
    if config.finetune.lora:
        peft_config = LoraConfig(**config.lora_config,
            target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "lm_head"
            ]
        )

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=2100,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field="text",
            max_seq_length=2100,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False
        )

    # Train model
    trainer.train()

    # Save trained model
    new_model_path = os.path.join(config.model.save_path, '%s_%s_%s' % (config.model.type, config.model.size, config.training_arguments.num_train_epochs))
    trainer.model.save_pretrained(new_model_path)