import json
import yaml
import torch
from torch.utils.data import DataLoader
from models import MusicT5, MusicGPT2
from datasets import load_dataset
from training import Trainer as CustomTrainer
# from pytorch_lightning import Trainer as PLTrainer
from transformers import (
    GPT2Tokenizer, 
    GPT2Config, 
    T5Tokenizer, 
    T5Config,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
    set_seed
)
from dataloader import SongsDataset, SongsCollator
from utils.quantize import encode_note, encode_duration, encode_gap

HYPS_FILE = './config/hyps.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def serialize_melody(midi_seq):
    """
    Serialize MIDI sequence to a string format for tokenization.
    Each note-duration-gap triplet is concatenated into a single string with separators.
    """
    serialized_seq = []
    for note, duration, gap in midi_seq:
        note_token = f"note {encode_note(note)}"
        duration_token = f"duration {encode_duration(duration)}"
        gap_token = f"gap {encode_gap(gap)}"
        serialized_seq.append(f"{note_token} {duration_token} {gap_token}")
    return " ".join(serialized_seq)

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set seed
    set_seed(config['seed'])

    # pl_trainer = PLTrainer()

    # Load model and tokenizer
    model_name = config['model']['model_name']
    del config['model']['model_name']

    if 't5' in model_name:
        t5_config = T5Config.from_pretrained(
            't5-small'
            )
        # print(t5_config)
        model = MusicT5(t5_config, **config['model']) # TODO create config object
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path='t5-small')
    elif 'gpt' in model_name:
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        model = MusicGPT2(gpt2_config, **config['model']) # TODO create config object
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Model not implemented")

    model.to(device)

    # Create dataset and collator
    batch_size = config['training']['batch_size']

    #train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    train_dataset = load_dataset('csv', data_files={'train': 'data/new_dataset/train.csv', 'valid': 'data/new_dataset/valid.csv', 'test': 'data/new_dataset/test.csv'})
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator(tokenizer=tokenizer, output_eos=config['data']['output_eos'], max_length=config['data']['max_sequence_length'], use_syllables=config['data']['use_syllables'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4, persistent_workers=True)


    def preprocess_function(examples):
        
        inputs = examples['lyrics']
        targets = examples['midi_notes']
        results = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=config['data']['max_sequence_length'])
        target_texts = []

        for item in targets:
            
            midi_seq = json.loads(item)[:config['data']['max_sequence_length']]
            serialized_melody = serialize_melody(midi_seq)
            target_texts.append(serialized_melody)
        targets_encoding = tokenizer(target_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=config['data']['max_sequence_length'])

        results['labels'] = targets_encoding['input_ids']

        return results

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # pl_trainer.fit(model, tokenizer, train_loader, val_loader, test_loader)

    # Create trainer
    # custom_trainer = CustomTrainer(
    #     model=model, 
    #     pl_trainer=None,
    #     device=device,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     collator=collator,
    #     **config['training']
    # )



    args = Seq2SeqTrainingArguments(
        './runs',
        evaluation_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="steps",
        save_steps=50000,
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0,
        save_total_limit=0,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="tensorboard"
    )
    custom_trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['valid'],
    )

    # Train model
    custom_trainer.train()

    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()