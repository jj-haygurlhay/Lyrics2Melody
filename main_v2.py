import json
import yaml
import torch
from torch.utils.data import DataLoader
from models import MusicT5, MusicGPT2
from datasets import load_dataset, load_metric
from training import Trainer as CustomTrainer
import evaluate
# from pytorch_lightning import Trainer as PLTrainer
from transformers import (
    GPT2Tokenizer, 
    GPT2Config, 
    T5Tokenizer, 
    T5Config,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    GenerationConfig,
    set_seed
)
from dataloader import SongsDataset, SongsCollator
from utils.quantize import encode_note, encode_duration, encode_gap, MIDI_NOTES, DURATIONS, GAPS
import numpy as np
import nltk
from datetime import datetime

HYPS_FILE = './config/hyps.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def serialize_melody(midi_seq):
    """
    Serialize MIDI sequence to a string format for tokenization.
    Each note-duration-gap triplet is concatenated into a single string with separators.
    """
    serialized_seq = []
    for note, duration, gap in midi_seq:
        note_token = f"<note{encode_note(note)}>"
        duration_token = f"<duration{encode_duration(duration)}>"
        gap_token = f"<gap{encode_gap(gap)}>"
        serialized_seq.append(f"{note_token} {duration_token} {gap_token}")
        # serialized_seq.append(encode_note(note))
        # serialized_seq.append(encode_duration(duration))
        # serialized_seq.append(encode_gap(gap))
    return " ".join(serialized_seq)
    #return serialized_seq

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set seed
    set_seed(config['seed'])


    # Load model and tokenizer
    model_name = 't5-small'
    t5_config = T5Config.from_pretrained(model_name)
    gen_config = GenerationConfig.from_model_config(t5_config)
    #gen_config.max_length = config['data']['max_sequence_length']


    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.generation_config = gen_config
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    tokenizer.add_tokens([f'<note{i}>' for i in range(len(MIDI_NOTES))])
    tokenizer.add_tokens([f'<duration{i}>' for i in range(len(DURATIONS))])
    tokenizer.add_tokens([f'<gap{i}>' for i in range(len(GAPS))])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Create dataset and collator
    batch_size = config['training']['batch_size']

    #train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    dataset = load_dataset('csv', data_files={'train': 'data/new_dataset/train.csv', 'valid': 'data/new_dataset/valid.csv', 'test': 'data/new_dataset/test.csv'})

    def preprocess_function(examples):
        
        inputs = ['notes: ' + lyrics for lyrics in examples['lyrics']]
        targets = examples['midi_notes']
        results = tokenizer(inputs, truncation=True, max_length=config['data']['max_sequence_length'])
        target_texts = []

        for item in targets:
            midi_seq = json.loads(item)
            serialized_melody = serialize_melody(midi_seq)
            target_texts.append(serialized_melody)

        targets_encoding = tokenizer(target_texts, truncation=True, max_length=config['data']['max_sequence_length'])
        #print(targets_encoding['input_ids'])
        results['labels'] = targets_encoding['input_ids']
        #results['labels'] = target_texts
        return results

    dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # Create trainer
    now = datetime.now()
    out_dir = f'./runs/{now.strftime("%Y-%m-%d_%H-%M-%S")}'
    args = Seq2SeqTrainingArguments(
        out_dir,
        logging_dir= out_dir,
        optim='adafactor', # 'adam' or 'adafactor
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        save_strategy="epoch",
        learning_rate=2e-5,
        warmup_steps=0,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="tensorboard",
    )

    metric = evaluate.load('rouge')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)
        
        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    custom_trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        compute_metrics=compute_metrics
    )

    # Train model
    custom_trainer.train()
    custom_trainer.save_model()

if __name__ == "__main__":
    main()