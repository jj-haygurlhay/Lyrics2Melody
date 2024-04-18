import os
import numpy as np
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader
from dataloader.collator import SongsCollator
from dataloader.vocab import Lang
from transformers import (
    set_seed,
    get_linear_schedule_with_warmup
)
from dataloader import SongsDataset
from models import CustomModelRNN
from torch.optim import AdamW
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import csv

from project_utils.BLEUscore import bleu_score
from project_utils.mmd import Compute_MMD
from threading import Thread

HYPS_FILE = './config/hyps.yaml'
EOS_token = 1
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thread_list = []


def train_epoch(dataloader, model, encoder_optimizer, decoder_optimizer, 
                encoder_lr_scheduler, decoder_lr_scheduler,
                criterion, epoch, note_loss_weight, duration_loss_weight, gap_loss_weight):

    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for data in progress_bar:
        input_tensor = data['input_ids'].to(device)
        target_notes = data['labels']['notes'].to(device)
        target_durations = data['labels']['durations'].to(device)
        target_gaps = data['labels']['gaps'].to(device)

        target_tensor = torch.cat([target_notes.unsqueeze(-1), target_durations.unsqueeze(-1), target_gaps.unsqueeze(-1)], dim=-1)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _, _, _, _ = model(input_tensor, target_tensor)

        loss_notes = criterion(
            decoder_outputs_notes.view(-1, decoder_outputs_notes.size(-1)),
            target_notes.view(-1)
        )
        loss_durations = criterion( 
            decoder_outputs_durations.view(-1, decoder_outputs_durations.size(-1)),
            target_durations.view(-1)
        )
        loss_gaps = criterion(
            decoder_outputs_gaps.view(-1, decoder_outputs_gaps.size(-1)),
            target_gaps.view(-1)
        )
        loss = note_loss_weight * loss_notes + duration_loss_weight * loss_durations + gap_loss_weight * loss_gaps

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix({'Training Loss': total_loss / len(dataloader)})

    return total_loss / len(dataloader)

def get_bleu_scores(predicted, true, results, i, ngram):
    results[i] = bleu_score(predicted, true, max_n=ngram+1, weights=[1/(ngram+1)]*(ngram+1))

def log_results(results, csv_writer):
    csv_writer.writerow([
        results['epoch'],
        results['train_loss'],
        results['val_loss'],
        results['notes']['bleu'][0],
        results['notes']['bleu'][1],
        results['notes']['bleu'][2],
        results['notes']['bleu'][3],
        results['notes']['bleu'][4],
        results['notes']['mmd'],
        results['durations']['bleu'][0],
        results['durations']['bleu'][1],
        results['durations']['bleu'][2],
        results['durations']['bleu'][3],
        results['durations']['bleu'][4],
        results['durations']['mmd'],
        results['gaps']['bleu'][0],
        results['gaps']['bleu'][1],
        results['gaps']['bleu'][2],
        results['gaps']['bleu'][3],
        results['gaps']['bleu'][4],
        results['gaps']['mmd']
    ])

def evaluate_model(model, dataloader, criterion, note_loss_weight, duration_loss_weight, gap_loss_weight, print_predictions, generate_temp):
    with torch.no_grad():
        total_loss = 0
        true_notes = []
        true_durations = []
        true_gaps = []
        predicted_notes = []
        predicted_durations = []
        predicted_gaps = []
        progress_bar = tqdm(dataloader, desc=f"Validation: ")
        is_printed = False
        for data in progress_bar:
            input_tensor = data['input_ids'].to(device)
            target_notes = data['labels']['notes'].to(device)
            target_durations = data['labels']['durations'].to(device)
            target_gaps = data['labels']['gaps'].to(device)

            decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _, decoded_notes, decoded_durations, decoded_gaps = model(input_tensor, generate_temp=generate_temp)

            loss_notes = criterion(
                decoder_outputs_notes.view(-1, decoder_outputs_notes.size(-1)),
                target_notes.view(-1)
            )
            loss_durations = criterion( 
                decoder_outputs_durations.view(-1, decoder_outputs_durations.size(-1)),
                target_durations.view(-1)
            )
            loss_gaps = criterion(
                decoder_outputs_gaps.view(-1, decoder_outputs_gaps.size(-1)),
                target_gaps.view(-1)
            )
            loss = note_loss_weight * loss_notes + duration_loss_weight * loss_durations + gap_loss_weight * loss_gaps

            total_loss += loss.item()

            if not is_printed and print_predictions:
                print(f"\nInput: {data['input_ids'][0].cpu().numpy()}")
                print(f"\nTarget notes: {data['labels']['notes'][0].cpu().numpy()}")
                print(f"Predicted notes: {decoded_notes[0].cpu().numpy()}")
                print(f"\nTarget durations: {data['labels']['durations'][0].cpu().numpy()}")
                print(f"Predicted durations: {decoded_durations[0].cpu().numpy()}")
                print(f"\nTarget gaps: {data['labels']['gaps'][0].cpu().numpy()}")
                print(f"Predicted gaps: {decoded_gaps[0].cpu().numpy()}")
                is_printed = True
            progress_bar.set_postfix({'Validation Loss': total_loss / len(dataloader)})
            true_notes.extend(target_notes.cpu().numpy())
            true_durations.extend(target_durations.cpu().numpy())
            true_gaps.extend(target_gaps.cpu().numpy())
            predicted_notes.extend(decoded_notes.cpu().numpy())
            predicted_durations.extend(decoded_durations.cpu().numpy())
            predicted_gaps.extend(decoded_gaps.cpu().numpy())
        
        # Convert to numpy arrays
        predicted_notes = np.asarray(predicted_notes)
        true_notes = np.asarray(true_notes)
        predicted_durations = np.asarray(predicted_durations)
        true_durations = np.asarray(true_durations)
        predicted_gaps = np.asarray(predicted_gaps)
        true_gaps = np.asarray(true_gaps)

        return total_loss / len(dataloader), predicted_notes, predicted_durations, predicted_gaps, true_notes, true_durations, true_gaps

def log_scores(epoch, train_loss, val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps, csv_writer, csv_file=None):
    # Compute BLEU scores
    max_n = 2
    results = [None] * (3*max_n + 3)
    id = 0
    for i in range(max_n):
        thread = Thread(target=get_bleu_scores, args=(pred_notes, true_notes, results, id, i))
        thread.start()
        thread_list.append(thread)
        id += 1
    for i in range(max_n):
        thread = Thread(target=get_bleu_scores, args=(pred_durations, true_durations, results, id, i))
        thread.start()
        thread_list.append(thread)
        id += 1
    for i in range(max_n):
        thread = Thread(target=get_bleu_scores, args=(pred_gaps, true_gaps, results, id, i))
        thread.start()
        thread_list.append(thread)
        id += 1

    # Compute MMD
    thread = Thread(target=Compute_MMD, args=(pred_notes, true_notes, results, id))
    thread.start()
    thread_list.append(thread)
    id += 1
    thread = Thread(target=Compute_MMD, args=(pred_durations, true_durations, results, id))
    thread.start()
    thread_list.append(thread)
    id += 1
    thread = Thread(target=Compute_MMD, args=(pred_gaps, true_gaps, results, id))
    thread.start()
    thread_list.append(thread)


    # Retrieve results
    for thread in thread_list:
        thread.join()
    thread_list.clear()

    bleu_scores_notes = results[:max_n]
    bleu_scores_durations = results[max_n:2*max_n]
    bleu_scores_gaps = results[2*max_n:3*max_n]
    mmd_notes = results[3*max_n]
    mmd_durations = results[3*max_n + 1]
    mmd_gaps = results[3*max_n + 2]

    val_results = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'notes': {
            'bleu': bleu_scores_notes + (5 - max_n) * [0],
            'mmd': mmd_notes
        },
        'durations': {
            'bleu': bleu_scores_durations + (5 - max_n) * [0],
            'mmd': mmd_durations
        },
        'gaps': {
            'bleu': bleu_scores_gaps + (5 - max_n) * [0],
            'mmd': mmd_gaps
        },
    }
    print(val_results)
    log_results(val_results, csv_writer)
    csv_file.flush()

def train(train_dataloader, val_dataloader, model, n_epochs, learning_rate=0.001, weight_decay=0.01,
               note_loss_weight=0.8, duration_loss_weight=0.2, gap_loss_weight=0.2, output_folder='/log', print_predictions=False, generate_temp=1.0):
    train_losses, val_losses = [], []

    encoder_optimizer = AdamW(model.encoder.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
    decoder_optimizer = AdamW(model.decoder.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
    encoder_lr_scheduler = get_linear_schedule_with_warmup(encoder_optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * n_epochs)
    decoder_lr_scheduler = get_linear_schedule_with_warmup(decoder_optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * n_epochs)
    criterion = nn.CrossEntropyLoss()

    csv_file = os.path.join(output_folder, 'training_log.csv')
    # Keep track of lowest mmd
    lowest_mmd = float('inf')
    lowest_mmd_epoch = 0
    thread = None
    with open(csv_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(
            ['Epoch', 'Training Loss', 'Validation Loss', 
             'BLEU Notes 1', 'BLEU Notes 2', 'BLEU Notes 3', 'BLEU Notes 4', 'BLEU Notes 5', 'MMD Notes', 
             'BLEU Durations 1', 'BLEU Durations 2', 'BLEU Durations 3', 'BLEU Durations 4', 'BLEU Durations 5', 'MMD Durations', 
             'BLEU Gaps 1', 'BLEU Gaps 2', 'BLEU Gaps 3', 'BLEU Gaps 4', 'BLEU Gaps 5', 'MMD Gaps' ]
             )
        for epoch in range(1, n_epochs + 1):
            model.train()
            loss = train_epoch(train_dataloader, model, encoder_optimizer, decoder_optimizer, encoder_lr_scheduler, decoder_lr_scheduler, criterion, epoch, note_loss_weight, duration_loss_weight, gap_loss_weight)
            train_losses.append(loss)
            print(f"Epoch {epoch}, Training Loss: {loss}")

            model.eval()
            val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps = evaluate_model(model, val_dataloader, criterion, note_loss_weight, duration_loss_weight, gap_loss_weight, print_predictions, generate_temp)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")
            if thread is not None:
                thread.join()
            thread = Thread(target=log_scores, args=(epoch, loss, val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps, csv_writer, csv_file))
            thread.start()
            mmd_notes = Compute_MMD(pred_notes, true_notes)

            if mmd_notes < lowest_mmd:
                lowest_mmd = mmd_notes
                lowest_mmd_epoch = epoch
                torch.save(model.state_dict(), os.path.join(output_folder, f'model_best_mmd.pt'))
        thread.join()
        print(f"\nLowest MMD: {lowest_mmd} at epoch {lowest_mmd_epoch}")
    return train_losses, val_losses

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create output file
    out_dir = config['out_dir']
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(out_dir, run_dir)
    os.makedirs(output_folder, exist_ok=True)

    # Save config file
    with open(os.path.join(output_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set seed
    set_seed(config['seed'])

    
    # Create language objects
    syllables = Lang('syllables')
    lines = open('./vocab/syllables.txt', 'r', encoding='utf-8').read().strip().split('\n')
    for syllable in lines:
        syllables.addWord(syllable)

    model = CustomModelRNN(
        input_size=syllables.n_words,
        decoder_hidden_size=config['model']['decoder_hidden_size'],
        encoder_hidden_size=config['model']['encoder_hidden_size'],
        embedding_dim=config['model']['embedding_dim'], 
        SOS_token=0, 
        MAX_LENGTH=config['data']['max_sequence_length'], 
        dropout_p=config['model']['dropout'],
        num_layers=config['model']['num_layers'],
        device=device, 
        )

    batch_size = config['training']['batch_size']

        

    # Create dataset and collator
    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator(syllables_lang=syllables, output_eos=EOS_token, max_length=config['data']['max_sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True,  num_workers=0)

    train_loss, val_loss = train(
        train_loader, 
        val_loader, 
        model, 
        n_epochs=config['training']['epochs'], 
        learning_rate=config['training']['lr'], 
        weight_decay=config['training']['weight_decay'], 
        note_loss_weight=config['training']['note_loss_weight'],
        duration_loss_weight=config['training']['duration_loss_weight'],
        gap_loss_weight=config['training']['gap_loss_weight'],
        output_folder=output_folder,
        print_predictions=config['training']['print_predictions'],
        generate_temp=config['training']['generate_temp']
    )
    
    # Plot training and validation loss
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid()
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))

    # Save model
    torch.save(model.state_dict(), os.path.join(output_folder, 'model.pt'))

if __name__ == "__main__":
    main()