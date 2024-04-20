import os
import csv
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_file = os.path.join(log_dir, 'logs.csv')
        self.write_headers()
    
    def write_headers(self):
        with open(self.log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 
                'train_loss', 'val_loss', 
                'bleu_notes_1', 'bleu_notes_2', 'bleu_notes_3', 'bleu_notes_4', 'bleu_notes_5', 'mmd_notes', 
                'bleu_durations_1', 'bleu_durations_2', 'bleu_durations_3', 'bleu_durations_4', 'bleu_durations_5', 'mmd_durations', 
                'bleu_gaps_1', 'bleu_gaps_2', 'bleu_gaps_3', 'bleu_gaps_4', 'bleu_gaps_5', 'mmd_gaps']
                )
            f.flush()

    def add_hparams(self, hparams):
        self.writer.add_hparams(hparams, {})

    def log_metric(self, name, value, step):
        self.writer.add_scalar(name, value, step)
    
    def log_results(self, results):
        with open(self.log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                results['epoch'],
                results['train_loss'],
                results['val_loss'],
                *results['notes']['bleu'],
                results['notes']['mmd'],
                *results['durations']['bleu'],
                results['durations']['mmd'],
                *results['gaps']['bleu'],
                results['gaps']['mmd']
            ])
            f.flush()