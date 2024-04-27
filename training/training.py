from threading import Thread
import numpy as np
import torch
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch import nn
from tqdm.auto import tqdm
from training.evaluate import Evaluator
from training.logging import Logger

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, **hparams):
        self.model = model
        self.device = device
        self.out_dir = hparams['out_dir']
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.epochs = hparams["epochs"]
        self.curr_epoch = 0
        self.lr = float(hparams["lr"])
        self.weight_decay = float(hparams["weight_decay"])
        self.batch_size = hparams["batch_size"]

        num_training_steps = int(len(train_loader) * self.epochs)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        self.criterion = nn.CrossEntropyLoss()

        self.note_loss_weight     = hparams['note_loss_weight']
        self.duration_loss_weight = hparams['duration_loss_weight']
        self.gap_loss_weight      = hparams['gap_loss_weight']

        self.print_predictions = hparams['print_predictions']
        self.generate_temp = hparams['generate_temp']

        self.logger = Logger(self.out_dir)

        self.evaluator = Evaluator(self.logger)

    def train(self):
        lowest_mmd = 100
        lowest_mmd_epoch = 0
        for epoch in range(1, self.epochs + 1):
            self.curr_epoch = epoch
            loss = self.train_epoch()
            print(f"Epoch {epoch}, Training Loss: {loss}")

            val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps = self.validate_epoch()
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")
            
            mmd_notes = self.evaluator.evaluate_preds(epoch, loss, val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps)
            self.evaluator.retrieve_results()

            if mmd_notes < lowest_mmd:
                lowest_mmd = mmd_notes
                lowest_mmd_epoch = epoch
                self.save_model(name="best")
            self.save_model(name="latest")

        print(f"\nLowest MMD: {lowest_mmd} at epoch {lowest_mmd_epoch}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.curr_epoch}")
        for data in progress_bar:
            input_tensor = data['input_ids'].to(self.device)
            target_notes = data['labels']['notes'].to(self.device)
            target_durations = data['labels']['durations'].to(self.device)
            target_gaps = data['labels']['gaps'].to(self.device)

            target_tensor = torch.cat([target_notes.unsqueeze(-1), target_durations.unsqueeze(-1), target_gaps.unsqueeze(-1)], dim=-1)

            self.optimizer.zero_grad()
            
            decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _, _, _, _ = self.model(input_tensor, target_tensor)

            loss_notes = self.criterion(
                decoder_outputs_notes.view(-1, decoder_outputs_notes.size(-1)),
                target_notes.view(-1)
            )
            loss_durations = self.criterion( 
                decoder_outputs_durations.view(-1, decoder_outputs_durations.size(-1)),
                target_durations.view(-1)
            )
            loss_gaps = self.criterion(
                decoder_outputs_gaps.view(-1, decoder_outputs_gaps.size(-1)),
                target_gaps.view(-1)
            )
            loss = self.note_loss_weight * loss_notes + self.duration_loss_weight * loss_durations + self.gap_loss_weight * loss_gaps

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'Training Loss': total_loss / len(self.train_loader)})

        return total_loss / len(self.train_loader)


    def validate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            true_notes = []
            true_durations = []
            true_gaps = []
            predicted_notes = []
            predicted_durations = []
            predicted_gaps = []
            progress_bar = tqdm(self.val_loader, desc=f"Validation: ")
            is_printed = False
            for data in progress_bar:
                input_tensor = data['input_ids'].to(self.device)
                target_notes = data['labels']['notes'].to(self.device)
                target_durations = data['labels']['durations'].to(self.device)
                target_gaps = data['labels']['gaps'].to(self.device)

                decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _, decoded_notes, decoded_durations, decoded_gaps = self.model(input_tensor, generate_temp=self.generate_temp)

                loss_notes = self.criterion(
                    decoder_outputs_notes.view(-1, decoder_outputs_notes.size(-1)),
                    target_notes.view(-1)
                )
                loss_durations = self.criterion( 
                    decoder_outputs_durations.view(-1, decoder_outputs_durations.size(-1)),
                    target_durations.view(-1)
                )
                loss_gaps = self.criterion(
                    decoder_outputs_gaps.view(-1, decoder_outputs_gaps.size(-1)),
                    target_gaps.view(-1)
                )
                loss = self.note_loss_weight * loss_notes + self.duration_loss_weight * loss_durations + self.gap_loss_weight * loss_gaps

                total_loss += loss.item()

                if not is_printed and self.print_predictions:
                    print(f"\nInput: {data['input_ids'][0].cpu().numpy()}")
                    print(f"\nTarget notes: {data['labels']['notes'][0].cpu().numpy()}")
                    print(f"Predicted notes: {decoded_notes[0].cpu().numpy()}")
                    print(f"\nTarget durations: {data['labels']['durations'][0].cpu().numpy()}")
                    print(f"Predicted durations: {decoded_durations[0].cpu().numpy()}")
                    print(f"\nTarget gaps: {data['labels']['gaps'][0].cpu().numpy()}")
                    print(f"Predicted gaps: {decoded_gaps[0].cpu().numpy()}")
                    is_printed = True

                progress_bar.set_postfix({'Validation Loss': total_loss / len(self.val_loader)})
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

        return total_loss / len(self.val_loader), predicted_notes, predicted_durations, predicted_gaps, true_notes, true_durations, true_gaps

    def save_model(self, name=""):
        os.makedirs(self.out_dir, exist_ok=True)
        model_save_path = f"./{self.out_dir}/model_{name}.pt"
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Saved model checkpoint to {model_save_path}")
        
class TrainerTransformer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, **hparams):
        self.model = model
        self.device = device
        self.out_dir = hparams['out_dir']
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.epochs = hparams["epochs"]
        self.curr_epoch = 0
        self.lr = float(hparams["lr"])
        self.weight_decay = float(hparams["weight_decay"])
        self.batch_size = hparams["batch_size"]

        num_training_steps = int(len(train_loader) * self.epochs)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        self.criterion = nn.CrossEntropyLoss()

        self.note_loss_weight     = hparams['note_loss_weight']
        self.duration_loss_weight = hparams['duration_loss_weight']
        self.gap_loss_weight      = hparams['gap_loss_weight']

        self.print_predictions = hparams['print_predictions']
        self.generate_temp = hparams['generate_temp']

        self.logger = Logger(self.out_dir)

        self.evaluator = Evaluator(self.logger)

    def train(self):
        lowest_mmd = 100
        lowest_mmd_epoch = 0
        for epoch in range(1, self.epochs + 1):
            self.curr_epoch = epoch
            loss = self.train_epoch()
            print(f"Epoch {epoch}, Training Loss: {loss}")

            val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps = self.validate_epoch()
            
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")
            
            # remove SOS
            pred_notes = pred_notes[:, 1:]
            pred_durations = pred_durations[:, 1:]
            pred_gaps = pred_gaps[:, 1:]
            true_notes = true_notes[:, 1:]
            true_durations = true_durations[:, 1:]
            true_gaps = true_gaps[:, 1:]

            mmd_notes = self.evaluator.evaluate_preds(epoch, loss, val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps)
            self.evaluator.retrieve_results()

            if mmd_notes < lowest_mmd:
                lowest_mmd = mmd_notes
                lowest_mmd_epoch = epoch
                self.save_model(name="best")
            self.save_model(name="latest")

        print(f"\nLowest MMD: {lowest_mmd} at epoch {lowest_mmd_epoch}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.curr_epoch}")
        for data in progress_bar:
            input_tensor = data['input_ids'].to(self.device)
            attn_mask = data['attention_mask'].to(self.device)
            target_notes = data['labels']['notes'].to(self.device)
            target_durations = data['labels']['durations'].to(self.device)
            target_gaps = data['labels']['gaps'].to(self.device)
            
            decoder_input = torch.cat([target_notes.unsqueeze(-1),
                                   target_durations.unsqueeze(-1),
                                   target_gaps.unsqueeze(-1)
                                   ], dim=-1)

            self.optimizer.zero_grad()
            
            decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps = self.model(input_tensor, attn_mask, decoder_input)
            
            # # Loss calculated from the first token after SOS
            # target_notes = target_notes[:, 1:]
            # target_durations = target_durations[:, 1:]
            # target_gaps = target_gaps[:, 1:]
            
            # # Exclude the first output from the decoder corresponding to the prediction from SOS
            # decoder_outputs_notes = decoder_outputs_notes[:, 1:, :]
            # decoder_outputs_durations = decoder_outputs_durations[:, 1:, :]
            # decoder_outputs_gaps = decoder_outputs_gaps[:, 1:, :]
            
            loss_notes = self.criterion(decoder_outputs_notes.transpose(1, 2), target_notes)
            loss_durations = self.criterion(decoder_outputs_durations.transpose(1, 2), target_durations)
            loss_gaps = self.criterion(decoder_outputs_gaps.transpose(1, 2), target_gaps)
            loss = self.note_loss_weight * loss_notes + self.duration_loss_weight * loss_durations + self.gap_loss_weight * loss_gaps
            
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Training Loss': total_loss / len(self.train_loader)})

        return total_loss / len(self.train_loader)


    def validate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            true_notes = []
            true_durations = []
            true_gaps = []
            predicted_notes = []
            predicted_durations = []
            predicted_gaps = []
            progress_bar = tqdm(self.val_loader, desc=f"Validation: ")
            is_printed = False
            for data in progress_bar:
                input_tensor = data['input_ids'].to(self.device)
                attn_mask = data['attention_mask'].to(self.device)
                target_notes = data['labels']['notes'].to(self.device)
                target_durations = data['labels']['durations'].to(self.device)
                target_gaps = data['labels']['gaps'].to(self.device)

                decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, logits_notes, logits_durations, logits_gaps = self.model.generate(input_tensor, attn_mask, temperature=self.generate_temp)

                # # Loss calculated from the first token after SOS
                # target_notes = target_notes[:, 1:]
                # target_durations = target_durations[:, 1:]
                # target_gaps = target_gaps[:, 1:]
                # logits_notes = logits_notes[:, 1:]
                # logits_durations = logits_durations[:, 1:]
                # logits_gaps = logits_gaps[:, 1:]
                
                loss_notes     = self.criterion(logits_notes.transpose(1, 2), target_notes)
                loss_durations = self.criterion(logits_durations.transpose(1, 2), target_durations)
                loss_gaps      = self.criterion(logits_gaps.transpose(1, 2), target_gaps)
                loss = self.note_loss_weight * loss_notes + self.duration_loss_weight * loss_durations + self.gap_loss_weight * loss_gaps
                
                total_loss += loss.item()

                if not is_printed and self.print_predictions:
                    print(f"\nInput: {data['input_ids'][0].cpu().numpy()}")
                    print(f"\nTarget notes: {target_notes[0].cpu().numpy()}")
                    print(f"Predicted notes: {decoder_outputs_notes[0].cpu().numpy()}")
                    print(f"\nTarget durations: {target_durations[0].cpu().numpy()}")
                    print(f"Predicted durations: {decoder_outputs_durations[0].cpu().numpy()}")
                    print(f"\nTarget gaps: {target_gaps[0].cpu().numpy()}")
                    print(f"Predicted gaps: {decoder_outputs_gaps[0].cpu().numpy()}")
                    is_printed = True

                progress_bar.set_postfix({'Validation Loss': total_loss / len(self.val_loader)})
                true_notes.extend(target_notes.cpu().numpy())
                true_durations.extend(target_durations.cpu().numpy())
                true_gaps.extend(target_gaps.cpu().numpy())
                predicted_notes.extend(decoder_outputs_notes.cpu().numpy())
                predicted_durations.extend(decoder_outputs_durations.cpu().numpy())
                predicted_gaps.extend(decoder_outputs_gaps.cpu().numpy())
        
        # Convert to numpy arrays
        predicted_notes = np.asarray(predicted_notes)
        true_notes = np.asarray(true_notes)
        predicted_durations = np.asarray(predicted_durations)
        true_durations = np.asarray(true_durations)
        predicted_gaps = np.asarray(predicted_gaps)
        true_gaps = np.asarray(true_gaps)

        return total_loss / len(self.val_loader), predicted_notes, predicted_durations, predicted_gaps, true_notes, true_durations, true_gaps

    def save_model(self, name=""):
        os.makedirs(self.out_dir, exist_ok=True)
        model_save_path = f"./{self.out_dir}/model_{name}.pt"
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Saved model checkpoint to {model_save_path}")