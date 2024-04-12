import torch
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from midi2audio import FluidSynth
from inference.generate_midi import GenerateMidi
from utils.generate_filename import generate_filename


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, **hparams):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.checkpoint_path = "./outputs/checkpoints/"
        
        self.feedback_mode = hparams.get('feedback_mode', False)
        self.epochs = hparams.get("epochs", 5)
        self.lr = float(hparams.get("lr", 1e-4))
        self.batch_size = hparams.get("batch_size", 8)
        
        self.model.decoder.feedback_mode = self.feedback_mode

        num_training_steps = len(train_loader) * self.epochs
        self.optimizer = AdamW(model.parameters(), lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        
        self.writer = SummaryWriter()
        self.writer.add_hparams(hparams, {})
        self.best_val_loss = float('inf')

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.model.decoder.feedback_mode = self.feedback_mode
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                
                note_targets = batch['note_targets'].to(self.device)
                duration_targets = batch['duration_targets'].to(self.device)
                gap_targets = batch['gap_targets'].to(self.device)

                note_logits, duration_logits, gap_logits = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                loss = self.model.compute_loss(note_logits, duration_logits, gap_logits, note_targets, duration_targets, gap_targets)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'Training Loss': total_loss / (progress_bar.n + 1)})

            avg_training_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/Training", avg_training_loss, epoch)
            print(f"\nEpoch {epoch+1}, Training Loss: {avg_training_loss}")
            self.validate(epoch)

            if (epoch + 1) % 5 == 0:
                self.save_model(epoch)

        self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        self.model.decoder.feedback_mode = False
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                
                note_targets = batch['note_targets'].to(self.device)
                duration_targets = batch['duration_targets'].to(self.device)
                gap_targets = batch['gap_targets'].to(self.device)

                note_logits, duration_logits, gap_logits = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                loss = self.model.compute_loss(note_logits, duration_logits, gap_logits, note_targets, duration_targets, gap_targets)

                total_loss += loss.item()

        avg_validation_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/Validation", avg_validation_loss, epoch)
        print(f"Validation Loss: {avg_validation_loss}")

        if avg_validation_loss < self.best_val_loss:
            self.best_val_loss = avg_validation_loss
            self.save_model(epoch, "best")

    def test(self):
        self.model.eval()
        self.model.decoder.feedback_mode = False
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                
                note_targets = batch['note_targets'].to(self.device)
                duration_targets = batch['duration_targets'].to(self.device)
                gap_targets = batch['gap_targets'].to(self.device)

                note_logits, duration_logits, gap_logits = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                loss = self.model.compute_loss(note_logits, duration_logits, gap_logits, note_targets, duration_targets, gap_targets)
                
                total_loss += loss.item()

                # Decode outputs to human-readable format
                notes, durations, gaps = self.model.decode_outputs(note_logits, duration_logits, gap_logits)

                # Generate and save MIDI and WAV files
                midi_filename = generate_filename("generated_music", "mid", i)
                wav_filename = generate_filename("generated_music", "wav", i)
                midi_generator = GenerateMidi(notes, durations, gaps)
                midi_pattern = midi_generator.create_midi_pattern_from_discretized_data(list(zip(notes, durations, gaps)))
                midi_pattern.write(midi_filename)
                FluidSynth().midi_to_audio(midi_filename, wav_filename)
                print(f"Generated MIDI and WAV files saved as {midi_filename} and {wav_filename}")

        avg_test_loss = total_loss / len(self.test_loader)
        self.writer.add_scalar("Loss/Test", avg_test_loss)
        print(f"Test Loss: {avg_test_loss}")


    def save_model(self, epoch, name=""):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        model_save_path = os.path.join(self.checkpoint_path, f"model_epoch_{name if name else epoch+1}.bin")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Saved model checkpoint to {model_save_path}")
