import torch
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, pl_trainer, device, train_loader, val_loader, test_loader, collator, **hparams):
        self.model = model
        self.pl_trainer = pl_trainer
        self.device = device
        self.checkpoint_path = "./checkpoints/"
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.collator = collator
        self.epochs = hparams["epochs"]
        self.lr = float(hparams["lr"])
        self.batch_size = hparams["batch_size"]


        num_training_steps = int(len(train_loader) / self.batch_size * self.epochs)
        self.optimizer = AdamW(model.parameters(), lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        

        self.writer = SummaryWriter()
        self.writer.add_hparams(hparams, {})

        self.best_val_loss = -100

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

            for i, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'Training Loss': total_loss / len(self.train_loader)})

            avg_training_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/Training", avg_training_loss, epoch)
            
            # Periodic model saving
            # if (epoch + 1) % 5 == 0:
            #     self.save_model(epoch)
        

            print(f"\nEpoch {epoch+1}, Training Loss: {total_loss / len(self.train_loader)}")
            self.validate(epoch)

        self.save_model(epoch)
        self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc="Validating")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        avg_validation_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_validation_loss}")
        self.writer.add_scalar("Loss/Validation", avg_validation_loss, epoch)

        if avg_validation_loss < self.best_val_loss:
            self.best_val_loss = avg_validation_loss
            self.save_model(epoch=None, name="best")

    def test(self):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.test_loader, desc="Testing")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        print(f"\nTest Loss: {total_loss / len(self.test_loader)}")
        self.writer.add_scalar("Loss/Test", total_loss / len(self.test_loader))

    def save_model(self, epoch, name=""):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if name == "":
            model_save_path = f"./{self.checkpoint_path}/model_epoch_{epoch+1}.bin"
        else:
            model_save_path = f"./{self.checkpoint_path}/model_epoch_{name}.bin"
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Saved model checkpoint to {model_save_path}")
