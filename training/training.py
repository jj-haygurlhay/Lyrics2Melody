import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, device, collator, epochs=4, batch_size=8, lr=5e-5):
        self.model = model
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.device = device
        self.collator = collator
        self.epochs = epochs
        self.lr = float(lr)
        self.optimizer = AdamW(model.parameters(), lr=self.lr)
        num_training_steps = int(len(train_dataset) / batch_size * epochs)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        
        self.batch_size = batch_size

        self.writer = SummaryWriter()
        self.checkpoint_path = "./checkpoints/"

    def train(self):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator, pin_memory=True)
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for batch in progress_bar:
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
                progress_bar.set_postfix({'Training Loss': total_loss / len(train_loader)})

                self.writer.add_scalar("Loss/Train", total_loss / len(train_loader), epoch)
            
            # Periodic model saving
            if (epoch + 1) % 5 == 0:
                self.save_model(epoch)
        

            print(f"\nEpoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}")
            self.validate()

        self.writer.close()

    def validate(self):
        val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, collate_fn=self.collator, pin_memory=True)
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(val_loader, desc="Validating")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        print(f"\nValidation Loss: {total_loss / len(val_loader)}")
        self.writer.add_scalar("Loss/Validation", total_loss / len(val_loader), epoch)

    def test(self):
        test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, collate_fn=self.collator, pin_memory=True)
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(test_loader, desc="Testing")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        print(f"\nTest Loss: {total_loss / len(test_loader)}")

    def save_model(self, epoch):
        model_save_path = f"./{self.checkpoint_path}/model_epoch_{epoch+1}.bin"
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Saved model checkpoint to {model_save_path}")
