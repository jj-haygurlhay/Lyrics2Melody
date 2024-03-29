import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, test_dataset, device, collator, epochs=4, lr=5e-5):
        self.model = model
        self.tokenizer = tokenizer
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.device = device
        self.collator = collator
        self.epochs = epochs
        
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=0, 
            num_training_steps=len(train_dataset) * epochs
        )

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, collate_fn=self.collator)
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                self.optimizer.zero_grad()

                input_ids, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'Training Loss': total_loss / len(train_loader)})

            print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}")
            self.validate()

    def validate(self):
        val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, collate_fn=self.collator)
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(val_loader, desc="Validating")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss / len(val_loader)}")

    def test(self):
        test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, collate_fn=self.collator)
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(test_loader, desc="Testing")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        print(f"Test Loss: {total_loss / len(test_loader)}")


