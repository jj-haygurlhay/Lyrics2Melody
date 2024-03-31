from models.base_model import BaseModel 
from transformers import T5ForConditionalGeneration

class MusicT5(BaseModel):
    def __init__(self, config, note_loss_weight=1.0, duration_loss_weight=1.0, gap_loss_weight=1.0):
        super().__init__(note_loss_weight=note_loss_weight, duration_loss_weight=duration_loss_weight, gap_loss_weight=gap_loss_weight)
        # conditional generation -> designed for seq-to-seq tasks
        self.t5 = T5ForConditionalGeneration(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        For training, labels should be provided. For inference/generation, labels will be None.
        """
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **generation_kwargs):
        """
        Generate method for inference. 
        Can give additional arguments for generation such as max_length, 
        num_beams, etc., as keyword arguments (generation_kwargs).
        """
        generated_tokens = self.t5.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        
        return generated_tokens
    
    def decode_generated_sequence(self, tokenizer, generated_token_ids):
        # Convert token IDs back to tokens (words)
        tokens = tokenizer.convert_ids_to_tokens(generated_token_ids, skip_special_tokens=True)
        
        notes = []
        durations = []
        gaps = []
        
        for token in tokens:
            if token.startswith("note"):
                note = int(token.replace("note", ""))
                notes.append(note)
            elif token.startswith("duration"):
                duration = float(token.replace("duration", ""))
                durations.append(duration)
            elif token.startswith("gap"):
                gap = float(token.replace("gap", ""))
                gaps.append(gap)

        return notes, durations, gaps