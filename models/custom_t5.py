import torch
from torch import nn
from transformers import T5EncoderModel, T5Config, T5ForConditionalGeneration, T5Model
from utils.quantize import decode_note, decode_duration, decode_gap

class LyricsEncoder(nn.Module):
    def __init__(self, pretrained_model_name="t5-small"):
        super(LyricsEncoder, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_name)

        # additional layers can be added here for further processing
        self.custom_layers = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, self.encoder.config.d_model),
            nn.ReLU(),
        )

    def forward(self, input_ids, attention_mask=None):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        custom_output = self.custom_layers(encoder_output.last_hidden_state)
        
        return custom_output


class MultiHeadMusicDecoder(nn.Module):
    def __init__(self, config, feedback_mode):
        super(MultiHeadMusicDecoder, self).__init__()
        self.feedback_mode = feedback_mode
        self.shared_decoder_layers = nn.GRU(input_size=config.d_model, 
                                             hidden_size=config.d_model, 
                                             num_layers=config.num_decoder_layers, 
                                             batch_first=True)
        
        # Initialize separate output heads for notes, durations, and gaps
        self.note_head = nn.Linear(config.d_model, config.note_vocab_size)
        self.duration_head = nn.Linear(config.d_model, config.duration_vocab_size)
        self.gap_head = nn.Linear(config.d_model, config.gap_vocab_size)

    def forward(self, encoder_hidden_states, decoder_input_ids=None, initial_input=None):
        if self.feedback_mode:
            # Iterative feedback mode
            outputs = []
            input = initial_input
            hidden = None
            for _ in range(decoder_input_ids.size(1)):  # assuming sequence length dimension
                output, hidden = self.shared_decoder_layers(input, hidden)
                note_logits = self.note_head(output)
                duration_logits = self.duration_head(output)
                gap_logits = self.gap_head(output)
                outputs.append((note_logits, duration_logits, gap_logits))

                # use previous output as the next input (teacher forcing during training)
                input = torch.cat((note_logits, duration_logits, gap_logits), dim=-1)
            return torch.stack(outputs, dim=1)  # match sequence length dimension
        else:
            # Non-iterative mode
            output, _ = self.shared_decoder_layers(encoder_hidden_states)
            note_logits = self.note_head(output)
            duration_logits = self.duration_head(output)
            gap_logits = self.gap_head(output)
            return note_logits, duration_logits, gap_logits

class CustomSeq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(CustomSeq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.note_loss_weight = kwargs.get('note_loss_weight', 0.5)
        self.duration_loss_weight = kwargs.get('duration_loss_weight', 0.25)
        self.gap_loss_weight = kwargs.get('gap_loss_weight', 0.25)
        
        self.feedback_mode = kwargs.get('feedback_mode', False)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, initial_input=None):
        encoder_outputs = self.encoder(input_ids, attention_mask)
        
        if self.feedback_mode:
            if initial_input is None:
                raise ValueError("Initial input is required for iterative feedback mode")
            # need to make sure that `initial_input` is used as the starting point for the feedback loop
            decoder_outputs = self.decoder(encoder_hidden_states=encoder_outputs,
                                           decoder_input_ids=decoder_input_ids,
                                           initial_input=initial_input)
        else:
            # non-iterative mode -> generate all outputs at once
            decoder_outputs = self.decoder(encoder_hidden_states=encoder_outputs)
        
        return decoder_outputs
    
    def compute_loss(self, note_logits, duration_logits, gap_logits, note_targets, duration_targets, gap_targets):
        ignore_index = -1
        note_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(note_logits.view(-1, note_logits.size(-1)), note_targets.view(-1))
        duration_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(duration_logits.view(-1, duration_logits.size(-1)), duration_targets.view(-1))
        gap_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(gap_logits.view(-1, gap_logits.size(-1)), gap_targets.view(-1))
        
        total_loss = (note_loss * self.note_loss_weight +
                    duration_loss * self.duration_loss_weight +
                    gap_loss * self.gap_loss_weight)
        return total_loss

    def create_causal_mask(self, input_ids):
        """
        Creates a causal attention mask to prevent tokens from attending to future tokens.
        The mask will have shape [batch_size, seq_length, seq_length].
        
        Args:
            input_ids (torch.Tensor): The input IDs tensor with shape [batch_size, seq_length].
        
        Returns:
            torch.Tensor: The causal attention mask with values 0 for allowed attentions and -inf for blocked attentions.
        """
        batch_size, seq_length = input_ids.size()
        mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.float32) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1) 
        return mask

def decode_outputs(note_logits, duration_logits, gap_logits):
    predicted_notes = torch.argmax(torch.softmax(note_logits, dim=-1), dim=-1)
    predicted_durations = torch.argmax(torch.softmax(duration_logits, dim=-1), dim=-1)
    predicted_gaps = torch.argmax(torch.softmax(gap_logits, dim=-1), dim=-1)

    # Assuming you have mapping functions defined
    notes = [decode_note(n.item()) for n in predicted_notes]
    durations = [decode_duration(d.item()) for d in predicted_durations]
    gaps = [decode_gap(g.item()) for g in predicted_gaps]

    return notes, durations, gaps