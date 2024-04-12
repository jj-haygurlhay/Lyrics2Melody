import torch
from torch import nn
from transformers import T5EncoderModel, T5Model, T5Config
from utils.quantize import decode_note, decode_duration, decode_gap

class LyricsEncoder(nn.Module):
    def __init__(self, pretrained_model_name="t5-small"):
        super(LyricsEncoder, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

class MultiHeadMusicDecoder(nn.Module):
    def __init__(self, config, pretrained_model_name="t5-small"):
        super(MultiHeadMusicDecoder, self).__init__()
        self.t5_model = T5Model.from_pretrained(pretrained_model_name)
        
        self.note_head = nn.Linear(config.d_model, config.note_vocab_size)
        self.duration_head = nn.Linear(config.d_model, config.duration_vocab_size)
        self.gap_head = nn.Linear(config.d_model, config.gap_vocab_size)

    def forward(self, encoder_hidden_states, decoder_input_ids, attention_mask=None):

        if decoder_input_ids is None:
            decoder_input_ids = torch.full((encoder_hidden_states.size(0), 1),
                                           self.t5_model.config.pad_token_id,
                                           dtype=torch.long,
                                           device=encoder_hidden_states.device)
        
        decoder_outputs = self.t5_model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=self.create_causal_mask(decoder_input_ids.size(1)),
            encoder_attention_mask=attention_mask
        )
        
        note_logits = self.note_head(decoder_outputs.last_hidden_state)
        duration_logits = self.duration_head(decoder_outputs.last_hidden_state)
        gap_logits = self.gap_head(decoder_outputs.last_hidden_state)

        return note_logits, duration_logits, gap_logits

    def create_causal_mask(self, size):
        """Creates a causal mask to hide future tokens."""
        mask = torch.triu(torch.ones((size, size), device=self.t5_model.device), diagonal=1).bool()
        return mask.unsqueeze(0)

class CustomSeq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(CustomSeq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.note_loss_weight = kwargs.get('note_loss_weight', 0.5)
        self.duration_loss_weight = kwargs.get('duration_loss_weight', 0.25)
        self.gap_loss_weight = kwargs.get('gap_loss_weight', 0.25)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        note_logits, duration_logits, gap_logits = self.decoder(
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask
        )

        return note_logits, duration_logits, gap_logits
    
    def compute_loss(self, note_logits, duration_logits, gap_logits, note_targets, duration_targets, gap_targets):
        ignore_index = -1
        note_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(note_logits.view(-1, note_logits.size(-1)), note_targets.view(-1))
        duration_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(duration_logits.view(-1, duration_logits.size(-1)), duration_targets.view(-1))
        gap_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(gap_logits.view(-1, gap_logits.size(-1)), gap_targets.view(-1))
        
        total_loss = (note_loss * self.note_loss_weight +
                    duration_loss * self.duration_loss_weight +
                    gap_loss * self.gap_loss_weight)
        return total_loss


    def decode_outputs(self, note_logits, duration_logits, gap_logits):
        predicted_notes = torch.argmax(torch.softmax(note_logits, dim=-1), dim=-1)
        predicted_durations = torch.argmax(torch.softmax(duration_logits, dim=-1), dim=-1)
        predicted_gaps = torch.argmax(torch.softmax(gap_logits, dim=-1), dim=-1)

        # per item in batch
        batch_decoded_notes = []
        batch_decoded_durations = []
        batch_decoded_gaps = []

        for i in range(predicted_notes.shape[0]):
            notes = [decode_note(n.item()) for n in predicted_notes[i]]
            durations = [decode_duration(d.item()) for d in predicted_durations[i]]
            gaps = [decode_gap(g.item()) for g in predicted_gaps[i]]

            batch_decoded_notes.append(notes)
            batch_decoded_durations.append(durations)
            batch_decoded_gaps.append(gaps)

        return batch_decoded_notes, batch_decoded_durations, batch_decoded_gaps