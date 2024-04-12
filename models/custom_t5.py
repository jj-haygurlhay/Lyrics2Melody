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

    def forward(self, encoder_hidden_states, decoder_input_ids=None, initial_input=None, mask=None):
        if self.feedback_mode:
            outputs = []
            input = initial_input
            hidden = None
            for i in range(decoder_input_ids.size(1)):
                output, hidden = self.shared_decoder_layers(input, hidden)
                if mask is not None:
                    output = output.masked_fill(mask[:, i].unsqueeze(1) == 0, 0)

                note_logits = self.note_head(output)
                duration_logits = self.duration_head(output)
                gap_logits = self.gap_head(output)
                outputs.append((note_logits, duration_logits, gap_logits))

                input = torch.cat((note_logits, duration_logits, gap_logits), dim=-1)
            return torch.stack(outputs, dim=1)
        else:
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

        mask = self.create_combined_mask(decoder_input_ids) if decoder_input_ids is not None else None

        if self.feedback_mode and initial_input is not None:
            decoder_outputs = self.decoder(encoder_hidden_states=encoder_outputs,
                                        decoder_input_ids=decoder_input_ids,
                                        initial_input=initial_input,
                                        mask=mask)
        else:
            decoder_outputs = self.decoder(encoder_hidden_states=encoder_outputs, mask=mask)

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

    def create_combined_mask(self, input_ids):
        pad_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_len]

        # causal mask -> prevents attention to future tokens
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=input_ids.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, seq_len, seq_len]

        combined_mask = pad_mask & causal_mask

        return combined_mask.float().masked_fill(combined_mask == False, 0).masked_fill(combined_mask == True, float('-inf'))


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