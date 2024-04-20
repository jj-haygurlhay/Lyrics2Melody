from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from project_utils.quantize import MIDI_NOTES, DURATIONS, GAPS

# Inspired from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class CustomModelRNN(BaseModel):

    def __init__(self, input_size, encoder_hidden_size, embedding_dim, decoder_hidden_size, num_layers, device, SOS_token=0, MAX_LENGTH=100, dropout_p=0.1):
        super().__init__()
        self.device = device

        # Define Encoder
        self.encoder = EncoderRNN(
            input_size=input_size, 
            encoder_hidden_size=encoder_hidden_size, 
            decoder_hidden_size=decoder_hidden_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout_p=dropout_p
            )

        # Define decoders
        self.decoder = AttnDecoderRNN(
            embedding_dim=embedding_dim,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            output_size_note=len(MIDI_NOTES)+2, 
            output_size_duration=len(DURATIONS)+2,
            output_size_gap=len(GAPS)+2,
            num_layers=num_layers,
            SOS_token=SOS_token,
            MAX_LENGTH=MAX_LENGTH,
            dropout_p=dropout_p,
            device=device
        )

        # Initialize weights
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        self.encoder.to(device)
        self.decoder.to(device)

    def forward(self, x, target=None, generate_temp=1.0, topk=None):            
        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, decoder_hidden, attentions, decoded_notes, decoded_durations, decoded_gaps = self.decoder(encoder_outputs, encoder_hidden, target, generate_temp, topk)
        return decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, decoder_hidden, attentions, decoded_notes, decoded_durations, decoded_gaps

class EncoderRNN(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, num_layers, embedding_dim, decoder_hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, encoder_hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        self.fc = nn.Linear((encoder_hidden_size * 2) * num_layers, decoder_hidden_size * num_layers)
        self.dropout = nn.Dropout(dropout_p)
        self.decoder_hidden_size = decoder_hidden_size

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)

        hidden = torch.reshape(hidden.permute(1, 0, 2), (-1, hidden.shape[2] * hidden.shape[0]))
        hidden = torch.tanh(self.fc(hidden))
        hidden = torch.reshape(hidden, (-1, input.shape[0], self.decoder_hidden_size))
        return output, hidden

# Inspired from https://github.com/bentrevett/pytorch-seq2seq/blob/main/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.attn_fc = nn.Linear((encoder_hidden_size * 2) + decoder_hidden_size, decoder_hidden_size)
        self.v_fc = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_length = encoder_outputs.shape[0]
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src length, decoder hidden dim]
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src length, decoder hidden dim]
        attention = self.v_fc(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1)

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_dim, encoder_hidden_size, decoder_hidden_size, output_size_note, output_size_duration, output_size_gap, num_layers, dropout_p=0.1, device='cpu', SOS_token = 0, MAX_LENGTH = 100):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size_note + output_size_duration + output_size_gap, embedding_dim)
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        self.gru = nn.GRU((encoder_hidden_size * 2) +  (3 * embedding_dim), decoder_hidden_size, num_layers=num_layers, dropout=dropout_p if num_layers > 1 else 0)
        self.out_note = nn.Linear((encoder_hidden_size * 2) + decoder_hidden_size + embedding_dim, output_size_note)
        self.out_duration = nn.Linear((encoder_hidden_size * 2) + decoder_hidden_size + embedding_dim, output_size_duration)
        self.out_gap = nn.Linear((encoder_hidden_size * 2) + decoder_hidden_size + embedding_dim, output_size_gap)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.SOS_token = SOS_token
        self.MAX_LENGTH = MAX_LENGTH

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, temperature=1.0, topk=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs_notes = []
        decoder_outputs_durations = []
        decoder_outputs_gaps = []
        attentions = []
        decoded_notes = []
        decoded_durations = []
        decoded_gaps = []

        for i in range(self.MAX_LENGTH):
            decoder_output_note,decoder_output_duration, decoder_output_gap, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs_notes.append(decoder_output_note)
            decoder_outputs_durations.append(decoder_output_duration)
            decoder_outputs_gaps.append(decoder_output_gap)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i] # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                
                # Apply temperature to the output 
                decoder_output_note = decoder_output_note[:, -1, :] / temperature
                decoder_output_duration = decoder_output_duration[:, -1, :] / temperature
                decoder_output_gap = decoder_output_gap[:, -1, :] / temperature

                # Get topk
                if topk is not None:
                    topk_note, _ = torch.topk(decoder_output_note, topk[0])
                    topk_duration, _ = torch.topk(decoder_output_duration, topk[1])
                    topk_gap, _ = torch.topk(decoder_output_gap, topk[2])
                    decoder_output_note[decoder_output_note < topk_note[:, [-1]]] = -float('Inf')
                    decoder_output_duration[decoder_output_duration < topk_duration[:, [-1]]] = -float('Inf')
                    decoder_output_gap[decoder_output_gap < topk_gap[:, [-1]]] = -float('Inf')

                # Apply softmax to the output
                probs_note = F.softmax(decoder_output_note, dim=-1)
                probs_duration = F.softmax(decoder_output_duration, dim=-1)
                probs_gap = F.softmax(decoder_output_gap, dim=-1)

                # Sample from the output
                note_next_tokens = torch.multinomial(probs_note.squeeze(1), 1)
                duration_next_tokens = torch.multinomial(probs_duration.squeeze(1), 1)
                gap_next_tokens = torch.multinomial(probs_gap.squeeze(1), 1)

                # Compute next scores
                note_scores = torch.gather(decoder_output_note.squeeze(1), -1, note_next_tokens)
                duration_scores = torch.gather(decoder_output_duration.squeeze(1), -1, duration_next_tokens)
                gap_scores = torch.gather(decoder_output_gap.squeeze(1), -1, gap_next_tokens)

                note_scores, note_indices = note_scores.sort(descending=True, dim=1)
                duration_scores, duration_indices = duration_scores.sort(descending=True, dim=1)
                gap_scores, gap_indices = gap_scores.sort(descending=True, dim=1)

                decoder_input_note = torch.gather(note_next_tokens, -1, note_indices)
                decoder_input_duration = torch.gather(duration_next_tokens, -1, duration_indices)
                decoder_input_gap = torch.gather(gap_next_tokens, -1, gap_indices)

                decoded_notes.append(decoder_input_note)
                decoded_durations.append(decoder_input_duration)
                decoded_gaps.append(decoder_input_gap)

                decoder_input = torch.cat((decoder_input_note, decoder_input_duration, decoder_input_gap), dim=-1)

        decoder_outputs_notes = torch.cat(decoder_outputs_notes, dim=1)
        decoder_outputs_durations = torch.cat(decoder_outputs_durations, dim=1)
        decoder_outputs_gaps = torch.cat(decoder_outputs_gaps, dim=1)

        attentions = torch.cat(attentions, dim=1)

        if target_tensor is None:
            decoded_notes = torch.cat(decoded_notes, dim=1)
            decoded_durations = torch.cat(decoded_durations, dim=1)
            decoded_gaps = torch.cat(decoded_gaps, dim=1)

        return (decoder_outputs_notes, 
                decoder_outputs_durations, 
                decoder_outputs_gaps, 
                decoder_hidden, 
                attentions, 
                decoded_notes, 
                decoded_durations, 
                decoded_gaps)


    def forward_step(self, input, hidden, encoder_outputs):
        input = input.permute(1, 0)
        embedded = self.dropout(self.embedding(input))
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_weights = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(attn_weights, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        embedded_note = embedded[0, :, :].unsqueeze(0)
        embedded_duration = embedded[1, :, :].unsqueeze(0)
        embedded_gap = embedded[2, :, :].unsqueeze(0)
        input_gru = torch.cat((embedded_note, embedded_duration, embedded_gap, weighted), dim=2)

        if len(hidden.shape) == 2:
            hidden = hidden.unsqueeze(0)

        output, hidden = self.gru(input_gru, hidden)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        out_note = self.out_note(torch.cat((output, weighted, embedded_note.squeeze(0)), dim=1))
        out_duration = self.out_duration(torch.cat((output, weighted, embedded_duration.squeeze(0)), dim=1))
        out_gap = self.out_gap(torch.cat((output, weighted, embedded_gap.squeeze(0)), dim=1))
        return out_note.unsqueeze(1), out_duration.unsqueeze(1), out_gap.unsqueeze(1), hidden, attn_weights.squeeze(1)   