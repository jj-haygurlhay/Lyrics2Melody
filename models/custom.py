from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quantize import MIDI_NOTES, DURATIONS, GAPS


class CustomModel(BaseModel):

    def __init__(self, encoder, device, SOS_token=0, MAX_LENGTH=100, dropout_p=0.1, train_encoder=False):
        super().__init__()
        self.device = device

        # Take pretrained encoder
        self.encoder = encoder

        # Define decoders
        self.decoder = AttnDecoderRNN(
            hidden_size=encoder.config.n_positions,
            output_size_note=len(MIDI_NOTES)+2, 
            output_size_duration=len(DURATIONS)+2,
            output_size_gap=len(GAPS)+2,
            SOS_token=SOS_token,
            MAX_LENGTH=MAX_LENGTH,
            dropout_p=dropout_p,
            device=device
        )

        self.train_encoder = train_encoder

        self.encoder.to(device)
        self.decoder.to(device)

    def forward(self, x, attn, target=None):
        if not self.train_encoder:
            with torch.no_grad():
                encoder_hidden, encoder_outputs = self.encoder(x, attn, return_dict=False, output_hidden_states=True, output_attentions=False)
        else:
            encoder_hidden, encoder_outputs  = self.encoder(x, attn, return_dict=False, output_hidden_states=True, output_attentions=False)
        encoder_hidden = encoder_hidden.permute(1, 0, 2)[-1, :, :].unsqueeze(0).contiguous()
        decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, decoder_hidden, attentions = self.decoder(encoder_outputs[0], encoder_hidden, target)
        return decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, decoder_hidden, attentions


# Code from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size_note, output_size_duration, output_size_gap, dropout_p=0.1, device='cpu', SOS_token = 0, MAX_LENGTH = 100):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size_note + output_size_duration + output_size_gap, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(4 * hidden_size, hidden_size, batch_first=True)
        self.out_note = nn.Linear(hidden_size, output_size_note)
        self.out_duration = nn.Linear(hidden_size, output_size_duration)
        self.out_gap = nn.Linear(hidden_size, output_size_gap)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.SOS_token = SOS_token
        self.MAX_LENGTH = MAX_LENGTH

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs_notes = []
        decoder_outputs_durations = []
        decoder_outputs_gaps = []
        attentions = []

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
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi_note = decoder_output_note.topk(1)
                _, topi_duration = decoder_output_duration.topk(1)
                _, topi_gap = decoder_output_gap.topk(1)
                decoder_input_note = topi_note.squeeze(-1).detach()  # detach from history as input
                decoder_input_duration = topi_duration.squeeze(-1).detach()
                decoder_input_gap = topi_gap.squeeze(-1).detach()
                decoder_input = torch.cat((decoder_input_note, decoder_input_duration, decoder_input_gap), dim=-1)

        decoder_outputs_notes = torch.cat(decoder_outputs_notes, dim=1)
        decoder_outputs_notes = F.log_softmax(decoder_outputs_notes, dim=-1)
        decoder_outputs_durations = torch.cat(decoder_outputs_durations, dim=1)
        decoder_outputs_durations = F.log_softmax(decoder_outputs_durations, dim=-1)
        decoder_outputs_gaps = torch.cat(decoder_outputs_gaps, dim=1)
        decoder_outputs_gaps = F.log_softmax(decoder_outputs_gaps, dim=-1)

        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input)).reshape(input.size(0), 1, -1)

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        out_note = self.out_note(output)
        out_duration = self.out_duration(output)
        out_gap = self.out_gap(output)
        return out_note, out_duration, out_gap, hidden, attn_weights   