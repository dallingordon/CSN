import torch
import torch.nn as nn


class SwissArmyLayer(nn.Module):

    def __init__(self, t_seq_bits
                 , t_layer_dim
                 , t_num_layers
                 , num_layers
                 , one_hot_vocab_len=None  # do not include padding here.  0-9:10.  it adds one below!
                 , one_hot_embedding_dim=None
                 , input_embedding_dim=None
                 ):
        super(SwissArmyLayer, self).__init__()

        self.t_layers = nn.ModuleList()
        if t_num_layers > 0:

            self.t_layers.append(nn.Linear(t_seq_bits, t_layer_dim))  # First layer: t_bits x t_bits_hidden
            for _ in range(t_num_layers - 1):
                self.t_layers.append(nn.Linear(t_layer_dim, t_layer_dim))
        else:
            # t_seq_bits is zero, so set t_layer_dim
            t_layer_dim = t_seq_bits

        hidden_dim = t_layer_dim

        if input_embedding_dim is not None:
            hidden_dim = hidden_dim + input_embedding_dim

        if one_hot_vocab_len is not None and one_hot_embedding_dim is not None:
            hidden_dim = hidden_dim + one_hot_embedding_dim
            self.embedding = nn.Embedding(one_hot_vocab_len + 1, one_hot_embedding_dim, padding_idx=one_hot_vocab_len)

        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)])  # should work with zero too.

    def forward(self, t_seq, one_hot_idx=None, input_embedding=None):

        if one_hot_idx is not None:
            e = self.embedding(one_hot_idx)  # i question this relu

        if len(self.t_layers) > 0:
            for layer in self.t_layers:
                t_seq = torch.relu(layer(t_seq))
        if one_hot_idx is None:
            x = torch.cat((t_seq, input_embedding), dim=-1)
        else:
            if input_embedding is not None:
                x = torch.cat((t_seq, e, input_embedding),
                              dim=-1)  # Shape: [batch, t_seq_len, t_bits_hidden + embedding_dim + input_dim]
            else:
                x = torch.cat((t_seq, e), dim=-1)

        for layer in self.layers:
            x = torch.relu(layer(x))

        return x



class SeqEncoder(nn.Module):
    def __init__(self, t_seq_bits, t_seq_len, t_layer_dim, t_num_layers, fc_layers, encoder_layers, one_hot_vocab_len,
                 one_hot_embedding_dim):
        super(SeqEncoder, self).__init__()
        self.seq_len = t_seq_len
        # First layer setup like test_initial
        self.initial_layer = SwissArmyLayer(t_seq_bits=t_seq_bits,
                                            t_layer_dim=t_layer_dim,
                                            t_num_layers=t_num_layers,  # Fixed to 2 as per your example
                                            num_layers=fc_layers,  # Fixed number of layers
                                            one_hot_vocab_len=one_hot_vocab_len,
                                            one_hot_embedding_dim=one_hot_embedding_dim,
                                            input_embedding_dim=None)  # No input_embedding in initial layer

        input_embedding_dim = one_hot_embedding_dim
        if t_num_layers == 0:
            input_embedding_dim = input_embedding_dim + t_seq_bits
        else:
            input_embedding_dim = input_embedding_dim + t_layer_dim

        self.encoder_layers = nn.ModuleList()
        for _ in range(encoder_layers):
            self.encoder_layers.append(SwissArmyLayer(t_seq_bits=t_seq_bits,
                                                      t_layer_dim=t_layer_dim,
                                                      t_num_layers=t_num_layers,
                                                      num_layers=fc_layers,
                                                      one_hot_vocab_len=one_hot_vocab_len,
                                                      one_hot_embedding_dim=one_hot_embedding_dim,
                                                      input_embedding_dim=input_embedding_dim))
            # Update hidden_dim for future layers based on input_embedding_dim
            input_embedding_dim = input_embedding_dim + one_hot_embedding_dim
            if t_num_layers == 0:
                input_embedding_dim = input_embedding_dim + t_seq_bits
            else:
                input_embedding_dim = input_embedding_dim + t_layer_dim

    def forward(self, t_seq, one_hot_idx):
        # Pass through the initial layer
        x = self.initial_layer(t_seq, one_hot_idx=one_hot_idx)

        x = x.sum(dim=1, keepdim=True)
        # Pass through each encoder layer
        for layer in self.encoder_layers:
            x = x.expand(-1, self.seq_len, -1)
            x = layer(t_seq, one_hot_idx=one_hot_idx, input_embedding=x)
            x = x.sum(dim=1, keepdim=True)

        x = x.squeeze() #final layer squozen for decoder input
        return x

import torch
import torch.nn as nn

class SeqDecoder(nn.Module):
    def __init__(self, t_seq_bits, t_layer_dim, t_num_layers, fc_layers, decoder_layers, input_embedding_dim):
        super(SeqDecoder, self).__init__()

        # Set the initial input embedding dimension
        self.input_embedding_dim = input_embedding_dim

        # Create decoder layers (all the same, like test_decoder)
        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_layers):
            self.decoder_layers.append(SwissArmyLayer(t_seq_bits=t_seq_bits,
                                                      t_layer_dim=t_layer_dim,
                                                      t_num_layers=t_num_layers,
                                                      num_layers=fc_layers,
                                                      one_hot_vocab_len=None,
                                                      one_hot_embedding_dim=None,
                                                      input_embedding_dim=self.input_embedding_dim))
            # Update the input_embedding_dim for subsequent layers
            if t_num_layers == 0:
                self.input_embedding_dim += t_seq_bits
            else:
                self.input_embedding_dim += t_layer_dim

    def forward(self, t_seq, input_embedding):
        x = input_embedding

        # Pass through each decoder layer, updating the input embedding
        for layer in self.decoder_layers:
            x = layer(t_seq, input_embedding=x)

        return x



class SeqModel(nn.Module):
    def __init__(self, config):
        super(SeqModel, self).__init__()

        # Extracting values from config
        t_seq_bits = config['t_seq_bits']
        seq_len = config['t_seq_len']
        t_bits = config['t_bits']

        encoder_config = config['encoder']
        decoder_config = config['decoder']
        output_config = config['output']

        # Instantiate SeqEncoder
        self.encoder = SeqEncoder(
            t_seq_bits=t_seq_bits,
            t_seq_len=seq_len,
            t_layer_dim=encoder_config['t_layer_dim'],
            t_num_layers=encoder_config['t_num_layers'],
            fc_layers=encoder_config['fc_layers'],
            encoder_layers=encoder_config['encoder_layers'],
            one_hot_vocab_len=encoder_config['one_hot_vocab_len'],
            one_hot_embedding_dim=encoder_config['one_hot_embedding_dim']
        )

        # Calculate input_embedding_dim based on encoder output dimension
        encoder_output_dim = self.encoder.encoder_layers[-1].layers[-1].out_features

        # Instantiate SeqDecoder
        self.decoder = SeqDecoder(
            t_seq_bits=t_bits,
            t_layer_dim=decoder_config['t_layer_dim'],
            t_num_layers=decoder_config['t_num_layers'],
            fc_layers=decoder_config['fc_layers'],
            decoder_layers=decoder_config['decoder_layers'],
            input_embedding_dim=encoder_output_dim
        )

        decoder_output_dim = self.decoder.decoder_layers[-1].layers[-1].out_features
        # MSE Output Head
        self.mse_head = self._build_output_head(output_config['mse_output_layers'], decoder_output_dim,
                                                output_config['mse_dim'], final_activation='relu')

        # BCE Output Head
        self.bce_head = self._build_output_head(output_config['bce_output_layers'], decoder_output_dim,
                                                output_config['bce_dim'], final_activation='sigmoid')

    def _build_output_head(self, num_layers, input_dim, hidden_dim, final_activation):
        layers = []
        current_dim = input_dim

        # Create intermediate layers
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(current_dim, 1))  # Single output dimension

        # Final activation
        if final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, t_seq, one_hot_idx, t):
        # Encoder step
        encoded = self.encoder(t_seq, one_hot_idx=one_hot_idx)

        # Decoder step
        decoded = self.decoder(t, input_embedding=encoded)

        # Output layers
        mse_output = self.mse_head(decoded)
        bce_output = self.bce_head(decoded)

        return bce_output, mse_output
