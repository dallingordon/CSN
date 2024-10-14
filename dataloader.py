import os
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile  # If you're using scipy for reading .wav files


class WaveformDatasetPreload(Dataset):
    def __init__(self, directory, t_input, max_len, terminal_pad, seq_vocab_len, seq_max_len, seq_t):
        """
        directory: Directory containing the .wav files.
        t_input: Time input array for all files.
        max_len: Maximum length of time steps needed for all files.
        terminal_pad: Number of zeros to pad at the end of each audio file.
        seq_max_len: maximum len of input sequence in tokens.
        """
        self.directory = directory
        self.files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')][:100], key=lambda x: int(x.split('.')[0]))
        self.t_input = t_input # max is taken care of outside.

        self.terminal_pad = terminal_pad  # Fixed number of zeros to pad
        self.seq_max_len = seq_max_len  # Maximum number of integers in the filename sequence
        self.seq_t = seq_t[:seq_max_len]
        self.seq_vocab_len = seq_vocab_len
        # Preload and pad filenames
        self.padded_file_name_integers = self._prepare_padded_filenames()

        # **Change here**: Preload and pad audio files during initialization
        self.wav_data_list = [self._load_and_pad(os.path.join(directory, f)) for f in self.files]
        self.file_indices = []
        self.total_length = 0

        # Calculate lengths of all files and their indices
        for i, wav_data in enumerate(self.wav_data_list):
            length = wav_data.size(1)  # Assuming data is [channels, time], we take the time dimension
            self.file_indices.extend([(i, j) for j in range(length)])
            self.total_length += length

    def _prepare_padded_filenames(self):
        """
        Converts filenames into sequences of integers, right-padded with 0s up to seq_max_len length.
        """
        padded_filenames = []
        for file_name in self.files:
            # Extract the number from the file name (without the '.wav' extension)
            file_name_base = file_name.split('.')[0]
            # Convert the number into a list of integers
            file_name_integers = [int(char) for char in file_name_base]
            # Pad the list with zeros until it matches the required length
            padded_file_name = file_name_integers + [self.seq_vocab_len] * (self.seq_max_len - len(file_name_integers))
             #### embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=10) required!
            # Convert to PyTorch tensor
            padded_filenames.append(torch.tensor(padded_file_name, dtype=torch.long))
        return padded_filenames

    def _load_and_pad(self, file_path):
        """
        **Change here**: Load and pad audio file only once during initialization.
        """
        sample_rate, data = wavfile.read(file_path)
        data = torch.tensor(data).unsqueeze(0)  # Convert to tensor and add channel dimension

        # Normalize the data to the range [-1, 1] based on int16
        if data.dtype == torch.int16:
            data = data / 32768.0  # Normalize int16 data
        elif data.dtype == torch.int32:
            data = data / 2147483648.0  # Normalize int32 data
        elif data.dtype == torch.float32:
            pass  # If it's already float, assume it's in [-1, 1]

        # Pad the data with zeros at the end
        pad_length = self.terminal_pad
        data_padded = torch.nn.functional.pad(data, (0, pad_length), mode='constant', value=0)
        return data_padded

    def _generate_target(self, wav_data):
        """
        Helper function to generate the target tensor.
        The target will have 1 in all positions except for the final terminal_pad zeros.
        """
        target = torch.ones_like(wav_data)  # Create a target tensor with all ones
        # Set the last terminal_pad positions to zero
        target[:, -self.terminal_pad:] = 0
        return target

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # **No file loading happens here**; just retrieving preloaded data
        file_idx, local_idx = self.file_indices[idx]
        wav_data = self.wav_data_list[file_idx][:, local_idx]  # Slice based on channel and index
        t_step = self.t_input[local_idx]  # Time input for the specific index
        target = self._generate_target(self.wav_data_list[file_idx])[:, local_idx]  # Generate the target tensor

        # Return the preprocessed padded file name integers for the given file_idx
        padded_file_name_integers = self.padded_file_name_integers[file_idx]

        num_padding = (padded_file_name_integers == self.seq_max_len).sum().item()

        # Retain only non-padded values in seq_t and zero out the rest
        retained_len = len(self.seq_t) - num_padding
        seq_t_adjusted = self.seq_t.clone()  # Clone to avoid modifying the original tensor
        if retained_len > 0:
            seq_t_adjusted[retained_len:] = 0  # Zero out the right-padded elements

        return wav_data, t_step, target, padded_file_name_integers, seq_t_adjusted