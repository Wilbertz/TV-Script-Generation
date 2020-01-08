import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # Get the number of batches we can make
    n_batches = len(words) // batch_size
    # Keep only enough characters to make full batches
    words = words[:n_batches * batch_size]
    
    x, y = [], []
    
    for idx_start in range(0, len(words) - sequence_length):
        idx_end = sequence_length + idx_start

        x_batch = words[idx_start:idx_end]
        y_batch = words[idx_end]

        x.append(x_batch)
        y.append(y_batch)

    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    # return a data loader
    return data_loader


test_text = range(50)

t_loader = batch_data(test_text, sequence_length=5, batch_size=10)
data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)