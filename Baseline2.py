import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
from model_setup import setup

# Initialize the configuration using the setup function
config = setup('war_and_peace.txt', block_size=8, batch_size=4)

# Accessing variables and functions from the config dictionary
text = config['text']
chars = config['chars']
vocab_size = config['vocab_size']
train_data = config['train_data']
val_data = config['val_data']  # Added val_data extraction
get_batch = config['get_batch']
decode = config['decode']
encode = config['encode']
block_size = config['block_size']
batch_size = config['batch_size']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 500
eval_iters = 200
max_iters = 5000

# Verify loaded data
print(f"Vocabulary size: {vocab_size}")
print(f"First 100 encoded characters: {config['encoded_data'][:100]}")

# Generate a batch and print details
xb, yb = get_batch(train_data, config['batch_size'], config['block_size'])
xb, yb = xb.to(device), yb.to(device)
print(f"Input batch shape: {xb.shape}")
print(f"Target batch shape: {yb.shape}")

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            ## YOUR SOLUTION HERE ##
            B, T, C = logits.shape
            logits = (B*T, C)
            targets = (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

