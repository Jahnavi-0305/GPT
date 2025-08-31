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
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  #1 --> creating values for vocab

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)   # 2 --> Collecting logits of each token in vocab
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape     
            # if vocab = "hello" and logits contains values of 'h' then B = 1 (just one sequence: "hello") 
            # T = 5 (five characters: h, e, l, l, o) 
            # C = 5 (vocab size is 5)
            logits = (B*T, C)      #Because cross_entropy cannot compute the loss over 3D inputs—it needs a flat list of predictions for each token.
            targets = (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)  
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, idx_next), dim=1)  
        return idx
    
# Assign the generate method to the BigramLanguageModel class for token generation functionality
BigramLanguageModel.generate = generate
    
    # Initialize the model and move to the device
model = BigramLanguageModel(vocab_size).to(device)
# Define the optimizer for the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define the estimate_loss function
@torch.no_grad()
def estimate_loss():
    model.eval() 
    losses = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        loss_values = torch.zeros(eval_iters)
        for k in range(eval_iters):
            ## YOUR SOLUTION HERE##
            X, Y = get_batch(data, batch_size, block_size)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            loss_values[k] = loss.item()
        losses[split] = loss_values.mean().item()
    model.train()  # Set model back to training mode
    return losses

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        
        ## YOUR SOLUTION HERE ##
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Evaluate the loss
    ## YOUR SOLTION HERE ##
    logits, loss = model(xb, yb)
    
    # Zero out the gradients to avoid accumulation
    optimizer.zero_grad(set_to_none=True)
    
    # Backpropagation
    loss.backward()
    
    # Optimizer step
    optimizer.step()

# Generate new text with the trained model
## YOUR SOLUTION HERE##
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print("Generated text:")
print(generated_text)
