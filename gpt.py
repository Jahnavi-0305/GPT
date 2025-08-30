import re
import torch
# Load dataset
with open('war_and_peace.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# Let's inspect the length and first 200 characters of the text to understand its structure and content.

print("Length of dataset in characters:", len(text))
# Let's look at the first 200 characters
print("\nPreview of first 200 characters:")
print(text[:200])

# Extracting Unique Characters from Text Data

chars = sorted(list(set(text)))  # Convert set to list, then sort unique characters
vocab_size = len(chars)          # Calculate vocabulary size
print(' '.join(chars))           # Display unique characters
print('\nVocab_size is', vocab_size)  # Display vocabulary size

text = re.sub(r'[^a-zA-Z\s,.:;?_&$!"()\-\*\[\]]', '', text.lower()).strip()

# show the output
# print the length of preprocessed text:
print(f'Total length of cleaned text: {len(text)}')

# print to preview the cleaned text
print(f'\n\nPreview of the cleaned text:\n{text[:200]}')
chars= sorted(list(set(text)))
vocab_size =len(chars)

# show the output
# print the unique characters and the vocabulary size:
print(' '.join(chars))
print('\n Vocab_size is',vocab_size)
stoi = { ch:i for i,ch in enumerate(chars) } 
itos = { i:ch for i,ch in enumerate(chars) } 
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])
# show the output
# print statements to see the output for encoding and decoding the string "the princess smiled".
print(encode("the princess smiled"))
print(decode(encode("the princess smiled"))) # Bigram to index
# Encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# show output
# Print the shape and data type of the data tensor, along with the first 100 elements to verify encoding.
print(data.shape, data.dtype)
print(data[:100]) # Preview of the encoded data
# Split the data into train and validation sets
n = int(0.9 * len(data)) # The first 90% of data is for training and the remaining 10% is for validation
train_data = data[:n]
val_data = data[n:]
# show output
# print the lengths of both datasets:
print(f"Training data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
b = 0  # Select the first sequence in the batch
# Display the entire tensor for the first sequence
print("Input tensor (xb):",xb[b])
for t in range(block_size):  # time dimension
    context = xb[b, :t+1]
    target = yb[b, t]
    print(f"When input is {context.tolist()} the target: {target}")
