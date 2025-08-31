import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2  # Batch size, sequence length, embedding dimension
x = torch.randn(B, T, C)
print("Shape of x:", x.shape)

x_slice = x[0, :4, :]  # Shape: [4, 2]

print("Shape of x_slice:", x_slice.shape)
print("x_slice:\n", x_slice)
b = 0
t = 3
xprev = x[b, :t+1]  # Get embeddings from time 0 to 3
mean_embedding = torch.mean(xprev, dim=0)

print("Mean embedding up to time step", t, ":\n", mean_embedding)
# Loop method
xbow_loop = torch.zeros_like(x)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow_loop[b, t] = torch.mean(xprev, dim=0)

# Vectorized method
x_cumsum = torch.cumsum(x, dim=1)
counts = torch.arange(1, T + 1).view(1, T, 1).to(x.dtype)
xbow_vectorized = x_cumsum / counts

# Compare embeddings at batch 0, time step 5
b = 0
t = 5
embedding_loop = xbow_loop[b, t]
embedding_vectorized = xbow_vectorized[b, t]

print("Embedding from loop method:\n", embedding_loop)
print("Embedding from vectorized method:\n", embedding_vectorized)
print("Difference:\n", embedding_loop - embedding_vectorized)
