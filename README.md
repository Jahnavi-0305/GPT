# GPT

Step 1: Building a Baseline Model 1
1. Preprocessing using regex
2. Tokenization using encoding, decoding functions
3. Batching data into training and validation sets. Then, it is organized into batches for parallel processing

About Dataset: War and Peace by Leo Tolstoy, which contains over 3.2 million characters. 
I’ve inspected the first 200 characters of the dataset and extracted all the unique characters, resulting in a vocabulary size of 112

Step 2: 
1. Model Implementation: Construct a neural network that uses an embedding layer to predict the next character in a sequence.
2. Optimizer Setup: Define the optimizer to update the model’s parameters during training.
3. Loss Function: Measure how well the model’s predictions align with the actual data. This guides the training process by minimizing the loss value.
4. Training Loop: Iteratively update the model’s parameters using backpropagation while monitoring performance on both training and validation datasets.
5. Text Generation: Use the trained model to generate new sequences of text based on learned patterns.
