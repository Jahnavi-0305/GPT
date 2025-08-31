# GPT

BIGRAM MODEL: 
Step 1: Building a Baseline Model 1
1. Preprocessing using regex
2. Tokenization using encoding, decoding functions
3. Batching data into training and validation sets. Then, it is organized into batches for parallel processing

About Dataset: War and Peace by Leo Tolstoy, which contains over 3.2 million characters. 
I’ve inspected the first 200 characters of the dataset and extracted all the unique characters, resulting in a vocabulary size of 112

Step 2: Building a Baseline Model 1
1. Model Implementation: Construct a neural network that uses an embedding layer to predict the next character in a sequence.
2. Optimizer Setup: Define the optimizer to update the model’s parameters during training.
3. Loss Function: Measure how well the model’s predictions align with the actual data. This guides the training process by minimizing the loss value.
4. Training Loop: Iteratively update the model’s parameters using backpropagation while monitoring performance on both training and validation datasets.
5. Text Generation: Use the trained model to generate new sequences of text based on learned patterns.

Baseline 1 and Baseline 2 are bigram models. They predict the next character using only the previous character, processing the text one step at a time.

For better performance, we use transformers. Transformers can look at all the previous characters in a sequence when predicting the next one. This helps the model remember important things from earlier in the text, called long-range dependencies.

For example:
"The princess opened the door, and she saw the dragon."

A bigram model only sees the last character, so it doesn’t know that “she” refers to “the princess”.

A transformer sees the whole previous text, so it understands the context and predicts the next character much better.

USING TRANSFORMERS:



