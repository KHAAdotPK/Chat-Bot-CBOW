# My Progress on Word Embeddings and Model Implementation

## Overview
I have been working on implementing fundamental NLP models from scratch in C/C++.  
So far, I’ve implemented **Skip-gram** and **CBOW**, and I am currently developing an **Encoder-Decoder** model.  
The key idea is not to rely on external machine learning libraries, but to understand and code everything from the ground up.

---

## Word Embeddings
- I generate word embeddings using two separate weight matrices: **W1** and **W2**.  
- **W1** is used to produce the embedding for the input (target) word.  
- **W2** (transposed) is used to produce embeddings for all context words.  

The process:
1. Take the embedding vector for a given target word from **W1**.  
2. Compare it (via dot product or cosine similarity) with all possible context word embeddings from **W2ᵀ**.  
3. This gives a similarity distribution across the vocabulary.

This approach ensures that both **input** and **output** embeddings are preserved separately, instead of collapsing them into a single set.

---

## Tokenization & Data Structure
When a user enters a sentence:
- The sentence is **tokenized**.  
- Each token is stored in a **linked list**, with details like:
  - The token string itself.  
  - Its position in the sentence.  
  - Its location/index in the vocabulary.  

This gives me a structured way to track both **semantic similarity** (from embeddings) and **syntactic position** (from linked list metadata).

---

## Mathematical Formulation

### 1. Input Representation
If the vocabulary size is \\( V \\) and the embedding dimension is \\( D \\):

- Input one-hot vector:
\[
x \in \mathbb{R}^V
\]
(where only the index of the target word is 1).

- First weight matrix (input embeddings):
\[
W_1 \in \mathbb{R}^{V \times D}
\]

- Second weight matrix (output embeddings):
\[
W_2 \in \mathbb{R}^{D \times V}
\]

---

### 2. Forward Pass
The embedding for the input word is:
\[
h = x^\top W_1 \quad \in \mathbb{R}^D
\]

The raw scores for all vocabulary words are:
\[
z = h W_2 \quad \in \mathbb{R}^V
\]

The predicted probability distribution (softmax):
\[
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
\]

---

### 3. Cosine Similarity
For an input word embedding \\( h \\) and an output embedding \\( u_i \\) (column from \\( W_2 \\)):

\[
\text{cosine\_sim}(h, u_i) = \frac{h \cdot u_i}{\|h\| \, \|u_i\|}
\]

I plan to use these similarities together with positional data from the linked list.

---

### 4. Loss Function
The typical loss for training is **cross-entropy** between the predicted distribution \\( \hat{y} \\) and the true context distribution \\( y \\):

\[
L = - \sum_{i=1}^{V} y_i \log(\hat{y}_i)
\]

Where \\( y \\) is a one-hot vector for the actual context word.

---

## Using Embeddings in Practice
My idea is to use **cosine similarity** between:
- The input word embedding (from **W1**), and  
- All vocabulary embeddings (from **W2ᵀ**)  

Together with:
- The token’s position in the user’s prompt, and  
- Its index in the vocabulary.  

The goal is to leverage these two dimensions — **semantic closeness** and **structural position** — to generate responses.

---

## Philosophy
I am not aiming for sudden success or shortcuts.  
My goal is to truly understand how these models work internally, by building them with the bare minimum: C/C++, arrays, and linked lists.  

Even having a **single machine** capable of running a working model is already a big success for me.  
I want to endure the difficulties, be brutally honest with the process, and keep moving forward step by step.

---

## Next Steps
- Continue refining the **Encoder-Decoder pipeline**.  
- Explore how the linked list of tokens can interact with similarity scores to build coherent responses.  
- Experiment with training data to evaluate the usefulness of keeping **W1** and **W2** separate.  
