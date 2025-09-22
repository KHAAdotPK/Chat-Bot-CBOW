#### Exploring the nuances of the use of a transposed matrix (W2.T) for matching embeddings.

Which matrix corresponds to “target” vs “context” depends on the training variant:
- In **skip-gram**: input = target word → W1 encodes target-role embeddings and W2 encodes context-role embeddings.
- In **CBOW**: input = (surrounding) context → W1 encodes context-role embeddings and W2 encodes target-role embeddings.

By training both matrices together we teach the model two complementary skills:
1. Given a word in its input role, what contexts typically surround it? (W1 (target) → W2 (context) comparisons)  
2. Given a context (or hidden vector built from context words), which target/center words typically fit? (W1 (context) → W2 (target) comparisons).

Using the embeddings from `W1` and matching them against the transposed `W2` is a perfectly valid approach. Here’s a deeper look at what that means in the context of **CBOW**.

---
##### The Two Sets of Embeddings:
In a **CBOW** (and **Skip-gram**) model, every word in vocabulary actually learns two different vector representations:
1. **The Input Vector** (**from** `W1`): This vector represents a word when it's used as part of the context (an input word used to predict the target). The matrix `W1` contains all of these input vectors.
2. **The Output Vector** (**from** `W2`): This vector represents a word when it's the target word that the model is trying to predict. The matrix `W2` contains these output vectors. By transposing `W2`, we align it so that each row corresponds to a word's output vector, just like in `W1`.

So, when we take a word's vector from `W1` and find its similarity with all the vectors from the transposed `W2`, we are essentially asking: "When this word appears in the context, how well does it predict every other potential target word?"

---

##### Why This Is Interesting?

- **Different Perspectives**: We are comparing a word's ability to be a context with its ability to be a target finder as well. Both sets of vectors capture semantic meaning, but from slightly different perspectives.
- **Common Practice**: While both `W1` and `W2` are learned, most practitioners simply use the **input vectors** (`W1`) as the final word embeddings for downstream tasks. They are generally considered to be of higher quality.
- **A More Robust Approach**: Some research suggests that the best single vector representation for a word can be found by **averaging or summing** its input vector (from `W1`) and its output vector (from the transposed `W2`). This combines both "perspectives" into a single, potentially more robust embedding.

---

Practical notes & suggestions

- If you use W1 vs W2ᵀ similarity as your ranking signal (like we already do), either use dot product (for raw model scores) or cosine similarity (for semantic similarity). If you use cosine, normalize vectors first.

- Consider storing both W1 and W2 transposed in the program for efficient matrix multiplications.

- If you want a single, stable embedding for downstream tasks, use embedding = (W1 + W2ᵀ) / 2 — many practitioners do this. But keeping both gives more flexibility (as we prefered).

---

##### Exta:

1. W1 holds input-role embeddings; W2 holds output-role embeddings. Which role is “context” vs “target” depends on whether you run skip-gram (input=target) or CBOW (input=context).
2. CBOW/skip-gram are about predicting a center/target word from context (CBOW) or predicting context words from a target (skip-gram). Use “target/center word”.

We understand how the model's “brain” works: it stores two complementary sets of word relationships in the weights.

- **W1 — input embeddings.** W1 captures how a word behaves when used as input to the model. In other words: “When we see word X as an input, what kinds of surrounding words / contexts are associated with it?”  
- **W2 — output embeddings.** W2 captures how a word behaves when it is the prediction target (the center/target word). In other words: “Given a context or hidden representation, which word is most likely to be the center/target?”

Note: whether W1 corresponds to the *target* or the *context* depends on the training variant (skip-gram vs CBOW). In skip-gram the model predicts context words from a given target, while in CBOW it predicts the target from context words. Regardless, the two matrices are trained together and complement each other: W1 encodes words in their input role, and W2 encodes words in their output role. By tuning both matrices we refine two complementary skills — mapping words → contexts and mapping contexts → words — which is exactly the behavior we want for building a domain-specific retrieval or response system.


