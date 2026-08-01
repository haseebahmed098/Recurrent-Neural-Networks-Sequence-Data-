#  Recurrent Neural Networks & Sequence Data
## Complete Beginner's Guide — Movie Review Sentiment Classification with LSTM/GRU

---

## 1. Project Overview

**What is the project about?**
You are building a neural network that reads a movie review (as plain text) and predicts whether the review is **positive** or **negative**. This is called **sentiment analysis**, and it is one of the classic first projects for learning how to handle *sequence data* (text, time series, audio) with deep learning.

**What problem does it solve?**
Computers don't understand words — they understand numbers. And a sentence isn't just a "bag" of words; word *order* carries meaning ("not good" vs "good"). Regular neural networks (Dense layers) or CNNs (from your last lab) don't naturally track order or handle variable-length input well. This project solves that by using a **Recurrent Neural Network (RNN)**, specifically an **LSTM (Long Short-Term Memory)** layer, which is purpose-built to process sequences one step at a time while remembering what came before.

**Main objective**
Train a model on the IMDB Movie Reviews dataset (50,000 labeled reviews built into Keras) that takes a sequence of word-index integers and outputs a single number between 0 and 1 representing "how positive" the review is.

---

## 2. Purpose of the Project

**Why is this project important?**
It's the standard entry point into **Natural Language Processing (NLP)** with deep learning. Almost every real NLP system — spam filters, chatbots, translation tools, review analyzers — starts from the same core idea taught here: turn text into numbers, feed it through a sequence-aware layer, and classify or generate from the result.

**Real-world use**
- E-commerce platforms auto-flagging negative product reviews for follow-up
- Social media monitoring (brand sentiment tracking)
- Customer support ticket triage (urgent/angry vs calm)
- Financial news sentiment feeding into trading signals
- App store review analysis for product teams

---

## 3. Learning Outcomes

By completing this lab you will learn:

- **Text preprocessing for neural networks**: tokenizing text into integer sequences, padding/truncating to a fixed length
- **Embedding layers**: how to turn sparse word-index integers into dense, meaningful vectors
- **Recurrent architectures**: SimpleRNN, LSTM, and GRU — what each does differently and why LSTM/GRU exist
- **The vanishing gradient problem** and why it matters for long sequences
- **Binary classification with Keras**: compiling with `binary_crossentropy`, using a sigmoid output
- **Model training workflow**: `fit()`, validation splits, epochs, batch size
- **Evaluating and visualizing** training curves (loss/accuracy over epochs)
- **Error analysis**: inspecting *what* the model gets wrong and reasoning about *why*
- **Experimental comparison**: systematically varying an architecture (RNN cell type) or hyperparameter (sequence length) and reporting the trade-offs

**Career relevance**
Sequence modeling is foundational to modern AI — the same ideas (sequences, embeddings, attention over context) scale up directly into Transformers and LLMs. Recruiters and interviewers commonly expect familiarity with this workflow for any ML/AI role touching text, time-series, or speech data.

---

## 4. Importance of the Project

**Why is it valuable?**
It teaches the *general recipe* for sequence problems: represent → embed → recur → classify/predict. That recipe applies whether the sequence is a sentence, a sensor reading over time, a DNA strand, or a click-stream.

**Industry use**
- **Retail/E-commerce**: automated review summarization and sentiment dashboards
- **Finance**: sentiment-driven risk models from news/earnings calls
- **Healthcare**: clinical note classification, patient sentiment in feedback
- **Media**: content moderation, comment toxicity detection
- **Customer service**: intent detection in chat/email triage

---

## 5. Complete Development Guide (Step-by-Step)

### Step 0 — Environment
Run this in **Google Colab** (recommended — free GPU, and the IMDB dataset downloads automatically). Tools used:

| Tool/Library | Why it's used |
|---|---|
| **TensorFlow / Keras** | High-level deep learning API; has the IMDB dataset built in and simple RNN/LSTM/GRU layers |
| **NumPy** | Array manipulation (padding, indexing) |
| **Matplotlib** | Plotting loss/accuracy curves |
| **Google Colab** | Free hosted notebook with optional GPU acceleration, which matters because RNNs train slower than CNNs/Dense nets |

### Step 1 — Load the Data
```python
from tensorflow import keras
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
```
This loads 25,000 training + 25,000 test reviews. Each review is **already** converted into a list of integers, where each integer is a word's rank in frequency (e.g., 4 = the 4th most common word). `num_words=10000` keeps only the 10,000 most frequent words — rarer words are dropped, which keeps the vocabulary (and the model) manageable.

### Step 2 — Decode a Review (sanity check)
Keras also gives you a `word_index` dictionary (`word → index`). Reversing it lets you turn integers back into readable text, which is useful to **verify your data looks right** before trusting a model built on top of it.

### Step 3 — Pad the Sequences
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = 200
X_train_padded = pad_sequences(X_train, maxlen=maxlen)
```
Reviews have different lengths (a few words to thousands). Neural networks need a **fixed-size input**, so every review is padded with zeros at the front (or truncated) to exactly 200 tokens.

### Step 4 — Build the Model
```python
model = keras.Sequential([
    keras.layers.Input(shape=(maxlen,)),
    keras.layers.Embedding(input_dim=vocab_size, output_dim=32),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation="sigmoid")
])
```
- **Embedding(10000, 32)**: turns each word-index into a learned 32-number vector. Similar words end up with similar vectors after training (this is *far* better than treating word IDs as raw numbers, which would falsely imply that word 500 is "twice" word 250).
- **LSTM(32)**: reads the sequence of 200 embedding vectors one step at a time, maintaining a hidden state (short-term memory) and a cell state (long-term memory), and outputs a single 32-number summary vector for the whole review.
- **Dense(1, sigmoid)**: turns that summary into one number between 0 and 1 — the predicted probability the review is positive.

### Step 5 — Compile and Train
```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train_padded[:10000], y_train[:10000],
                     epochs=5, batch_size=64, validation_split=0.2)
```
- **Adam optimizer**: adaptive learning rate, a solid default for most deep learning tasks.
- **binary_crossentropy**: the standard loss for two-class (0/1) classification.
- **validation_split=0.2**: holds back 20% of training data each epoch to check for overfitting.

### Step 6 — Visualize Training
Plotting `history.history["loss"]` / `"accuracy"` (train vs validation) tells you whether the model is still improving, has converged, or is starting to overfit (validation loss rising while training loss keeps falling).

### Step 7 — Evaluate and Predict
```python
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
```
Evaluating on the **held-out test set** (never seen during training) is the real measure of how well the model generalizes.

### Step 8 — Experiment: GRU vs LSTM vs SimpleRNN, and sequence length
Swap `layers.LSTM(32)` for `layers.GRU(32)` or `layers.SimpleRNN(32)` — everything else stays the same. This is the easiest way to directly compare cell types on identical data. Similarly, re-running with `maxlen=50/100/200` shows the accuracy vs. training-time trade-off of longer context.

### Step 9 — Predict on a Custom Review
To classify your own sentence, you must preprocess it **exactly the same way** as the training data:
1. Lowercase and split into words
2. Map each word through `word_index`, adding 3 to match Keras's offset (0,1,2 are reserved for padding/start/unknown)
3. Use `2` for any word not found in the vocabulary
4. Pad to the same `maxlen` used in training
5. Call `model.predict()`

### Step 10 — Error Analysis
Find test reviews where `predicted != actual`, decode them back to text, and read them. This builds the crucial ML skill of **looking at your errors**, not just your aggregate accuracy number.

---

## 6. Project Architecture

```
Raw Review Text
      │
      ▼
[Keras IMDB dataset]  →  Already-tokenized integer sequences (word → index)
      │
      ▼
[pad_sequences]  →  Fixed-length sequences (maxlen=200)
      │
      ▼
[Embedding layer]  →  Each integer → dense 32-dim vector
      │
      ▼
[LSTM / GRU / SimpleRNN layer]  →  Reads sequence step-by-step,
                                     outputs one summary vector (32-dim)
      │
      ▼
[Dense(1, sigmoid)]  →  Single probability (0 = negative, 1 = positive)
      │
      ▼
Prediction: "Positive" or "Negative"
```

**How components interact:**
The Embedding layer's output (a matrix of shape `[200, 32]` per review) feeds directly into the recurrent layer, which internally loops over the 200 timesteps, updating its hidden state at each step using the current word vector + previous hidden state. Only the *final* hidden state (after seeing the whole review) is passed to the Dense output layer — this is the "many-to-one" pattern described in the lab PDF.

---

## 7. Code Explanation (Key Logic)

- **`keras.datasets.imdb.load_data(num_words=10000)`** — loading with a vocabulary cap keeps the Embedding layer's size manageable (10,000 × 32 ≈ 320K parameters instead of, say, millions).
- **`pad_sequences(X, maxlen=200)`** — by default pads/truncates from the *front* (`padding='pre'`, `truncating='pre'`), which in practice tends to preserve the more informative ending of a review when truncating.
- **`layers.LSTM(32)`** — the `32` is the number of hidden units (the size of the memory vector); more units = more capacity but slower training and more overfitting risk on a small dataset.
- **`model.fit(..., validation_split=0.2)`** — Keras automatically carves out the last 20% of the training array as validation data; it does **not** shuffle before splitting, so if your data were sorted by label this would be a problem (IMDB is not, so this is safe here).
- **`history.history`** — a dictionary Keras returns after training containing arrays of loss/accuracy per epoch, used directly for plotting.
- **`sample_predictions[i][0] > 0.5`** — converts the raw sigmoid probability to a binary label using the standard 0.5 decision threshold.

---

## 8. Best Practices

**Common mistakes to avoid**
- Forgetting to pad/truncate to the *same* `maxlen` used in training when predicting on new text
- Comparing GRU vs LSTM vs SimpleRNN on different data subsets/epochs (invalidates the comparison — keep everything else fixed)
- Reading only the final test accuracy without checking the loss curves for overfitting
- Using `validation_split` on non-shuffled, label-sorted data (not an issue here, but a classic bug elsewhere)
- Setting `maxlen` far larger than needed, which wastes compute on mostly-padding input

**Coding standards / optimization tips**
- Use `EarlyStopping` (`keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)`) so you don't have to guess the right number of epochs
- Add `Dropout` (e.g., after the Embedding or LSTM layer) to reduce overfitting
- Use `layers.Bidirectional(layers.LSTM(32))` to let the model read the sequence both forwards and backwards for richer context
- Batch size 64–128 is a good default trade-off between speed and gradient stability

**Security & performance considerations**
- If deploying, never `eval()` or run user-submitted text as code — this project only *reads* text as data, so this is low-risk, but always sanitize input length (cap review length server-side to avoid extremely long inputs slowing inference)
- Cache the tokenizer/word index — rebuilding it per request is wasteful
- For production, save the model with `model.save("model.keras")` and load once at server startup, not per-request

---

## 9. Project Improvements

- **Use pretrained embeddings** (GloVe, Word2Vec) instead of training the Embedding layer from scratch — helps a lot on small datasets
- **Add Bidirectional LSTM** layers for better context capture
- **Stack recurrent layers** (2+ LSTM layers) for more capacity on harder tasks
- **Try attention mechanisms** or a small Transformer encoder — the modern successor to RNNs for NLP
- **Multi-class sentiment** (1–5 stars) instead of just binary positive/negative
- **Deploy as an API** (Flask/FastAPI) with a simple web form for real-time predictions
- **Add explainability**: highlight which words most influenced the prediction (e.g., via attention weights or gradient-based saliency)
- **Handle out-of-vocabulary words better** with subword tokenization (e.g., WordPiece/BPE) instead of whole-word indexing

---

## 10. Challenges and Solutions

| Challenge | Solution |
|---|---|
| Training is much slower than CNNs from the previous lab | RNNs process sequences step-by-step (can't fully parallelize over time); use a smaller data subset while experimenting, a GPU runtime in Colab, and only train fully once your code is correct |
| Validation accuracy stalls or drops while training accuracy keeps rising | Classic overfitting — add Dropout, reduce model size, use EarlyStopping, or get more data |
| Vanishing gradients with SimpleRNN on long sequences | This is *expected* — it's exactly why LSTM/GRU exist; use them for longer sequences |
| Custom review prediction gives strange results | Almost always a preprocessing mismatch — double check the `+3` offset, the `maxlen`, and that unknown words map to `2` |
| Kernel/runtime crashes due to memory | Reduce `batch_size`, `maxlen`, or the training subset size |

---

## 11. Testing and Evaluation

- **Quantitative**: test accuracy from `model.evaluate()` on the untouched test set (target: comfortably above 80% is achievable even with a simple single-LSTM model on a 10k-review subset; using the full 25k training set typically pushes this higher)
- **Qualitative**: manually read a handful of predictions (Example 4 in the notebook) and the misclassified reviews (Task 4) to sanity-check the model isn't just exploiting a shortcut
- **Comparative**: the RNN-type comparison table (Task 2) and sequence-length experiment (Task 3) are themselves a form of evaluation — they test whether your conclusions ("LSTM > SimpleRNN for this task", "longer maxlen helps up to a point") hold up empirically rather than by assumption
- **Sanity checks**: confirm `model.summary()` parameter counts make sense (Embedding should dominate: `10000 × 32 = 320,000` params) and that loss decreases from epoch 1

---

## 12. Real-World Applications

- **E-commerce review triage** (Amazon, Yelp-style platforms) — auto-flag negative reviews for customer service follow-up
- **Social media brand monitoring** — track public sentiment about a product launch in near real-time
- **Call-center transcript analysis** — detect frustrated customers and escalate
- **Financial sentiment analysis** — gauge market mood from news headlines or earnings call transcripts
- **App store / product feedback analytics** for product management teams
- **Healthcare** — patient satisfaction survey analysis at scale

---

## 13. Conclusion

This lab walks you through the complete, standard pipeline for text classification with recurrent neural networks: turning raw text into padded integer sequences, embedding those integers into meaningful vectors, letting an LSTM (or GRU/SimpleRNN) read the sequence while maintaining memory across it, and producing a single sentiment prediction. Along the way you directly compare recurrent cell types and sequence lengths, and practice the essential skill of error analysis.

**Key takeaways**
- Sequence order matters, and RNNs (especially LSTM/GRU) are architected specifically to exploit it
- Embeddings turn sparse word IDs into meaningful, trainable vector representations
- LSTM/GRU solve the vanishing-gradient limitation of vanilla RNNs
- Always evaluate on a held-out test set, and always look at your model's actual mistakes

**Where to go next**
- Learn **Bidirectional RNNs** and **attention mechanisms**
- Study the **Transformer** architecture (the direct successor to RNNs in modern NLP, and the foundation of models like GPT/Claude)
- Try a real deployment: wrap this model behind a small web API
- Explore Hugging Face's pretrained NLP models for a taste of transfer learning in text

---

*Companion file: `Lab_14_RNN_Solved.ipynb` contains the fully completed, ready-to-run notebook implementing every exercise and task described above.*
