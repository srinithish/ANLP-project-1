# ANLP-project
## Sample predictions
![Result1](https://github.com/srinithish/ANLP-project-1/blob/master/Daily%20Mail%20predictions%20samples.JPG)


# Text Summarization with Deep Learning - README

## Project Overview
This project implements an automatic text summarization system using deep learning that transforms long product reviews into concise, meaningful summaries using an encoder-decoder architecture with attention mechanism and beam search.

## Features
- **Bidirectional LSTM Encoder**: Processes text in both directions for richer context understanding
- **Attention Mechanism**: Allows the model to focus on relevant parts of input when generating summaries
- **GloVe Word Embeddings**: Utilizes pre-trained word representations for better semantic understanding
- **Beam Search**: Generates higher quality summaries by exploring multiple generation paths
- **Automatic Text Preprocessing**: Handles contractions, stop words, and text normalization

## Quick Start

### Prerequisites
```bash
pip install tensorflow numpy pandas nltk beautifulsoup4 scikit-learn contractions
```

### Data Requirements
- CSV file with 'Text' and 'Summary' columns
- GloVe embeddings file (glove.6B.100d.txt)

### Usage
```python
# Set training flag
isTrain = True  # Set to False for inference only

# Run the script
python text_summarization_with_beamSearch_embedding_bidirectional.py
```

## Model Architecture

## Configuration
- **Max Text Length**: 400 words
- **Max Summary Length**: 15 words  
- **Embedding Dimension**: 100
- **Latent Dimension**: 300
- **Beam Width**: 10 candidates

## Technical Details

### Model Components

## Results
The model generates summaries in two modes:
- **Standard Decoding**: Fast generation using greedy selection
- **Beam Search**: Higher quality summaries exploring multiple generation paths

## File Structure
```
project/
├── text_summarization_with_beamSearch_embedding_bidirectional.py
├── Data/
│   ├── Reviews.csv
│   └── glove.6B.100d.txt
└── saved_model_All_with_embeddings_beam_bidirectional/
    └── FullModelWeights.h5
```

---

# Complete Technical Guide

This section provides an in-depth explanation of how the text summarization system works, perfect for understanding the deep learning concepts involved.

### Model Components
```python
# Load product review data
reviews_df = pd.read_csv(reviews_data)
data_df = reviews_df.filter(items=["Summary","Text"])
```

**What happens here:**
- Loads a CSV file containing product reviews with two columns: "Text" (full review) and "Summary" (short summary)
- Removes duplicates and missing values
- This creates pairs of (long text → short summary) for training

### 1.2 Text Cleaning Function
```python
def clean_text(text_string, remove_stop_words=False):
    text_string = text_string.lower()
    tokenized_words = word_tokenize(text_string)
    # Remove stop words, fix contractions
```

**What happens here:**
- Converts text to lowercase
- Splits sentences into individual words (tokenization)
- Removes common words like "the", "and", "is" (stop words)
- Fixes contractions like "don't" → "do not"

### 1.3 Length Filtering
```python
max_text_len = 400
max_summary_len = 15
data_df = data_df[(data_df["Summary_len"] <= max_summary_len) & (data_df["Text_len"] <= max_text_len)]
```

**What happens here:**
- Sets maximum lengths: reviews up to 400 words, summaries up to 15 words
- Filters out longer texts to keep training manageable
- Creates histograms to visualize text lengths

### 1.4 Adding Special Tokens
```python
data_df['Summary'] = data_df['Summary'].apply(lambda x : 'sostok '+ x + ' eostok')
```

**What happens here:**
- Adds "start of summary" (sostok) and "end of summary" (eostok) tokens
- These help the model know when to start and stop generating summaries

## Part 2: Text to Numbers Conversion

### 2.1 Tokenization
```python
x_tokenizer = Tokenizer(num_words=numReqXWords) 
x_tokenizer.fit_on_texts(list(xTrain))
xTrain_seq = x_tokenizer.texts_to_sequences(xTrain)
```

**What happens here:**
- Converts words to numbers (computers can't work with words directly)
- Creates a vocabulary: {"good": 1, "bad": 2, "product": 3, ...}
- Converts sentences to number sequences: "good product" → [1, 3]

### 2.2 Padding
```python
xTrain = pad_sequences(xTrain_seq, maxlen=max_text_len, padding='post')
```

**What happens here:**
- Makes all sequences the same length by adding zeros
- "good product" → [1, 3, 0, 0, 0, ...] (padded to 400 numbers)
- Needed because neural networks require fixed input sizes

## Part 3: Word Embeddings

### 3.1 Loading Pre-trained Embeddings
```python
def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
```

**What happens here:**
- Loads GloVe embeddings (pre-trained word representations)
- Each word becomes a 100-dimensional vector of numbers
- Words with similar meanings have similar vectors
- Example: "good" and "great" would have similar number patterns

## Part 4: The Neural Network Architecture

### 4.1 Encoder (Understanding the Input)
```python
encoder_inputs = Input(shape=(max_text_len,))
enc_emb = Embedding(...)(encoder_inputs)
encoder_lstm1 = Bidirectional(LSTM(latent_dim//2, return_sequences=True))
encoder_outputs, state_h, state_c = encoder_lstm1(enc_emb)
```

**What happens here:**
- **Input Layer**: Receives the padded number sequences
- **Embedding Layer**: Converts numbers back to dense word vectors
- **Bidirectional LSTM**: Reads the text both forwards and backwards
  - Forward: "This product is good" (left to right)
  - Backward: "good is product This" (right to left)
  - Captures context from both directions
- **Output**: Creates a "memory" of the entire input text

### 4.2 Decoder (Generating the Summary)
```python
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(...)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True)
decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
```

**What happens here:**
- **Input**: Takes the summary tokens (one at a time during training)
- **Embedding**: Converts summary words to vectors
- **LSTM**: Generates summary words sequentially
- **Initial State**: Uses the encoder's final memory as starting point

### 4.3 Attention Mechanism
```python
attentionLayer = Attention(use_scale=True)
attn_out = attentionLayer([decoder_lstm_outputs, encoder_outputs])
```

**What happens here:**
- **Problem**: Long texts are hard to remember entirely
- **Solution**: Attention lets the decoder "look back" at specific parts of the input
- When generating "delicious", it focuses on food-related parts of the review
- Like having a highlighter to mark relevant sections while writing

### 4.4 Final Output Layer
```python
decoder_dense = TimeDistributed(Dense(yVocab, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)
```

**What happens here:**
- **Dense Layer**: Converts LSTM output to vocabulary probabilities
- **Softmax**: Makes probabilities sum to 1
- **Output**: For each position, gives probability of each word in vocabulary

## Part 5: Training Process

### 5.1 Model Compilation
```python
FullModel.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
```

**What happens here:**
- **Optimizer**: Algorithm to adjust model weights (RMSprop)
- **Loss Function**: Measures how wrong the predictions are
- **Goal**: Minimize the difference between predicted and actual summaries

### 5.2 Training Loop
```python
history = FullModel.fit([xTrain, yTrain[:,:-1]], yTrain[:,1:], epochs=50)
```

**What happens here:**
- **Teacher Forcing**: During training, gives correct previous words
- **Input**: Full review + summary without last word
- **Target**: Summary without first word (shifted by one position)
- **Learning**: Model learns to predict next word given context

## Part 6: Inference (Making Predictions)

### 6.1 Separate Models for Prediction
```python
encoder_model_inf = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
decoder_model_inf = Model([decoder_inputs] + [...], [decoder_outputs_inf] + [...])
```

**What happens here:**
- **Encoder Model**: Processes input text once, outputs memory
- **Decoder Model**: Generates summary word by word
- **Separation**: Needed because inference works differently than training

### 6.2 Sequential Generation
```python
def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model_inf.predict(input_seq)
    target_seq[0, 0] = target_word_index['sostok']
    
    while not stop_condition:
        output_tokens, new_h, new_c = decoder_model_inf.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
```

**What happens here:**
1. Encode the full input text into memory
2. Start with "sostok" (start token)
3. Generate next word based on current context
4. Use that word to generate the next word
5. Continue until "eostok" (end token) or max length

## Part 7: Beam Search (Advanced Generation)

### 7.1 Beam Search Algorithm
```python
def decode_beam_sequence(input_seq):
    sampled_token_index = list(np.argpartition(-output_tokens, 10, axis=-1)[0,-1,:10])
    # Keep top 10 candidates at each step
```

**What happens here:**
- **Problem**: Greedy selection (always picking best word) can lead to poor overall summaries
- **Solution**: Keep track of multiple possible sequences simultaneously
- **Process**: 
  1. Generate top 10 words for first position
  2. For each of those, generate top 10 words for second position
  3. Keep overall top 10 sequences
  4. Continue until completion
- **Result**: Better quality summaries by considering multiple paths

## Key Concepts Explained

### What is an LSTM?
- **Long Short-Term Memory**: A type of neural network that can remember information over long sequences
- **Problem it solves**: Regular networks forget earlier parts of long texts
- **How it works**: Has "gates" that decide what to remember and forget
- **Bidirectional**: Reads text both forwards and backwards for better context

### What is Attention?
- **Purpose**: Helps the model focus on relevant parts of input when generating each word
- **Analogy**: Like looking back at notes while writing an essay
- **Mechanism**: Calculates which input words are most relevant for current output word
- **Benefit**: Handles long texts better than just using final encoder state

### What are Embeddings?
- **Purpose**: Convert words to numerical vectors that capture meaning
- **Example**: "king" - "man" + "woman" ≈ "queen"
- **Training**: Pre-trained on large text corpora to learn word relationships
- **Dimension**: Each word becomes a 100-dimensional vector of numbers

## Model Architecture Flow

```
Input Text → Tokenization → Padding → Embedding → Bidirectional LSTM (Encoder)
                                                            ↓
                                                    [Memory States]
                                                            ↓
Start Token → Embedding → LSTM (Decoder) ← Attention ← Encoder Memory
     ↓                        ↓               ↑
Next Word ← Dense Layer ← Concatenate ←———————┘
     ↓
Continue until End Token
```

## Summary of the Complete Process

1. **Data Preparation**: Clean and tokenize text, create word-to-number mappings
2. **Model Architecture**: Build encoder-decoder with attention mechanism
3. **Training**: Teach model to predict summaries from full texts
4. **Inference**: Generate summaries word by word using trained model
5. **Beam Search**: Improve quality by considering multiple generation paths

This system essentially learns to "read" long texts and "write" short summaries by training on thousands of examples, similar to how humans learn to summarize by reading many texts and their corresponding summaries.
