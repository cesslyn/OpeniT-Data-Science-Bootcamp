# Checkpoint Questions
1. What does a tokenizer produce?
**A tokenizer produces a sequence of tokens such as words, subwords and characters and sequence of IDs or token IDs that convert raw text into a format that the transformer model can understand and process.**

2. Why do transformers need position information?
**Transformers need position information because attention alone does not understand the order of words in a sentence. It is a crucial component that helps to process and understand sequential data effectively.**
3. What is the difference between training and inference?
**Training is the process where the model learns patterns from labeled data and updates its parameters. It involves feature selection, data processing, and model optimization. On the other hand, inference happens after training, where the model uses learned knowledge to make predictions or generate outputs on new data. It applies trained model to real-world data for predictions.**

# Exercise 1A: Label the pipeline
For the sentence "The decoder generated a summary", list a plausible sequence of stages from raw text to model prediction.
Leave room for token ids, embeddings, attention, and output logits.
**The sentence “The decoder generated a summary” is first tokenized into smaller units or token IDs. These token IDs are converted into embeddings, and positional information is added so the model understands  word order. The attention mechanism then allows tokens to interact and gather contextual meaning. After passing through feed-forward layers, the hidden states are processed by the prediction head to produce output logits or predictions.**

# Exercise 2A: Explain the mask
A batch contains one short sentence and one long sentence. Why should padding tokens be masked?
**Padding tokens should be masked because they do not contain meaningful information. If the model attends to padding tokens, it may learn incorrect patterns and reduce overall accuracy. Masking ensures that attention focuses only on valid tokens in the sequence.**

# Exercise 2B: Interpret attention
Suppose the word "it" attends strongly to "model". What relationship might the attention head be capturing?
**If the word “it” attends strongly to “model,” the attention head is likely capturing a reference relationship. The model understands that “it” refers back to the word “model” in the sentence.**

# Reflection
Why is a bidirectional encoder suitable for sentiment or topic classification?
**A bidirectional encoder is suitable for sentiment or topic classification because it can analyze context from both the left and right sides of a word. This allows the model to better understand meaning, tone, and relationships between words within the entire sentence.**

# Exercise 4A
In one paragraph, explain why a decoder should not see future target tokens during training.
**A decoder should not see future target tokens during training because the goal is to predict the next token based only on previous tokes. If future words are visible, the model would simply copy answerss instead of learning proper sequence generation.**

# Exercise 5A
Mark each task as encoder-only, decoder-only, or encoder-decoder: spam classification, article summarization, story continuation, translation, retrieval embedding
**Spam classification - encoder-only**
**Artice Summarozation - encoder-decoder**
**Story continuation - decoder-only**
**Transalation - encoder-decoder**
**Retrieval embedding - encoder-only**

# Exercise 6A
Write a validation rule that checks whether every classification row has a non-empty label and every generation row has a
non-empty target_text.
**A validation rule can check whether classification rows contain a non-empty label and whether generation rows contain a non-empty target_text field. If either field is empty, the row should be flagged or removed before training. Below is the example code:**
if task_type = "encoder_classification" and label == "":
    print("Invalid classificatio row")
if "generation" in task_type and target_text == "":
    print("Invalid generation row")

# Exercise 7A
Describe one risk of using the training target text during inference.
**One risk of using the training target text during inference is data leakage wherein the model simply memorize the exact text it was trained on instead of producing reuslts indepdently. This would make evaluation results unreliable.**

# Mini Project - Encoder Classifier
            Field                                           Your notes
    Model                                 distilbert-base-uncased
    Task and rows used                    encoder_classification 
    Tokenizer settings                    max_length=128, padding=max_length, truncation=True
    Epochs                                5
    Learning rate                         2e-5
    Metric                                Accuracy
    Best result                           0.5
    Most insteresting error               Very short neutral inputs were sometimes classified because of limited training data.

# Appendix A
**Architecture chosen and why**
A **unified transformer setup** was chosen and used with encoder-only models for classification because of their bidirectional context, decoder-only models for autoregressive text generation, and encoder-decoder models for summarization and translation because they support cross-attention between input and output sequences.

**Data fields used**
The data fields used includes an id for tracking, tack_type to route samples to the correct model, split to separate training/validation/test data, input_text as supervised output for generationtasks, and label as the categorical target for classification after encoding.

**Training settings**
Training uses subword tokenization, task-specific attention masking such as bidirectional, causal, or cross-attention, cross-entropy loss for both classification and generation, AdamW optimization, and a warmup-decay learning rate schedule with dropout formregularization, while ensuring only training splits are used.

**Inference examples**
encoder_classification:
Input: "The valisation loss kept increasing after epoch three."
Output: negative

decoder_generation:
Input: "Write one sentence about tokenization."
Output: Tokenization splits text into smaller units such as words or subwords.

encoder_decoder_summarization:
Input: long technical paragraph
Output: condensed summary preserving key meaning

encoder_decoder_translation:
Input: "Hello, how are you?"
Output: Translated equivalent in target language.

**Errors and Improvements**
Main issues include overfitting due to small or limited data, class imbalance affecting classification, and incosistent preprocessing across tasks. The improvements include expanding the dataset, balancing label distribution, standardizing tokenization, and improving batching strategy per task type.