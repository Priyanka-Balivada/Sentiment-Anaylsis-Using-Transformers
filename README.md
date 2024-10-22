## Reproducibility
The algorithms and methods used in this research were implemented using pre-trained BERT and ELECTRA models from Hugging Face. These models were fine-tuned on a tweet sentiment dataset from https://huggingface.co/datasets/cardiffnlp/tweet_eval. The notebooks provided contain all the necessary code to replicate the results.

**Steps to Reproduce:**
- Clone the repository.
- Install the required dependencies listed in requirements.txt.
- Open the Jupyter notebooks bert_tweet_eval.ipynb and electra_tweet_eval.ipynb.
- Run the cells in sequence to replicate the experiments.
- Ensure GPU support is available for faster model training.
- WandB Setup:
  - When prompted for WandB (Weights and Biases) login.
  - Log in using your API key from your WandB account.

## Algorithms and Code
**BERT Implementation:**
- The BERT model is implemented in bert_tweet_eval.ipynb.
- **Model**: BertForSequenceClassification is loaded using the Hugging Face transformers library.
- **Training**: The model is trained using the Hugging Face Trainer API with hyperparameters like a learning rate of 2e-5, batch size of 16, and 1, 3, 5 epochs.
- **Evaluation**: The model is evaluated on test data using accuracy, precision, recall, and F1 scores.

**ELECTRA Implementation:**
- The ELECTRA model is implemented in electra_tweet_eval.ipynb.
- **Model**: ElectraForSequenceClassification is used for sentiment classification.
- **Training**: Trained similarly to BERT but with 5 epochs for better convergence.
- **Evaluation**: The model's performance is evaluated using the same metrics as BERT.

## Proof of Results
The trained models have been published on Hugging Face. You can view and download the fine-tuned models below:

**BERT Models:**
- 1 Epoch (https://huggingface.co/Priyanka-Balivada/bert-1-epoch-sentiment)
- 3 Epochs (https://huggingface.co/Priyanka-Balivada/bert-3-epoch-sentiment)
- 5 Epochs (https://huggingface.co/Priyanka-Balivada/bert-5-epoch-sentiment)
  
**ELECTRA Models:**
- 1 Epoch (https://huggingface.co/Priyanka-Balivada/electra-1-epoch-sentiment)
- 3 Epochs (https://huggingface.co/Priyanka-Balivada/electra-3-epoch-sentiment)
- 5 Epochs (https://huggingface.co/Priyanka-Balivada/electra-5-epoch-sentiment)

## Materials & Methods
**Computing Infrastructure:**
- Operating System: Windows 64-bit operating system
- Hardware: 8GB RAM, Intel Core i5 CPU
- Software: Python>=3.8, Jupyter Notebooks, Hugging Face Transformers, PyTorch

## Conclusions
Both BERT and ELECTRA models performed well on the tweet dataset, with ELECTRA showing marginally better accuracy in sentiment classification. The transformer-based architectures effectively capture the sentiment from short, noisy texts.

## Limitations
- **Data Size**: The dataset may not be representative of all tweet sentiments, especially underrepresented categories.
- **Model Fine-Tuning**: Further hyperparameter tuning could yield better performance.
- **Generalizability**: The model may not perform as well on non-English tweets or other social media platforms due to domain differences.
