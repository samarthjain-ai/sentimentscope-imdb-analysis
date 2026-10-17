# SentimentScope — IMDB Sentiment Analysis

A Transformer-based Natural Language Processing (NLP) project for classifying movie reviews as positive or negative using PyTorch.

## About the Project

SentimentScope explores how Transformer architectures can understand text and perform binary sentiment classification.

I worked on this project as part of my AI/ML Nanodegree learning journey, gaining practical experience with text preprocessing, tokenization, PyTorch dataset creation, Transformer components, model training, and evaluation.

The project uses the IMDB movie-review dataset and a custom Transformer-based neural network to predict the sentiment of movie reviews.

## Objectives

- Load and prepare the IMDB movie-review dataset.
- Explore review data using descriptive statistics and visualizations.
- Tokenize text using the `bert-base-uncased` tokenizer.
- Implement a custom PyTorch Dataset and DataLoaders.
- Understand Transformer components, including attention and feed-forward layers.
- Train a neural network for positive and negative sentiment classification.
- Evaluate model performance on validation and test data.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Pandas
- Matplotlib
- Jupyter Notebook
- Natural Language Processing (NLP)

## Model Architecture

The notebook explores a custom Transformer architecture with components including:

- Token embeddings
- Self-attention and multi-head attention
- Feed-forward neural networks
- Layer normalization and residual connections
- A classification output layer

The `bert-base-uncased` tokenizer is used for text tokenization. The classification architecture is implemented separately in PyTorch.

## Dataset

The project uses the IMDB movie-review dataset, containing positive and negative reviews.

The notebook is designed to prepare:
- Training data
- Validation data
- Test data

The expected dataset archive is `aclImdb_v1.tar.gz`. The dataset must be obtained separately and extracted according to the notebook's instructions.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/samarthjain-ai/sentimentscope-imdb-analysis.git
cd sentimentscope-imdb-analysis
```

### 2. Set up the environment

Install Python and Jupyter Notebook, then install the required libraries:

```bash
pip install torch transformers pandas matplotlib numpy
```

### 3. Prepare the dataset

Place `aclImdb_v1.tar.gz` in the appropriate working directory and extract it using:

```bash
tar -xzf aclImdb_v1.tar.gz
```

Ensure the extracted dataset contains the expected training and testing review folders.

### 4. Run the notebook

Open `SentimentScope_starter.ipynb` in Jupyter Notebook or JupyterLab and execute the cells in order.

A GPU is recommended for Transformer training.

## Learning Outcomes

- Text preprocessing and tokenization
- Exploratory data analysis
- Custom Dataset implementation in PyTorch
- Transformer and attention mechanisms
- Neural network training and evaluation
- Sentiment classification using NLP

## Project Status

This repository documents my work and learning on the project. Training results and final test accuracy should be reported only after running the complete notebook successfully.

## Author

**Samarth Jain**

- GitHub: https://github.com/samarthjain-ai
- LinkedIn: https://www.linkedin.com/in/samarth-jain-a90b2337b
