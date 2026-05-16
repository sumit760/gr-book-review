---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- genre-classification
- goodreads
- book-review-classification
- distilroberta
- roberta
- transformers
- kaggle
- wandb
base_model: distilroberta-base
model-index:
- name: distilroberta-base-goodreads-genres
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Goodreads Book Review / Genre Classification Dataset
      type: goodreads
    metrics:
    - type: accuracy
      value: 0.61583
      name: Accuracy
    - type: f1
      value: 0.61632
      name: F1 Score
    - type: loss
      value: 2.66787
      name: Eval Loss
---

# distilroberta-base-goodreads-genres

## Model Overview

This model is a fine-tuned version of `distilroberta-base` for Goodreads book review / genre classification. It is designed to classify book-related text, such as book reviews, descriptions, or summaries, into genre categories.

The model was developed as part of an MLOps assignment using Hugging Face Transformers, Kaggle Notebook, Hugging Face Hub, and Weights & Biases for experiment tracking.

## Model Details

- **Model name:** `distilroberta-base-goodreads-genres`
- **Base model:** `distilroberta-base`
- **Model type:** Transformer-based sequence classification model
- **Task:** Text classification
- **Domain:** Goodreads book reviews / genre classification
- **Language:** English
- **Library:** Hugging Face Transformers
- **Training platform:** Kaggle Notebook
- **Experiment tracking:** Weights & Biases
- **Model repository:** https://huggingface.co/sumitp76/distilroberta-base-goodreads-genres

## Important Links

- **Kaggle Notebook:** https://www.kaggle.com/code/sumitpiitj/gr-book-review/edit
- **Hugging Face Model:** https://huggingface.co/sumitp76/distilroberta-base-goodreads-genres
- **W&B Project Dashboard:** https://wandb.ai/sumit-k-pal-76-iitj/mlops-assignment2/table?nw=nwusersumitkpal76

### Inference Workflow

The inference workflow describes how the trained model is used to predict the genre of a new book review or description.

## Step-by-step Inference Process

1. **Input text is provided**

   A user provides a book review, summary, or description as input text.

   Example:

   ```text
   A young wizard discovers his magical powers and enters a hidden world of adventure.
   ```

2. **Load the trained model and tokenizer**

   The trained model and tokenizer are loaded directly from the Hugging Face Hub.

   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification

   model_id = "sumitp76/distilroberta-base-goodreads-genres"

   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForSequenceClassification.from_pretrained(model_id)
   ```

3. **Tokenize the input text**

   The tokenizer converts the raw text into numerical tokens that the model can understand.

   ```python
   text = "A young wizard discovers his magical powers and enters a hidden world of adventure."

   inputs = tokenizer(
       text,
       return_tensors="pt",
       truncation=True,
       padding=True
   )
   ```

4. **Generate model prediction**

   The tokenized input is passed to the model. The model returns logits, which are raw prediction scores for each genre class.

   ```python
   import torch

   with torch.no_grad():
       outputs = model(**inputs)

   logits = outputs.logits
   ```

5. **Convert logits into predicted class**

   The class with the highest score is selected as the predicted genre.

   ```python
   predicted_class_id = torch.argmax(logits, dim=1).item()
   predicted_label = model.config.id2label[predicted_class_id]

   print("Predicted genre:", predicted_label)
   ```

6. **Return the predicted genre**

   The final output is the predicted Goodreads genre label.

### Simple Inference Using Pipeline

The easiest way to use the model is with the Hugging Face `pipeline`.

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="sumitp76/distilroberta-base-goodreads-genres"
)

text = "A young wizard discovers his magical powers and enters a hidden world of adventure."

result = classifier(text)

print(result)
```

### Inference Output

The output will contain the predicted label and confidence score.

Example:

```text
[{'label': 'Fantasy', 'score': 0.92}]
```

The exact label name depends on the label mapping saved in the model configuration.


## Setup Instructions

### 1. Clone or open the project

The training was performed in a Kaggle Notebook.

Kaggle Notebook:

https://www.kaggle.com/code/sumitpiitj/gr-book-review/edit

### 2. Install dependencies

Install the required Python libraries:

```bash
pip install transformers
pip install datasets
pip install evaluate
pip install accelerate
pip install huggingface_hub
pip install wandb
pip install scikit-learn
pip install pandas
pip install numpy
pip install torch
```

In Kaggle, many packages may already be installed. If needed, install missing packages inside a notebook cell:

```bash
!pip install transformers datasets evaluate accelerate huggingface_hub wandb scikit-learn
```

### 3. Set up Huggging Face token

To push the model to Hugging Face Hub, create a Hugging Face access token with Write permission.

In Kaggle:

Go to Add-ons
Open Secrets
Add your Hugging Face token
Save it using the name:

```bash
HF_TOKEN
```

Then load it in the notebook:

```bash
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

login(token=HF_TOKEN)
```

### 4. Set up W&B tracking

Log in to Weights & Biases:

```bash
import wandb

wandb.login()
```

Initialize a W&B run:

```bash
wandb.init(
    project="mlops-assignment2",
    name="distilroberta-base-goodreads-genres"
)
```

W&B dashboard:

```bash
https://wandb.ai/sumit-k-pal-76-iitj/mlops-assignment2/table?nw=nwusersumitkpal76
```

### 5. Train the model

The model was trained using Hugging Face Trainer.

General training flow:

```bash
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = "distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)
```

Tokenize the dataset:

```bash
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

Train using Trainer:

```bash
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```

### 6. Evaluate the model

After training, evaluate the model:

```bash
results = trainer.evaluate()
print(results)
```

### 7. Push the model to Hugging Face Hub

```bash
repo_id = "sumitp76/distilroberta-base-goodreads-genres"

model.push_to_hub(repo_id, token=HF_TOKEN)
tokenizer.push_to_hub(repo_id, token=HF_TOKEN)
```

Model link:

```bash
https://huggingface.co/sumitp76/distilroberta-base-goodreads-genres
```

## How to use the model

You can use the model directly with the Hugging Face pipeline.

```bash
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="sumitp76/distilroberta-base-goodreads-genres"
)

text = "A young wizard discovers his magical powers and enters a hidden world of adventure."

result = classifier(text)

print(result)
```

## Training Details

### Training Platform

The model was trained on Kaggle Notebook.

Platform: Kaggle
Notebook link: https://www.kaggle.com/code/sumitpiitj/gr-book-review/edit
Framework: Hugging Face Transformers
Experiment tracking: Weights & Biases
Model hosting: Hugging Face Hub

### Base Model

The base model used was:

```bash
distilroberta-base
```

distilroberta-base is a smaller and faster version of RoBERTa. It is suitable for text classification tasks where a balance between performance and efficiency is required.

### Preprocessing Steps

The general preprocessing steps included:

1. Loading the Goodreads book review / genre dataset
2. Checking and cleaning missing values
3. Preparing the input text column
4. Encoding genre labels into numeric IDs
5. Splitting the dataset into training and evaluation sets
6. Tokenizing text using the distilroberta-base tokenizer
7. Applying truncation and padding
8. Training the model using Hugging Face Trainer


### Training Configuration

Update the values below according to the final notebook settings:

| Parameter           | Value                   |
| ------------------- | ----------------------- |
| Base model          | `distilroberta-base`    |
| Task                | Text Classification     |
| Optimizer           | AdamW                   |
| Loss function       | Cross-entropy loss      |
| Training/Eval Batch size          | 16/32          |
| Learning rate       | 2e-5       |
| Number of epochs    | 6              |
| Max sequence length | 256 |
| Evaluation strategy | steps |

### Results

| Metric    | Score |
| --------- | ----: |
| Accuracy  |  0.61583 |
| F1 Score  |  0.61632 |
| Eval Loss |  2.66787 |

### Result Link

Kaggle Notebook: https://www.kaggle.com/code/sumitpiitj/gr-book-review/edit
Hugging Face model: https://huggingface.co/sumitp76/distilroberta-base-goodreads-genres
W&B dashboard: https://wandb.ai/sumit-k-pal-76-iitj/mlops-assignment2/table?nw=nwusersumitkpal76











