# Intelligent-Software-Engineering-Coursework-made-by-Jimiao-Yu

A comparative study of text classification implementations using different deep learning frameworks.

## Requirements

### SVM Implementation
```bash
Python 3.8+
scikit-learn==1.2.2
imbalanced-learn==0.10.1
nltk==3.8.1
pandas==2.0.3
numpy==1.24.3
```

### BERT Implementation
```bash
Python 3.8+
torch==2.0.1
transformers==4.30.2
tensorflow==2.12.0
keras==2.12.0
mxnet-cu118==1.9.1
```

## Usage

### SVM Model

```bash
# Run SVM implementation （Please modify the code in the load_data section to read the correct data)
python br_classification_optimized.py

# Expected Output Format:
"""
Number of repeats: 10
Average Accuracy: 0.9206
Average Precision: 0.7130
Average Recall: 0.5563 
Average F1 score: 0.6170
Average AUC: 0.9099
"""
```

### BERT Model

```bash
# Run BERT implementation (Please modify the code in the load_data section to read the correct data)
python bert_classification.py

# Expected Output Format:
"""
Epoch 15/15
Train Loss: 0.0052
Accuracy: 0.9121
Precision: 0.6000
Recall: 0.8182
F1 Score: 0.6923
AUC: 0.9545
"""
```

## Experimental Results

### PyTorch Implementation
| Model       | Accuracy | Precision | Recall | F1 Score | AUC   |
|-------------|----------|-----------|--------|----------|-------|
| Baseline    | 0.6278   | 0.6077    | 0.7446 | 0.5557   | 0.7446|
| SVM         | 0.9206   | 0.7130    | 0.5563 | 0.6170   | 0.9099|
| **BERT**    | 0.9121   | 0.6000    | 0.8182 | 0.6923   | 0.9545|

### TensorFlow Implementation
| Model       | Accuracy | Precision | Recall | F1 Score | AUC   |
|-------------|----------|-----------|--------|----------|-------|
| Baseline    | 0.5611   | 0.6353    | 0.7214 | 0.5390   | 0.7214|
| SVM         | 0.8931   | 0.7376    | 0.7419 | 0.7376   | 0.9385|
| **BERT**    | 0.9265   | 0.8140    | 0.8333 | 0.8235   | 0.9687|

### Keras Implementation
| Model       | Accuracy | Precision | Recall | F1 Score | AUC   |
|-------------|----------|-----------|--------|----------|-------|
| Baseline    | 0.5590   | 0.6290    | 0.6974 | 0.5388   | 0.6974|
| SVM         | 0.8800   | 0.7035    | 0.5733 | 0.6193   | 0.8879|
| **BERT**    | 0.8947   | 0.6667    | 0.8000 | 0.7273   | 0.9511|

### incubator-mxnet Implementation
| Model       | Accuracy | Precision | Recall | F1 Score | AUC   |
|-------------|----------|-----------|--------|----------|-------|
| Baseline    | 0.6087   | 0.6140    | 0.7511 | 0.5486   | 0.7511|
| SVM         | 0.9083   | 0.7183    | 0.3375 | 0.4391   | 0.8199|
| BERT        | 0.8958   | 0.6667    | 0.3333 | 0.4444   | 0.6349|

## File Structure

```plaintext
.
├── br_classification_optimized.py     # SVM implementation with optimized classification
├── bert_classification.py             # BERT-based text classification model
├── manual.pdf                         # A manual to explain how to use the tool
├── replication.pdf                    # A clear instruction as to how we can replicate the results reported.
├── requirements.pdf                   # Any dependencies/versions that are required to compile or run the code.
├── datasets/
│   ├── pytorch.csv                    # PyTorch-related issue dataset
│   ├── incubator-mxnet.csv            # MXNet-related issue dataset
│   ├── keras.csv                      # Keras-related issue dataset
│   └── tensorflow.csv                 # TensorFlow-related issue dataset
└── README.md                          # Project documentation
```

## Reproducibility

```bash
Random seed: 42
Training epochs: 15
Batch size: 32
Validation split: 20%
```
