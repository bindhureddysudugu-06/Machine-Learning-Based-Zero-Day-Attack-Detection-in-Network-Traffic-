# Machine Learning Based Zero-Day Attack Detection in Network Traffic

## Overview
The aim of this project is to design and implement a machine learning-based anomaly detection system to detect zero-day attacks on the **NSL-KDD dataset**. The objective is to detect unknown attacks in network traffic, by training a model on normal traffic to detect anomalies in the test phase.

The project aims to implement a **zero-day evaluation** where one attack category is removed from the dataset during training - considered as unseen attacks - and used for testing purposes. For this experiment, **Probe** attack category had been excluded.

## Problem Statement
Conventional signature-based intrusion detection systems are capable of only detecting known attacks as they rely on a set of attack signatures. However, zero-day attacks have not been encountered before and don't match the signatures. This project overcomes that limitation by employing anomaly detection techniques that capture the normal behavior of traffic, and detect any abnormalities.

## Dataset
This project uses the **NSL-KDD dataset**, a widely-used intrusion detection dataset.

NSL-KDD contains a record for each network connection, which includes:
- 41 traffic-related features
-  1 label of either normal or one of the attacks
- 1 difficulty level field

In this project we classify the attack labels into:
- normal
- dos
- probe
- r2l
- u2r

## Project Objective
The aim of this project is to test the ability of anomaly detection models to detect new attacks in the absence of any attack signatures.

## Methodology

### 1. Data Loading
We load the NSL-KDD files (train and test) using Python and Pandas.

### 2. Label Grouping
Original attack labels are grouped into attack categories like: 'probe', 'dos', 'r2l' and 'u2r'.

### 3. Zero-Day Setup
This project is based on an attack-wise zero-day split:
- train on normal traffic
- test on normal + known (non-held-out) attacks
- test on normal + unknown attack (held-out)

In the executed experiment:
- held-out unseen attack = **Probe**

### 4. Preprocessing
The preprocessing pipeline includes:
- categorical feature encoding
- numerical feature scaling
- feature preparation for anomaly detection models

### 5. Models Used
The following anomaly detection models are used:
- **Isolation Forest**
- **One-Class SVM**

These methods were chosen because they can model the normal data and detect unseen attacks as anomalies.

### 6. Evaluation Metrics
The models were evaluated using:
- Precision
- Recall
- F1-score
- PR-AUC
- False Positive Rate

The metrics are more appropriate than accuracy for intrusion detection as they are more correlated with the detection of attacks and false alarms.

## Executed Experiment
The final executed experiment used **Probe** as the unseen attack category.

### Data Split
- Training normal samples: 53,875
- Validation samples: 26,936
- Test samples: 12,132

## Results

| Model | Precision | Recall | F1-score | PR-AUC | False Positive Rate |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.950858 | 0.663362 | 0.781509 | 0.956714 | 0.008547 |
| One-Class SVM | 0.765641 | 0.914911 | 0.833647 | 0.895909 | 0.069818 |

## Result Interpretation
The results indicate that both models successfully detected unseen Probe attacks.

- **One-Class SVM** achieved higher recall and F1-score, which means it detected more unseen attacks.
- **Isolation Forest** achieved higher precision, higher PR-AUC, and lower false positive rate, which means it produced fewer false alarms.

This reveals a practical trade-off:
- **One-Class SVM** is better for stronger zero-day attack detection
- **Isolation Forest** is better for better false-positive control

## Repository Structure

'''text
.
├── zero_day_detection.py
├── README.md
├── requirements.txt
└── outputs_probe/
    ├── metrics.csv
    ├── metrics.json
    └── run_summary.json
