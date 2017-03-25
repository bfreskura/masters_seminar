# Master's seminar

## Introduction
The goal of the seminar is to re-implement the paper: *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF*
You can find the complete paper in the `docs` folder.

### Paper Abstract
State-of-the-art sequence labeling systems
traditionally require large amounts of task-
specific knowledge in the form of hand-
crafted features and data pre-processing.
In this paper, we introduce a novel neu-
tral network architecture that benefits from
both word- and character-level representa-
tions automatically, by using combination
of bidirectional LSTM, CNN and CRF.
Our system is truly end-to-end, requir-
ing no feature engineering or data pre-
processing, thus making it applicable to
a wide range of sequence labeling tasks.
We evaluate our system on two data sets
for two sequence labeling tasks — Penn
Treebank WSJ corpus for part-of-speech
(POS) tagging and CoNLL 2003 cor-
pus for named entity recognition (NER).
We obtain state-of-the-art performance on
both datasets — 97.55% accuracy for POS
tagging and 91.21% F1 for NER.


## Installation

### Requirements
* Python 3.6
* pip packages from the `requirements.txt` file


## Acknowledgments
A Big thanks to [Martin Tutek](https://www.linkedin.com/in/mtutek) who helped
with the understanding of the paper.

Also thanks to [TakeLab](http://takelab.fer.hr/) for providing all the resources.
