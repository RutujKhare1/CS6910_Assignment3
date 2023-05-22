# CS6910 Assignment3

# English to Hindi Sequence-to-Sequence RNN Model

This repository contains the implementation of a Sequence-to-Sequence Recurrent Neural Network (RNN) model for translating English words to Hindi words. The model is implemented using the PyTorch deep learning framework.

## Dataset
The dataset used is  Aksharantar dataset released by AI4Bharat to train and evaluate the model which consists of pairs of English words and their corresponding Hindi translations. Each word is treated as a sequence of characters, and the goal of the model is to learn the mapping between the English and Hindi characters.

## Model Architecture
The model architecture follows the sequence-to-sequence paradigm, consisting of an encoder and a decoder. The encoder processes the input sequence of English characters, while the decoder generates the corresponding Hindi sequence.

The code is flexible such that the dimension of the input character embeddings, the hidden states of the encoders and decoders, the cell (RNN, LSTM, GRU), and the number of layers in the encoder and decoder can be changed.

## Training
To train the model, the dataset is already split into training, testing and validation sets. The model is trained using mini-batch stochastic gradient descent with teacher forcing. Teacher forcing is a technique where, during training, the model is fed with the ground truth Hindi characters as input to predict the next character.

The model's performance is evaluated using metrics such as accuracy and loss on the validation set. The training process continues until the model converges or reaches a predefined number of epochs.

## Evaluation
After training, the model can be used for translating English words to Hindi words. Given an English word as input, the model generates the corresponding Hindi word character by character until the end-of-sequence token is produced.

To evaluate the translation quality, one can use various metrics such as BLEU score, which measures the similarity between the generated Hindi word and the reference Hindi word.

## Dependencies
The following dependencies are required to run the code:
- Python (>=3.6)
- PyTorch (>=1.6)

## Usage
1. Clone this repository:
   ```
   git clone https://github.com/RutujKhare1/CS6910_Assignment3.git
   ```

2. Train the model:
   ```
   python train.py
   ```

3. The model will be trained and the script will show training accuracy and testing accuracy respectively.

NOTE : The python notebook 'FDL_A3.ipynb' is included where the sweep configuration script, generating predicition_vanilla.csv and predicition_attention.csv is included

LINK TO WANDB REPORT : https://wandb.ai/team_exe/fdl_a3/reports/CS6910-Assignment-3--Vmlldzo0NDI2MDc0
