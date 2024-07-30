# Singer Isolation from Mixture Audio Using Deep Learning

This project involves extracting vocal tracks from music mixtures using a deep learning approach. The process includes generating spectrograms of audio tracks, training a convolutional neural network (CNN) to isolate vocals, and evaluating the performance of the model.

## Table of Contents

1. [Introduction](#introduction)
2. [Hardware and Software Requirements](#hardware-and-software-requirements)
3. [Setup Instructions](#setup-instructions)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Vocal Isolation](#vocal-isolation)
8. [Evaluation](#evaluation)
9. [Visualization](#visualization)
10. [License](#license)

## Introduction

The goal of this project is to isolate the vocal component from a given music mixture using a CNN. The input to the model is a spectrogram of the mixture, and the output is a mask indicating the presence of vocals at different time-frequency bins. The mask is then used to reconstruct the vocal spectrogram, which is transformed back into an audio signal.

## Hardware and Software Requirements

- **Python 3.6 or higher**
- **Libraries**: NumPy, librosa, matplotlib, scikit-learn, torch, musdb
- **Hardware**: A computer with sufficient memory and processing power to handle audio data and train neural networks

