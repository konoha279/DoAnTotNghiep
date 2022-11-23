import numpy as np
import torch
import sys
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix


def main():

    return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("main.py <cnn module> <image check>")
        exit(0)
    main()