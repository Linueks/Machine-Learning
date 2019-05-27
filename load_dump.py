import matplotlib.pyplot as plt
import numpy as np
import joblib as jb
from sklearn import datasets, svm, metrics


classifier = jb.load('first_classifier.joblib')
