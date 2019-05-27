import matplotlib.pyplot as plt
import numpy as np
import joblib as jb
from sklearn import datasets, svm, metrics


# digits.data, digits.images, digits.target,
# digits.target_names, digits.DESCR
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
"""
for i, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, i + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Target: {label}')
"""

# reshape from (1, 8, 8) to (1, 64)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# gamma value can be found using
# grid search and cross validation
classifier = svm.SVC(gamma=0.001)


# training classifier on half the data
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])


# using it to predict on second half of data
expected_values = digits.target[n_samples // 2:]
predicted_values = classifier.predict(data[n_samples // 2:])


classification_report = metrics.classification_report(expected_values, predicted_values)
#print(f'Classification report for classifier {classifier}:\n {classification_report}')


images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted_values))
for i, (image, prediction) in enumerate(images_and_predictions[:25]):
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Prediction: {prediction}\n')


plt.tight_layout()
plt.show()


# dumped with python 3.6.0 | anaconda 4.3.0
# jb.dump(classifier, 'first_classifier.joblib')
