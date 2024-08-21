# Breast Cancer Diagnosis with Perceptron Neural Network

This project demonstrates the use of a Perceptron neural network to classify breast cancer diagnoses based on the Wisconsin Breast Cancer (Diagnostic) Data Set.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploration and Preprocessing](#exploration-and-preprocessing)
- [Outlier Detection](#outlier-detection)
- [Neural Network](#neural-network)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Installation

To run this project, you'll need to install the necessary libraries. You can do this by pip installing the following:

```
scikit-learn
matplotlib
numpy
pandas
```

Otherwise, you could use the Anaconda environment.

## Dataset

The dataset used is the Wisconsin Breast Cancer Data Set. Ensure you have the dataset file named `BreastCancerWisconsinDataSet.csv` in your working directory.

## Exploration and Preprocessing

After loading the dataset, the project performs initial exploration and preprocessing:

- **Exploration:** The dataset's header and basic information are inspected to understand its structure and identify any missing values.
- **Preprocessing:** The first column (ID numbers) and the last column (empty) are removed from the dataset.

## Outlier Detection

Outliers are checked using a custom function to ensure that there are no extreme values that could affect the performance of the neural network. After identifying columns with many zeroes, a new filtered version of the dataset is created.

## Neural Network

Two Perceptron models are built and tested:

1. **Model 1 (p1):** Trained on the original dataset.
2. **Model 2 (p2):** Trained on the filtered dataset, excluding rows with zero values in the concavity-related columns.

### Data Standardization

The features are normalized using `MinMaxScaler` to scale the data between 0 and 1.

### Data Visualization

Scatter plots are generated to check if the classes (Malignant and Benign) are linearly separable, which is a key requirement for using a Perceptron.

### Training and Testing

The dataset is split into training (70%) and testing (30%) sets. The models are then trained and evaluated using accuracy, classification reports, and confusion matrices.

## Results

- **Model 1 (p1):** Shows good performance on the original dataset.
- **Model 2 (p2):** Shows superior accuracy on the filtered dataset, with zero false negatives and minimal false positives.

## Conclusion

- **Model 1** is recommended for patients with no concavities in their breast mass (2.3% of the dataset).
- **Model 2** is recommended for patients with concavities in their breast mass (97.7% of the dataset) due to its higher accuracy and lower error rate.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
