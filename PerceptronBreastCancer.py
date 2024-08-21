### Importing Libraries ###
from sklearn import preprocessing # library for preprocessing support
from sklearn.model_selection import train_test_split # library for splitting samples into training and testing sets
from sklearn.linear_model import Perceptron # library with functions for executing the Perceptron neural network
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics # library for obtaining metrics for model evaluation
import matplotlib.pyplot as plt # library for plotting graphs
import numpy as np
import pandas as pd



### Loading the Data Set ###
# Path of the dataset that will be loaded into a dataframe
df = pd.read_csv("BreastCancerWisconsinDataSet.csv")



### Exploring the Data Set ###
# Checking the header of the dataframe
print(df.head())
# Data Wrangler is especially useful in this step



### Data Preprocessing ###
# Checking the info
print(df.info())

# As the first column is only made of id numbers and the last column is null, they're removed from the dataframe
df.drop(columns=[df.columns[0], df.columns[-1]], inplace=True)



### Dealing with Outliers ###
# Check the minimum and maximum values for each column in the dataset to see if there are any extreme outliers
# If there are not, we can use the MinMaxScaler
# For this, we'll make a function
def check_outliers(df=df):
    # Start a counter
    i = 0
    # Loop through each column
    for column in df.columns:
        # Sort the dataframe by the current column
        sorted_df = df.sort_values(by=column)
        # Check if the values are integers or floats
        if isinstance(sorted_df[column].iloc[0], int) or isinstance(sorted_df[column].iloc[0], float):
            # Avoid division by zero
            if sorted_df[column].iloc[0] == 0:
                print('\n *** Possible outlier(s)! Check with attention: ***')
            # Alert if the maximum value is over 50 times bigger than the minimum
            # elif (sorted_df[column].iloc[-1]/sorted_df[column].iloc[0]) > 50:
            #     print('\n *** Possible outlier(s)! Check with attention: ***')
        # Get the ten minimum values
        print(f'{column} min:', sorted_df[column].head(10).values)
        # Get the ten maximum values
        print(f'{column} max:', sorted_df[column].tail(10).values)
        print()
        # Increment the counter
        i += 1
    
    return

check_outliers()

# As there are some columns with many zeroes, we'll filter those and make a new dataframe
# Copy the dataframe
df_no2 = df.copy()

# deleting all the rows with zeroes
df_no2 = df_no2[~(df_no2 == 0).any(axis=1)]

# Checking the info of the new dataframe
print(df_no2.info())

# We can see that 13 rows from the original 569 were removed
# 13/569 = 0.022847
# Approximately 2.3% of the women in the dataset show no concavities in their breasts



### Separating the Input Data for the Neural Network ###
# We'll test if the neural network has better results with the original dataframe or the new dataframe
# Separating the target column from the dataset
y_original = df.iloc[:, 0].values
y_filtered = df_no2.iloc[:, 0].values

# Encoding the target column using the label encoder - Transforms the classes into 0 and 1 - Diagnosis (M = malignant = 1, B = benign = 0)
y_original = np.where(y_original == 'M', 1, 0)
y_filtered = np.where(y_filtered == 'M', 1, 0)

# Separating the columns with variables to determine the inputs of the neural network
x_original = df.iloc[:, 1:].values
x_filtered = df_no2.iloc[:, 1:].values

# Check if everything was selected correctly
print(x_original.shape)
print(y_original.shape)
print(x_filtered.shape)
print(y_filtered.shape)



### Feature Standardization ###
# As seen previously, except for cases with zeroes, there were no isolated extreme outliers in the maximum and minimum values of each column - so we can use the MinMaxScaler
# Normalization of data using sklearn - scaling data between 0 and 1
scaler = preprocessing.MinMaxScaler()
x_original = scaler.fit_transform(x_original)
x_filtered = scaler.fit_transform(x_filtered)



### Visualizing the Data ###
# Plotting some graphs to check if the samples are linearly separable
# If the two classes (Malignant and Benign) are linearly separable, we can use the Perceptron neural network
plt.scatter(x_original[:,1], x_original[:,2], c=y_original, alpha=0.6)
plt.title('Linear Separability')
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.show()
plt.clf()

plt.scatter(x_original[:,3], x_original[:,4], c=y_original, alpha=0.6)
plt.title('Linear Separability')
plt.xlabel('perimeter_mean')
plt.ylabel('area_mean')
plt.show()
plt.clf()



### Separating the Data for Training and Testing ###
# Splitting the dataset into training and testing samples: 70% of the values for training and 30% for testing
x_train_original, x_test_original, y_train_original, y_test_original = train_test_split(x_original, y_original, test_size=0.30, random_state=12)
x_train_filtered, x_test_filtered, y_train_filtered, y_test_filtered = train_test_split(x_filtered, y_filtered, test_size=0.30, random_state=12)



### Building and Testing the Models ###
# Making two Perceptron models: 'p1' and 'p2'
p1 = Perceptron()
p1.fit(x_train_original, y_train_original)

p2 = Perceptron()
p2.fit(x_train_filtered, y_train_filtered)



### Presentation of Metrics ###
# Model 1 ('p1'):
predictions_train_1 = p1.predict(x_train_original) # Validation of the trained sample set
train_score_1 = accuracy_score(predictions_train_1, y_train_original) # Accuracy evaluation of the classification of samples presented during training
print("Accuracy with training data: ", train_score_1)

predictions_test_1 = p1.predict(x_test_original) # Validation of the sample set that was not used in training
test_score_1 = accuracy_score(predictions_test_1, y_test_original) # Accuracy evaluation of the classification of samples that were not used in training
print("Accuracy with test data: ", test_score_1)

print(classification_report(predictions_test_1, y_test_original))

print("Number of epochs in training: ", p1.n_iter_)
print("List of parameters configured in Perceptron: ", p1.get_params())


# Model 2 ('p2'):
predictions_train_2 = p2.predict(x_train_filtered) # Validation of the trained sample set
train_score_2 = accuracy_score(predictions_train_2, y_train_filtered) # Accuracy evaluation of the classification of samples presented during training
print("Accuracy with training data: ", train_score_2)

predictions_test_2 = p2.predict(x_test_filtered) # Validation of the sample set that was not used in training
test_score = accuracy_score(predictions_test_2, y_test_filtered) # Accuracy evaluation of the classification of samples that were not used in training
print("Accuracy with test data: ", test_score)

print(classification_report(predictions_test_2, y_test_filtered))

print("Number of epochs in training: ", p2.n_iter_)
print("List of parameters configured in Perceptron: ", p2.get_params())



### Confusion Matrix
# Graphical presentation of the confusion matrix for the tests
conf_matrix = confusion_matrix(y_test_original, predictions_test_1)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['Malignant', 'Benign'])
cm_display.plot()
plt.show()
plt.clf()

conf_matrix = confusion_matrix(y_test_filtered, predictions_test_2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['Malignant', 'Benign'])
cm_display.plot()
plt.show()
plt.clf()



### Conclusion ###
# In cases where the patient’s breasts show no concavities (2.3% of the women in the dataset), the Perceptron model 'p1' should be used, as it was trained with datasets containing zero values for concavities.  
# Otherwise, if concavities are observed (97.7% of the women in the dataset), model 'p2' should be used, given its superior accuracy in this scenario – zero false negatives and only three false positives.