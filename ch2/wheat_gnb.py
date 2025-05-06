# wheat_gnb.py
# Gaussian NB on the Wheat Seeds dataset

# Anaconda3-2022.10  Python 3.9.13
# scikit 1.0.2  Windows 10/11 
# [Gaussian Naive Bayes Classification Using the scikit Library -- Visual Studio Magazine](https://visualstudiomagazine.com/articles/2023/05/31/gaussian-naive-bayes.aspx?Page=1)

import numpy as np
from pyhere import here
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------------

def show_confusion(cm):
  dim = len(cm)
  mx = np.max(cm)             # largest count in cm
  wid = len(str(mx)) + 1      # width to print
  fmt = "%" + str(wid) + "d"  # like "%3d"
  for i in range(dim):
    print("actual   ", end="")
    print("%3d:" % i, end="")
    for j in range(dim):
      print(fmt % cm[i][j], end="")
    print("")
  print("------------")
  print("predicted    ", end="")
  for j in range(dim):
    print(fmt % j, end="")
  print("")

# ---------------------------------------------------------

def main():
  # 0. prepare
  print("\nBegin scikit Gaussian naive Bayes demo ")
  print("Predict wheat species (0,1,2) from seven numerics ")
  np.random.seed(1)
  np.set_printoptions(precision=4, suppress=True)

  # 1. load data
  print("\nLoading train and test data ")
  train_file = here("data", "ch02", "seeds", "wheat_train_k.txt")
  x_train = np.loadtxt(train_file, usecols=[0,1,2,3,4,5,6],
    delimiter=",", comments="#", dtype=np.float32)
  y_train = np.loadtxt(train_file, usecols=7,
    delimiter=",", comments="#", dtype=np.int64) 

  test_file = here("data", "ch02", "seeds", "wheat_test_k.txt")
  x_test = np.loadtxt(test_file, usecols=[0,1,2,3,4,5,6],
    delimiter=",", comments="#", dtype=np.float32)
  y_test = np.loadtxt(test_file, usecols=7,
    delimiter=",", comments="#", dtype=np.int64) 
  print("Done ")

  print("\nData: ")
  print(x_train[0:4][:])
  print(". . .")
  print("\nActual species: ")
  print(y_train[0:4])
  print(". . .")

  # 2. create and train model
  # GaussianNB(*, priors=None, var_smoothing=1e-09)
  print("\nCreating Gaussian naive Bayes classifier ")
  model = GaussianNB()
  model.fit(x_train, y_train)
  print("Done ")

  # 3. evaluate model
  acc_train = model.score(x_train, y_train)
  print("\nAccuracy on train data = %0.4f " % acc_train)
  acc_test = model.score(x_test, y_test)
  print("Accuracy on test data =  %0.4f " % acc_test)

  # 3b. confusion matrix
  from sklearn.metrics import confusion_matrix
  y_predicteds = model.predict(x_test)
  cm = confusion_matrix(y_test, y_predicteds) 
  print("\nConfusion matrix for test data: ")
  show_confusion(cm)

  # 3c. precision, recall, F1 for binary classification
  # from sklearn.metrics import classification_report
  # report = classification_report(y, pred_classes) 
  # print(report) 

  # 4. use model
  print("\nPredicting species all 0.2 predictors: ")
  X = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]],
    dtype=np.float32)
  print(X)
  probs = model.predict_proba(X)
  print("\nPrediction probs: ")
  print(probs)

  predicted = model.predict(X)
  print("\nPredicted class: ")
  print(predicted)

  # 5. TODO: save model using pickle
  
  print("\nEnd demo ")

if __name__ == "__main__":
  main()
