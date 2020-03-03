import pandas as pd
import sklearn
#import the libraries

flower_data=pd.read_csv("flower.csv") #Read the data from the file
print(flower_data)

features_data=flower_data[["sepal_length","petal_length"]]
class_data=flower_data[["class"]] #Split the class and features data

from sklearn.model_selection import train_test_split #import the function to split the data
x_train, x_test, y_train, y_test = train_test_split(features_data, class_data, test_size=0.30)#Split the data into training and testing


from sklearn.tree import DecisionTreeClassifier #Import the decision tree classifier

my_decision_tree = DecisionTreeClassifier(max_depth=3)#Set the decision tree attributes
my_decision_tree.fit(x_train,y_train) #Train the model using training data

my_prediction=my_decision_tree.predict(x_test) #Use the decision tree to predict the the flower type using  testing features

print("The actual flower type:",y_test)
print("The predicted flower type:",my_prediction)

from sklearn.metrics import accuracy_score
my_accuracy=accuracy_score(y_test,my_prediction) #find the accuracy by comparing the predicted flower type and the actual type
print("My decision tree accuracy =",my_accuracy)
