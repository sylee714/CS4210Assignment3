#-------------------------------------------------------------------------
# AUTHOR: Seungyun Lee
# FILENAME: bagging_random_forest.py
# SPECIFICATION: A Python script that runs a single decision tree, an ensemble classifier, and a random forest on optic digits datasets
# FOR: CS 4210- Assignment #3
# TIME SPENT: 5 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

limit = 0

dbTraining = []
dbTest = []
X_training = []
Y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

def convert2XandY(data):
    x = []
    y = []
    for instance in data:
        instance_x = instance[:len(instance)-1]
        instance_y = instance[len(instance)-1]
        x.append(instance_x)
        y.append(instance_y)

    return [x, y]

#reading the training data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
      dbTraining.append (row)

#reading the test data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)
        classVotes.append([0,0,0,0,0,0,0,0,0,0]) #inititalizing the class votes for each test sample

    print("Started my base and ensemble classifier ...")

    for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample
        bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

        #populate the values of X_training and Y_training by using the bootstrapSample
        #--> add your Python code here
        x_and_y = convert2XandY(bootstrapSample)
        X_training = x_and_y[0]
        Y_training = x_and_y[1]

        #fitting the decision tree to the data
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
        clf = clf.fit(X_training, Y_training)

        first_base_classifier_correct_predictions = 0
        for i, testSample in enumerate(dbTest):

            # make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
            # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
            # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
            # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
            # this array will consolidate the votes of all classifier for all test samples
            #--> add your Python code here
            class_predicted_clf = clf.predict([testSample[:len(testSample) - 1]])[0]
            classVotes[i][int(class_predicted_clf)] = classVotes[i][int(class_predicted_clf)] + 1

            if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
            #--> add your Python code here
                if (class_predicted_clf == testSample[len(testSample) - 1]):
                    first_base_classifier_correct_predictions = first_base_classifier_correct_predictions + 1

        if k == 0: #for only the first base classifier, print its accuracy here
           #--> add your Python code here
           print("Finished my base classifier (fast but relatively low accuracy) ...")
           print("My base classifier accuracy: " + str(first_base_classifier_correct_predictions/len(dbTest) * 100) + "%")
           print("")

    number_of_correct_predictions = 0
    #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
    #--> add your Python code here
    for i, testSample in enumerate(dbTest):
        max_value = max(classVotes[i])
        max_index = classVotes[i].index(max_value)
        if max_index == int(testSample[len(testSample) - 1]):
            number_of_correct_predictions = number_of_correct_predictions + 1

    #printing the ensemble accuracy here
    print("Finished my ensemble classifier (slow but higher accuracy) ...")
    print("My ensemble accuracy: " + str(number_of_correct_predictions/len(dbTest) * 100) + "%")
    print("")

    print("Started Random Forest algorithm ...")

    #Create a Random Forest Classifier
    clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

    #Fit Random Forest to the training data
    clf.fit(X_training,Y_training)

    #make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
    #--> add your Python code here
    predicted_rf = []
    for i, testSample in enumerate(dbTest):
        class_predicted_rf = clf.predict([testSample[:len(testSample)-1]])[0]
        predicted_rf.append(class_predicted_rf)

    number_of_correct_predictions_rf = 0
    #compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    #--> add your Python code here
    for i, testSample in enumerate(dbTest):
        if int(predicted_rf[i]) == int(testSample[len(testSample)-1]):
            number_of_correct_predictions_rf = number_of_correct_predictions_rf + 1

    #printing Random Forest accuracy here
    print("Random Forest accuracy: " + str(number_of_correct_predictions_rf/len(dbTest) * 100) + "%")

    print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")




