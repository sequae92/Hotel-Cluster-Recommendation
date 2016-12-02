import pandas as pd
import random

#import original data
destinationsData = pd.read_csv("data/destinations.csv")
testData = pd.read_csv("data/test.csv")
trainData = pd.read_csv("data/train.csv")

#See how many rows are in the training and testing data
trainShape = trainData.shape
print("Rows in train.csv: ", trainShape)
testShape = testData.shape
print("Rows in test.csv: ", testShape)

#Find out how many unique users are in the training data
uniqueUsers = trainData.user_id.unique()
print("Unique users: ", len(uniqueUsers))

#Find out approximately how many rows of data for 8000 users
sampleUsers = [uniqueUsers[i] for i in sorted(random.sample(range(len(uniqueUsers)), 8000)) ]
selectedUserData = trainData[trainData.user_id.isin(sampleUsers)]
print("Selected User data size: ", len(selectedUserData))

#convert date to add month and year to the data
selectedUserData["date_time"] = pd.to_datetime(selectedUserData["date_time"])
selectedUserData["year"] = selectedUserData["date_time"].dt.year
selectedUserData["month"] = selectedUserData["date_time"].dt.month

#split into 2 sets, one for training and one for testing
train1 = selectedUserData[((selectedUserData.year == 2013) | ((selectedUserData.year == 2014) & (selectedUserData.month < 8)))]
test2 = selectedUserData[((selectedUserData.year == 2014) & (selectedUserData.month >= 8))]

#then strip out all rows that do not have is_booking = TRUE for our test dataset
test2 = test2[test2.is_booking == True]
print("Training User data size: ", len(train1))
print("Testing User data size: ", len(test2))

#Write the selected user data to a new file so we can use it for learning
train1.to_csv("data/train1.csv")
test2.to_csv("data/test2.csv")