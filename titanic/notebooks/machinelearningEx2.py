import pandas as pd

# Path of the file to read
iowa_file_path = "../data/raw/train2.csv"

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Call line below with no argument to check that you've loaded the data correctly
print(home_data.head)

home_data.describe()

avg_lot_size = home_data.LotArea.mean().round()
print(avg_lot_size)

newest_home_age = 2024 - home_data.YearBuilt.max()
print(newest_home_age)


y = home_data.SalePrice
print(y)

feature_names = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]

x = home_data[feature_names]
print(x.describe)
print(x.head(10))

from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor()
iowa_model.fit(x, y)

print(iowa_model.predict(x))

from sklearn.metrics import mean_absolute_error

print("First in-sample predictions:", iowa_model.predict(x.head()))
print("Actual target values for those homes:", y.head().tolist())

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=1)

iowa_modelSplitModel = DecisionTreeRegressor(random_state=1)
iowa_modelSplitModel.fit(train_X, train_y)


val_predictions =iowa_modelSplitModel.predict(val_X)

print("First in-sample predictions:", iowa_modelSplitModel.predict(val_X.head(5)))
print("Actual target values for those homes:", y.head(5).tolist())

val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

#https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting/tutorial