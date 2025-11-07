from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as  pd
import numpy as np


# load Dataset
df=pd.read_csv('housing.csv')

# Dataoverview
print("Data Overview:")
print(df.head())

X=df[['Area','Bedrooms','Bathrooms']]
y=df['Price']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Model Training
model=DecisionTreeRegressor()
model.fit(X_train,y_train)

# Making Model Predictions
y_pred=model.predict(X_test)

# Model Evaluation
print('Mean Square Error: \n',mean_squared_error(y_test,y_pred))
print('R2 Score : \n',r2_score(y_test,y_pred))
print('Mean Absolute Error : \n',mean_absolute_error(y_test,y_pred))

# Predicting a new house price
new_house=pd.DataFrame([[1800, 3, 2]], columns=['Area', 'Bedrooms', 'Bathrooms'])
predicted_price=model.predict(new_house)
print(f'\n💰 The predicted price for the new house is: Rs {predicted_price[0]:.2f}')