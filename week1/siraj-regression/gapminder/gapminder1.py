import pandas as pd
from sklearn.linear_model import LinearRegression


# Assign the dataframe to this variable.
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
model = LinearRegression()
bmi_life_model = model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Mak a prediction using the model
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)
