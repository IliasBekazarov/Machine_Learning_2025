import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv('restaurant_revenue.csv')

data = pd.get_dummies(data, columns=['Location', 'Cuisine', 'Parking Availability'], drop_first=True)

X = data.drop(['Name', 'Revenue'], axis=1)
y = data['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump((model, X.columns.tolist()), 'restaurant_model.pkl')
print("Model and feature names saved successfully!")

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Негизги (реалдуу) Revenue")
plt.ylabel("Божомолдонгон Revenue")
plt.title("Реалдуу vs Болжолдонгон Киреше (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.savefig('revenue_prediction_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
