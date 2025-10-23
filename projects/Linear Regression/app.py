from flask import Flask, render_template, request
import pandas as pd
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import namedtuple

app = Flask(__name__)

# 1️⃣ Готовую модель и имена признаков загружаем
model, feature_names = joblib.load('restaurant_model.pkl')

# 2️⃣ Загружаем тестовые данные (чтобы только проверить метрики)
data = pd.read_csv('test.csv')
data = pd.get_dummies(data, columns=['Location', 'Cuisine', 'Parking Availability'], drop_first=True)
X = data.drop(['Name', 'Revenue'], axis=1)
y = data['Revenue']

# 3️⃣ Предсказания и метрики (один раз при запуске)
y_pred = model.predict(X)
Metrics = namedtuple('Metrics', ['mae', 'mse', 'r2'])
metrics = Metrics(
    mae = mean_absolute_error(y, y_pred),
    mse = mean_squared_error(y, y_pred),
    r2  = r2_score(y, y_pred)
)

# 4️⃣ График "реальное vs предсказанное"
fig, ax = plt.subplots()
ax.scatter(y, y_pred, color='blue')
ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
ax.set_xlabel("Негизги (реалдуу) Revenue")
ax.set_ylabel("Божомолдонгон Revenue")
ax.set_title("Реалдуу vs Болжолдонгон Киреше")

buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
plot_data = base64.b64encode(buf.getvalue()).decode()
buf.close()
plt.close(fig)

# 5️⃣ Главная страница
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_revenue = None

    if request.method == 'POST':
        # Данные от пользователя
        rating = float(request.form['rating'])
        seating_capacity = int(request.form['seating_capacity'])
        avg_price = float(request.form['avg_price'])
        marketing_budget = float(request.form['marketing_budget'])
        social_followers = int(request.form['social_followers'])
        chef_exp = int(request.form['chef_exp'])
        num_reviews = int(request.form['num_reviews'])
        avg_review_len = float(request.form['avg_review_len'])
        ambience = float(request.form['ambience'])
        service = float(request.form['service'])
        weekend_res = int(request.form['weekend_res'])
        weekday_res = int(request.form['weekday_res'])
        location = request.form['location'].lower()
        cuisine = request.form['cuisine'].lower()
        parking = request.form['parking'].lower()

        # Создаем DataFrame для нового прогноза
        new_data = pd.DataFrame({
            'Rating': [rating],
            'Seating Capacity': [seating_capacity],
            'Average Meal Price': [avg_price],
            'Marketing Budget': [marketing_budget],
            'Social Media Followers': [social_followers],
            'Chef Experience Years': [chef_exp],
            'Number of Reviews': [num_reviews],
            'Avg Review Length': [avg_review_len],
            'Ambience Score': [ambience],
            'Service Quality Score': [service],
            'Weekend Reservations': [weekend_res],
            'Weekday Reservations': [weekday_res],
            'Location_Downtown': [1 if location == 'downtown' else 0],
            'Location_Rural': [1 if location == 'rural' else 0],
            'Cuisine_French': [1 if cuisine == 'french' else 0],
            'Cuisine_Indian': [1 if cuisine == 'indian' else 0],
            'Cuisine_Italian': [1 if cuisine == 'italian' else 0],
            'Cuisine_Japanese': [1 if cuisine == 'japanese' else 0],
            'Cuisine_Mexican': [1 if cuisine == 'mexican' else 0],
            'Parking Availability_Yes': [1 if parking == 'yes' else 0],
        })

        # Убедимся, что все нужные фичи есть
        for col in feature_names:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[feature_names]

        # Делаем прогноз
        predicted_revenue = model.predict(new_data)[0]

    return render_template('index.html', result=predicted_revenue, metrics=metrics, plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
