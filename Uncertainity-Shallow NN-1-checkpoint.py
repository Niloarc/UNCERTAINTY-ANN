# Training the Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor 
from sklearn.metrics import mean_squared_error, r2_score


file_path = r"C:\Users\nilou\OneDrive\Documents\PhD-ESF\Sem 03\ML course\Final Project\Paper_data.csv"
df = pd.read_csv(file_path)
df.head()

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model
def build_model(n_hidden=1, n_neurons=30, activation_fun='relu', neural_optimizer='adam', learning_rate=0.001):
    model = Sequential()
    model.add(Dense(n_neurons, input_shape=(X_train.shape[1],), activation=activation_fun))
    for _ in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation_fun))
    model.add(Dense(1))  

    # Optimizer selection
    if neural_optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif neural_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif neural_optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer!")

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


# Model optimization
param_distribs = {
    'model__n_hidden': [1, 2, 3],  
    'model__n_neurons': [30, 50, 100],  
    'model__activation_fun': ['relu', 'tanh', 'elu'],  
    'model__neural_optimizer': ['adam', 'rmsprop', 'sgd'],  
    'model__learning_rate': [0.01, 0.001, 0.0001],  
    'batch_size': [32, 64],  
    'epochs': [10, 20]  
}

keras_reg = KerasRegressor(
    model=build_model,  
    verbose=0
)

grid_search = GridSearchCV(
    estimator=keras_reg,
    param_grid=param_distribs,
    cv=3,  # cross-validation
    verbose=2,
    n_jobs=-1  
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = build_model(
    n_hidden=best_params['model__n_hidden'],
    n_neurons=best_params['model__n_neurons'],
    activation_fun=best_params['model__activation_fun'],
    neural_optimizer=best_params['model__neural_optimizer'],
    learning_rate=best_params['model__learning_rate']
)

history = best_model.fit(X_train, y_train, 
                         epochs=best_params['epochs'], 
                         batch_size=best_params['batch_size'], 
                         validation_split=0.2)

# Model evaluation
test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()


# Evaluation based on test set
test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='teal', edgecolor='k', alpha=0.6, label="Predicted vs. Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='deeppink', linestyle='--', linewidth=2, label="Ideal")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Neural Network: True vs Predicted Values")
plt.legend()
plt.grid(True)
plt.show()
best_model.save('Uncertainty_neural_network_model.keras')

from tensorflow.keras.models import load_model

loaded_model = load_model('Uncertainty_neural_network_model.keras')


def plot_nn_learning_curve(X_train, X_test, y_train, y_test, model, increments=10):
    train_sizes = np.linspace(10, len(X_train), increments, dtype=int)  
    train_scores = []
    test_scores = []

    for size in train_sizes:
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        model.fit(X_train_subset, y_train_subset, epochs=5, batch_size=32, verbose=0)
        y_train_pred = model.predict(X_train_subset).ravel()
        train_scores.append(r2_score(y_train_subset, y_train_pred))  # R² training
        y_test_pred = model.predict(X_test).ravel()
        test_scores.append(r2_score(y_test, y_test_pred))  # R² testing

    # Learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', color='deeppink')
    plt.plot(train_sizes, test_scores, 'o-', label='Test Score', color='teal')
    plt.fill_between(train_sizes, np.array(train_scores) - 0.05, np.array(train_scores) + 0.05, alpha=0.2, color='orchid')
    plt.fill_between(train_sizes, np.array(test_scores) - 0.05, np.array(test_scores) + 0.05, alpha=0.2, color='darkslategrey')
    plt.title("Neural Network Learning Curve", fontsize=14)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.show()

plot_nn_learning_curve(X_train, X_test, y_train, y_test, best_model)


