# Training the model
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Data set
file_path = r"C:\Users\nilou\OneDrive\Documents\PhD-ESF\Sem 03\ML course\Final Project\Paper_data.csv"
df = pd.read_csv(file_path)
df.head()

X = df.iloc[:, :-1]  # Input 
y = df.iloc[:, -1]   # Target which is MSP
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#t-SNE to reduce features

perplexity_values = [5, 9, 13, 18, 26, 30] 
plt.figure(figsize=(10, len(perplexity_values) * 4))

for i, perplexity in enumerate(perplexity_values):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    tsne_result = tsne.fit_transform(X)
    x_min, x_max = np.min(tsne_result[:, 0]), np.max(tsne_result[:, 0])
    y_min, y_max = np.min(tsne_result[:, 1]), np.max(tsne_result[:, 1])
    x_margin = (x_max - x_min) * 0.1  # Add margin (visualization)
    y_margin = (y_max - y_min) * 0.1
    
    # Plotting
    plt.subplot(len(perplexity_values), 1, i + 1)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, cmap='viridis', s=40)
    plt.colorbar(scatter, label='MFSP')
    plt.title(f't-SNE Visualization with Perplexity={perplexity}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

plt.tight_layout()
plt.show()


#Linear SVM model
import time  
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('lsvm', LinearSVR(random_state=42, max_iter=10000))  # Linear SVM for regression
])

param_grid = [
    {
        'lsvm__C': [0.01, 0.1, 1, 10],  # Regularization 
        'lsvm__epsilon': [0, 0.1, 1, 10],  # Epsilon boundary
        'lsvm__fit_intercept': [True, False],  # Intercept 
        'lsvm__loss': ['epsilon_insensitive'],  # Supported loss with dual=True
        'lsvm__dual': [True]  # Dual must be True for epsilon_insensitive
    },
    {
        'lsvm__C': [0.01, 0.1, 1, 10], 
        'lsvm__epsilon': [0, 0.1, 1, 10],
        'lsvm__fit_intercept': [True, False],
        'lsvm__loss': ['squared_epsilon_insensitive'],  # Supported with dual=True/False
        'lsvm__dual': [True, False]
    }
]

start_time = time.time()

# Optimization of model
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,  # cross-validation
    scoring='r2',  
    n_jobs=-1,  
    verbose=2,
    error_score="raise"  
)

grid_search.fit(X_train, y_train)
end_time = time.time()


training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")
print("Best Parameters:", grid_search.best_params_)
print(f"Best R-squared score: {grid_search.best_score_:.4f}")


# Test evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Error calculation based on test
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Set Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Error calculation based on training set
y_train_pred = grid_search.best_estimator_.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("Training Set Performance:")
print(f"Mean Squared Error (MSE): {train_mse:.4f}")
print(f"Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"R-squared (R2): {train_r2:.4f}")


# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='teal', edgecolor='k', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='deeppink', linestyle='--', linewidth=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("LSVM True vs Predicted Values")
plt.grid(True)
plt.show()

#MSP range prediction
y_pred_min = y_pred.min()
y_pred_max = y_pred.max()
print(f"Predicted MSP Range: [{y_pred_min:.4f}, {y_pred_max:.4f}]")

#Learning curve analysis
def plot_learning_curve(X_train, X_test, y_train, y_test, model, increments=10):
    train_sizes = np.linspace(10, len(X_train), increments, dtype=int)  
    train_scores = []
    test_scores = []

    for size in train_sizes:
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        model.fit(X_train_subset, y_train_subset)

        y_train_pred = model.predict(X_train_subset)
        train_scores.append(r2_score(y_train_subset, y_train_pred))  

        y_test_pred = model.predict(X_test)
        test_scores.append(r2_score(y_test, y_test_pred))  

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', color='deeppink')
    plt.plot(train_sizes, test_scores, 'o-', label='Test Score', color='teal')
    plt.fill_between(train_sizes, np.array(train_scores) - 0.05, np.array(train_scores) + 0.05, alpha=0.2, color='orchid')
    plt.fill_between(train_sizes, np.array(test_scores) - 0.05, np.array(test_scores) + 0.05, alpha=0.2, color='darkslategrey')
    plt.title("LSVM Learning Curve", fontsize=14)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("R2 Score", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.show()
plot_learning_curve(X_train.values, X_test.values, y_train.values, y_test.values, best_model)

#Uncertainty Analysis using model above
import seaborn as sns
feature_names = [
    "Equity", "Discount Rate", "Income Tax Rate", "Loan Interest", 
    "Loan Term", "EH % Solids", "FERM Arabinose to Ethanol", 
    "FERM Contamination Losses", "FERM Xylose to Ethanol", "PT Acid Loading", 
    "PT Glucan to Glucose", "EH Cellulose to Glucose", "EH Enzyme Loading", 
    "PT Xylan to Xylose", "Glucose Cost", "Ammonia Cost", "Feedstock Cost"
]

def univariate_uncertainty_with_kde_and_debug(model, X_train, feature_indices, feature_names, num_samples=5000):
    n_features = len(feature_indices)
    rows = 6  
    cols = (n_features + rows - 1) // rows  #
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)
    axes = axes.flatten()  

    total_simulation_time = 0  

    for i, feature_index in enumerate(feature_indices):
        start_time = time.time() 
        varied_samples = np.tile(X_train.iloc[0].values, (num_samples, 1))
        varied_samples[:, feature_index] = np.random.uniform(
            low=X_train.iloc[:, feature_index].min(),
            high=X_train.iloc[:, feature_index].max(),
            size=num_samples
        )

        mfsp_predictions = model.predict(varied_samples)
        
        # Debugging
        mfsp_min = mfsp_predictions.min()
        mfsp_max = mfsp_predictions.max()
        print(f"Feature {feature_names[i]} (index {feature_index}): Predicted MFSP range = [{mfsp_min:.4f}, {mfsp_max:.4f}]")

        end_time = time.time()  
        simulation_time = end_time - start_time
        total_simulation_time += simulation_time

        print(f"Run Time for Feature {feature_names[i]} (index {feature_index}): {simulation_time:.2f} seconds")

        # Histogram 
        ax1 = axes[i]
        sns.histplot(mfsp_predictions, bins=50, kde=True, color="skyblue", line_kws={'lw': 2}, ax=ax1)
        ax1.set_xlabel("MSP ($/GGE)", fontsize=12)
        ax1.set_ylabel("Counts", fontsize=12)
        ax1.set_title(feature_names[i], fontsize=14)
        ax2 = ax1.twinx()
        sns.ecdfplot(mfsp_predictions, color="green", ax=ax2)
        ax2.set_ylabel("Cumulative Probability", fontsize=12)

        for j in range(len(feature_indices), len(axes)):
        fig.delaxes(axes[j])

    plt.show()

univariate_uncertainty_with_kde_and_debug(
    best_model,  
    X_train,
    feature_indices=list(range(17)), 
    feature_names=feature_names
)


