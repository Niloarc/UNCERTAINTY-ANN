#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the dataset
file_path = r"C:\Users\nilou\OneDrive\Documents\PhD-ESF\Sem 03\ML course\Final Project\Paper_data.csv"
df = pd.read_csv(file_path)
df.head()

# Asigning X and y, as MSP is the last column.
X = df.iloc[:, :-1]  # Input features
y = df.iloc[:, -1]   # MSP

# Split the data (75-25 based on literature)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X


# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#Feature dimension reduction analysis with t-SNE (Based on literature)

perplexity_values = [5, 9, 13, 18, 26, 30] #Range of perplexity values to test

# Apply t-SNE for each perplexity value
plt.figure(figsize=(10, len(perplexity_values) * 4))

for i, perplexity in enumerate(perplexity_values):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    tsne_result = tsne.fit_transform(X)
    
    # Get axis limits dynamically based on the data as without it, will not show the whole range
    x_min, x_max = np.min(tsne_result[:, 0]), np.max(tsne_result[:, 0])
    y_min, y_max = np.min(tsne_result[:, 1]), np.max(tsne_result[:, 1])
    x_margin = (x_max - x_min) * 0.1  # Add margin (visualization)
    y_margin = (y_max - y_min) * 0.1
    
    # Plot the result
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


# In[13]:


#Linear SVM model
import time  # Import time module to measure run time
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Define the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('lsvm', LinearSVR(random_state=42, max_iter=10000))  # Linear SVM for regression
])

# Define the grid of hyperparameters to optimize
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

# Start timing the training run
start_time = time.time()

# Perform GridSearchCV to find the optimum parameters
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,  # cross-validation
    scoring='r2',  # Evaluate based on R-squared
    n_jobs=-1,  # Use all computer cores
    verbose=2,
    error_score="raise"  # Raise errors 
)

# Fit the optimum result of grid search on the training set
grid_search.fit(X_train, y_train)

# Stop timing 
end_time = time.time()

# Calculate total run time
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")

# Print the best parameters and their corresponding score
print("Best Parameters:", grid_search.best_params_)
print(f"Best R-squared score: {grid_search.best_score_:.4f}")


# In[15]:


# Evaluate on the test set to check for variance if training and test large difference
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate performance metrics for test set
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Set Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Evaluate the best model on the training set
y_train_pred = grid_search.best_estimator_.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("Training Set Performance:")
print(f"Mean Squared Error (MSE): {train_mse:.4f}")
print(f"Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"R-squared (R2): {train_r2:.4f}")


# Plot true vs. predicted values for test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='teal', edgecolor='k', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='deeppink', linestyle='--', linewidth=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("LSVM True vs Predicted Values")
plt.grid(True)
plt.show()


# In[16]:


#find the range of MSP for predictions
y_pred_min = y_pred.min()
y_pred_max = y_pred.max()
print(f"Predicted MSP Range: [{y_pred_min:.4f}, {y_pred_max:.4f}]")


# In[17]:


#Learning curve analysis

# Function to calculate training and test performance for increasing training sizes
def plot_learning_curve(X_train, X_test, y_train, y_test, model, increments=10):
    train_sizes = np.linspace(10, len(X_train), increments, dtype=int)  # Incremental training sizes
    train_scores = []
    test_scores = []

    for size in train_sizes:
        # Train the model on a subset of the training data
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        model.fit(X_train_subset, y_train_subset)

        # Evaluate performance on the training subset
        y_train_pred = model.predict(X_train_subset)
        train_scores.append(r2_score(y_train_subset, y_train_pred))  # R2 score for training

        # Evaluate performance on the test set
        y_test_pred = model.predict(X_test)
        test_scores.append(r2_score(y_test, y_test_pred))  # R2 score for testing

    # Plotting the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', color='deeppink')
    plt.plot(train_sizes, test_scores, 'o-', label='Test Score', color='teal')

    # Fill between lines for visualization
    plt.fill_between(train_sizes, np.array(train_scores) - 0.05, np.array(train_scores) + 0.05, alpha=0.2, color='orchid')
    plt.fill_between(train_sizes, np.array(test_scores) - 0.05, np.array(test_scores) + 0.05, alpha=0.2, color='darkslategrey')

    # Adding labels and title
    plt.title("LSVM Learning Curve", fontsize=14)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("R2 Score", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.show()


# Plot the learning curve
plot_learning_curve(X_train.values, X_test.values, y_train.values, y_test.values, best_model)


# In[18]:


#Uncertainty Analysis using the trained model of LSVM
import seaborn as sns


# Define feature names for univariate analysis
feature_names = [
    "Equity", "Discount Rate", "Income Tax Rate", "Loan Interest", 
    "Loan Term", "EH % Solids", "FERM Arabinose to Ethanol", 
    "FERM Contamination Losses", "FERM Xylose to Ethanol", "PT Acid Loading", 
    "PT Glucan to Glucose", "EH Cellulose to Glucose", "EH Enzyme Loading", 
    "PT Xylan to Xylose", "Glucose Cost", "Ammonia Cost", "Feedstock Cost"
]

def univariate_uncertainty_with_kde_and_debug(model, X_train, feature_indices, feature_names, num_samples=5000):
    n_features = len(feature_indices)
    rows = 6  # Adjust rows for visualization in presentation
    cols = (n_features + rows - 1) // rows  # Calculate columns dynamically
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)
    axes = axes.flatten()  # Flatten axes for indexing

    total_simulation_time = 0  # Initialize total run time (to compare aganist traditional method)

    for i, feature_index in enumerate(feature_indices):
        start_time = time.time()  # Start timing

        # Generate samples by varying one feature at each run
        varied_samples = np.tile(X_train.iloc[0].values, (num_samples, 1))
        varied_samples[:, feature_index] = np.random.uniform(
            low=X_train.iloc[:, feature_index].min(),
            high=X_train.iloc[:, feature_index].max(),
            size=num_samples
        )

        # Predict MSP
        mfsp_predictions = model.predict(varied_samples)
        
        # Debug: Check the range of predictions
        mfsp_min = mfsp_predictions.min()
        mfsp_max = mfsp_predictions.max()
        print(f"Feature {feature_names[i]} (index {feature_index}): Predicted MFSP range = [{mfsp_min:.4f}, {mfsp_max:.4f}]")

        end_time = time.time()  # End timing
        simulation_time = end_time - start_time
        total_simulation_time += simulation_time

        # Print run time for the feature
        print(f"Run Time for Feature {feature_names[i]} (index {feature_index}): {simulation_time:.2f} seconds")

        # Plot histogram with KDE
        ax1 = axes[i]
        sns.histplot(mfsp_predictions, bins=50, kde=True, color="skyblue", line_kws={'lw': 2}, ax=ax1)
        ax1.set_xlabel("MSP ($/GGE)", fontsize=12)
        ax1.set_ylabel("Counts", fontsize=12)
        ax1.set_title(feature_names[i], fontsize=14)

        # Overlay cumulative probability line
        ax2 = ax1.twinx()
        sns.ecdfplot(mfsp_predictions, color="green", ax=ax2)
        ax2.set_ylabel("Cumulative Probability", fontsize=12)

    # Print total simulation time
    print(f"Total Run Time: {total_simulation_time:.2f} seconds")

    # Hide unused subplots if there is
    for j in range(len(feature_indices), len(axes)):
        fig.delaxes(axes[j])

    plt.show()

# Call the function for univariate analysis 
univariate_uncertainty_with_kde_and_debug(
    best_model,  # The LSVM model which performed best
    X_train,
    feature_indices=list(range(17)),  # All features
    feature_names=feature_names
)


# In[19]:


#Bivariate Uncertainty Analysis ( changing 2 factors at a time)

# Define feature names for bivariate analysis
feature_names = [
    "Equity", "Discount Rate", "Income Tax Rate", "Loan Interest", 
    "Loan Term", "EH % Solids", "FERM Arabinose to Ethanol", 
    "FERM Contamination Losses", "FERM Xylose to Ethanol", "PT Acid Loading", 
    "PT Glucan to Glucose", "EH Cellulose to Glucose", "EH Enzyme Loading", 
    "Glucose Cost", "Ammonia Cost", "Feedstock Cost"
]

# Function definition:
def bivariate_uncertainty(model, X_train, feature_pair, feature_names, grid_size=50):
    # Generate matrix for two features to be changed together
    feature_1_range = np.linspace(
        X_train.iloc[:, feature_pair[0]].min(),
        X_train.iloc[:, feature_pair[0]].max(),
        grid_size
    )
    feature_2_range = np.linspace(
        X_train.iloc[:, feature_pair[1]].min(),
        X_train.iloc[:, feature_pair[1]].max(),
        grid_size
    )
    grid_x, grid_y = np.meshgrid(feature_1_range, feature_2_range)
    grid_samples = np.tile(X_train.iloc[0].values, (grid_size * grid_size, 1))
    grid_samples[:, feature_pair[0]] = grid_x.ravel()
    grid_samples[:, feature_pair[1]] = grid_y.ravel()

    # Predict MSP
    mfsp_predictions = model.predict(grid_samples)
    mfsp_predictions = mfsp_predictions.reshape(grid_size, grid_size)

    # Contour plot with labeled MSP lines
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, mfsp_predictions, cmap="plasma", levels=20)  
    plt.colorbar(contour, label="MSP ($/GGE)")
    contour_lines = plt.contour(grid_x, grid_y, mfsp_predictions, colors="black", linewidths=0.5, levels=10)
    plt.clabel(contour_lines, inline=True, fontsize=10, fmt="%.3f")  

    # Plot formatting
    plt.xlabel(feature_names[feature_pair[0]], fontsize=12)
    plt.ylabel(feature_names[feature_pair[1]], fontsize=12)
    plt.title(f"Bivariate Analysis: {feature_names[feature_pair[0]]} vs {feature_names[feature_pair[1]]}", fontsize=14)
    plt.grid(False)
    plt.show()

# Call the function for 8 bivariate plots (as tehy would be too many to plot all.)
feature_pairs = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (8, 9), (10, 11), (12, 13), (14, 15)
]

for pair in feature_pairs:
    bivariate_uncertainty(
        best_model,  # Use your trained Linear SVM model
        X_train,
        feature_pair=pair,
        feature_names=feature_names,
        grid_size=50  # Adjust resolution if needed
    )


# In[23]:


#Multivariate Uncertainty Analysis

# Define feature groups with their type:
feature_groups = {
    "Financial variables": [0, 1, 2, 3, 4],  
    "Technical variables": [5, 6, 7, 8, 9, 10, 11, 12, 13],  
    "Supply chain variables": [14, 15, 16],  
    "All variables": list(range(17))  # All features together
}

def multivariate_uncertainty_named(model, X_train, feature_groups, num_samples=50000):
    n_groups = len(feature_groups)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)  
    axes = axes.flatten()  # Flatten for index



    for i, (group_name, feature_indices) in enumerate(feature_groups.items()):
        start_time = time.time()  # Start timing the simulation

        # Generate random samples for the current feature group
        varied_samples = np.tile(X_train.iloc[0].values, (num_samples, 1))
        for feature_index in feature_indices:
            varied_samples[:, feature_index] = np.random.uniform(
                low=X_train.iloc[:, feature_index].min(),
                high=X_train.iloc[:, feature_index].max(),
                size=num_samples
            )

        # Predict MFSP
        mfsp_predictions = model.predict(varied_samples)
        simulation_time = time.time() - start_time  

        # Plot histogram and cumulative probability
        ax = axes[i] if n_groups > 1 else axes
        sns.histplot(mfsp_predictions, bins=50, kde=False, color="skyblue", ax=ax, label="Histogram")
        sns.kdeplot(mfsp_predictions, color="blue", ax=ax, label="KDE")
        
        # Add cumulative probability
        counts, bin_edges = np.histogram(mfsp_predictions, bins=50, density=True)
        cumulative = np.cumsum(counts) / np.sum(counts)
        ax2 = ax.twinx()
        ax2.plot(bin_edges[:-1], cumulative, color="green", label="Cumulative probability")
        
        # Custumizing the axis tags
        ax.set_title(f"{group_name} (Simulation Time: {simulation_time:.2f} seconds)", fontsize=16)
        ax.set_xlabel("MSP ($/GGE)", fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        ax2.set_ylabel("Cumulative Probability", fontsize=14, color="green")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.show()


# Call the function
start_time = time.time()

multivariate_uncertainty_named(
    best_model,
    X_train,
    feature_groups=feature_groups
)

total_time = time.time() - start_time
print(f"Total Simulation Time for All Groups: {total_time:.2f} seconds")



# In[ ]:




