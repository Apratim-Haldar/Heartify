from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'heart.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Univariate Analysis

# 1. Distribution of numerical features
def plot_histograms(df, columns):
    df[columns].hist(bins=15, figsize=(15, 6), layout=(2, 4))
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

# Numerical columns
numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Plot histograms
plot_histograms(df, numerical_columns)

# 2. Boxplots for numerical features to detect outliers
def plot_boxplots(df, columns):
    df[columns].plot(kind='box', subplots=True, layout=(2, 4), figsize=(15, 6))
    plt.suptitle("Boxplots of Numerical Features")
    plt.show()

# Plot boxplots
plot_boxplots(df, numerical_columns)

# 3. Bar charts for categorical features
def plot_bar_charts(df, columns):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, col in enumerate(columns):
        sns.countplot(x=col, data=df, ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f"Count of {col}")
    plt.tight_layout()
    plt.suptitle("Bar Charts of Categorical Features", y=1.02)
    plt.show()

# Categorical columns
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Plot bar charts
plot_bar_charts(df, categorical_columns)

# 4. Summary Statistics
print("Summary Statistics for Numerical Features:\n")
print(df[numerical_columns].describe())

# Bivariate Analysis

# 1. Correlation Matrix and Heatmap for numerical features
def plot_correlation_matrix(df, columns):
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

# Plot correlation matrix and heatmap
plot_correlation_matrix(df, numerical_columns)

# 2. Scatter plots for numerical features vs. target
def plot_scatter_plots(df, columns, target):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, col in enumerate(columns):
        sns.scatterplot(x=col, y=target, data=df, ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f"{col} vs {target}")
    plt.tight_layout()
    plt.suptitle(f"Scatter Plots of Numerical Features vs {target}", y=1.02)
    plt.show()

# Plot scatter plots
plot_scatter_plots(df, numerical_columns, 'HeartDisease')

# 3. Boxplots/Violin plots for categorical features vs. target
def plot_categorical_vs_target(df, columns, target):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, col in enumerate(columns):
        sns.boxplot(x=col, y=target, data=df, ax=axes[i//3, i%3])
        sns.violinplot(x=col, y=target, data=df, ax=axes[i//3, i%3], inner=None, color="0.8")
        axes[i//3, i%3].set_title(f"{col} vs {target}")
    plt.tight_layout()
    plt.suptitle(f"Boxplots and Violin Plots of Categorical Features vs {target}", y=1.02)
    plt.show()

# Plot boxplots and violin plots
plot_categorical_vs_target(df, categorical_columns, 'HeartDisease')





# Step 1: Data Preprocessing
def load_and_preprocess_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert categorical variables into numerical ones using OneHotEncoder
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # Updated argument name
    encoded_categorical_data = encoder.fit_transform(df[categorical_columns])

    # Convert the encoded data to a DataFrame
    encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical columns and add the encoded columns
    df_encoded = df.drop(columns=categorical_columns).join(encoded_df)

    # Define the features (X) and target (y)
    X = df_encoded.drop(columns=['HeartDisease'])
    y = df_encoded['HeartDisease']

    # Split the dataset into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder

# Step 2: Model Training and Evaluation
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Initialize the Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)

    # Train the model on the training data
    rf_model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf_model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Feature importance from the Random Forest
    feature_importance = rf_model.feature_importances_

    return rf_model, accuracy, precision, recall, f1, classification_rep, feature_importance

# Step 3: Save the Trained Model and Encoder
def save_model_and_encoder(model, encoder, model_filename='rf_model.pkl', encoder_filename='encoder.pkl'):
    # Save the trained model
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    # Save the encoder
    with open(encoder_filename, 'wb') as file:
        pickle.dump(encoder, file)

# Step 4: Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Initialize the Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    
    # Initialize GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                               cv=5, n_jobs=-1, verbose=2, scoring='f1')
    
    # Perform the grid search
    grid_search.fit(X_train, y_train)
    
    # Best hyperparameters
    best_params = grid_search.best_params_
    
    # Best model
    best_model = grid_search.best_estimator_
    
    return best_model, best_params

# Example Usage
if __name__ == "__main__":
    # File path to the CSV data
    file_path = 'heart.csv'  # Update with your actual file path

    # Load and preprocess data
    X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data(file_path)

    # Train and evaluate the initial model
    rf_model, accuracy, precision, recall, f1, classification_rep, feature_importance = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    
    # Display evaluation metrics of the initial model
    print(f"Initial Model Accuracy: {accuracy}")
    print(f"Initial Model Precision: {precision}")
    print(f"Initial Model Recall: {recall}")
    print(f"Initial Model F1 Score: {f1}")
    print("Initial Model Classification Report:\n", classification_rep)

    # Save the initial model and encoder
    save_model_and_encoder(rf_model, encoder, model_filename='rf_model.pkl', encoder_filename='encoder.pkl')
    
    # Perform hyperparameter tuning
    tuned_model, best_params = hyperparameter_tuning(X_train, y_train)
    
    # Evaluate the tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned)
    recall_tuned = recall_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned)
    
    # Display evaluation metrics of the tuned model
    print(f"Tuned Model Accuracy: {accuracy_tuned}")
    print(f"Tuned Model Precision: {precision_tuned}")
    print(f"Tuned Model Recall: {recall_tuned}")
    print(f"Tuned Model F1 Score: {f1_tuned}")
    
    # Compare performance
    if f1_tuned > f1:
        print("The tuned model performed better. Consider using this model.")
        # Save the tuned model if it is better
        save_model_and_encoder(tuned_model, encoder, model_filename='tuned_rf_model.pkl', encoder_filename='encoder.pkl')
        print("Tuned model saved as 'tuned_rf_model.pkl'.")
    else:
        print("The initial model performed better or equally well. The tuned model was not saved.")
