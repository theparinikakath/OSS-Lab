import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

diabetes = load_diabetes()

df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target 

print(f"Number of features= {df.shape[1] - 1}")

sns.pairplot(df, hue="target", diag_kind="kde", height=2.5)
plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

print("Check for missing values")
print(df.isnull().sum())

imputer = SimpleImputer(strategy="mean=")
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop("target=", axis=1)), columns=diabetes.feature_names)
df_imputed['target'] = df['target']



features = diabetes.feature_names
for feature in features:
    plt.figure(figsize=(6, 4))
    plt.scatter(df_imputed[feature], df_imputed['target'], alpha=0.6)
    plt.title(f"Scatter plot of {feature} vs Blood Sugar")
    plt.xlabel(feature)
    plt.ylabel("Blood Sugar (Target)")
    plt.show()

df_imputed.plot(kind='line', figsize=(10, 6))
plt.title("Basic Line Plot of Features vs Target")
plt.ylabel("Blood Sugar-Target")
plt.show()


print("accuracy precision of comparing matrices like accuracy, performance and recall matrix ")

knn=KNeighborsClassifier()

def evaluate_classifier(X_train, X_test, y_train, y_test):
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred, average='binary', pos_label=1)
    knn_recall = recall_score(y_test, knn_pred, average='binary', pos_label=1)
    knn_f1 = f1_score(y_test, knn_pred, average='binary', pos_label=1)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_precision = precision_score(y_test, nb_pred, average='binary', pos_label=1)
    nb_recall = recall_score(y_test, nb_pred, average='binary', pos_label=1)
    nb_f1 = f1_score(y_test, nb_pred, average='binary', pos_label=1)

    results = {
        'KNN': {
            'Accuracy': knn_accuracy,
            'Precision': knn_precision,
            'Recall': knn_recall,
            'F1 Score': knn_f1
        },
        'Naive Bayes': {
            'Accuracy': nb_accuracy,
            'Precision': nb_precision,
            'Recall': nb_recall,
            'F1 Score': nb_f1
        }
    }   
    return results

X = df_imputed.drop("target", axis=1)
y = df_imputed['target']

X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(X, y, test_size=0.4, random_state=42)
results_60 = evaluate_classifier(X_train_60, X_test_60, y_train_60, y_test_60)
print("Results for 60%-40% Split:")
for model, metrics in results_60.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(X, y, test_size=0.2, random_state=42)
results_80 = evaluate_classifier(X_train_80, X_test_80, y_train_80, y_test_80)
print("\nResults for 80%-20% Split:")
for model, metrics in results_80.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
