# ML-Powered Breast Cancer Diagnosis 👩‍⚕️🩺

This project develops and evaluates several machine learning models to accurately predict breast cancer diagnosis (benign vs. malignant) based on features from cell nuclei images.

## 🎯 Project Goal

The primary goal of this project is to compare the performance of multiple classification algorithms to identify the most effective and reliable model for breast cancer diagnosis. This project showcases the end-to-end machine learning pipeline, from data preprocessing to final model deployment considerations.

## 🗂️ Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
* **Description:** The dataset contains 569 instances with 30 real-valued features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image. The target variable is the diagnosis: Malignant (M) or Benign (B).

---

## 🚀 Methodology

The project follows a standard machine learning workflow, which is fully documented in the provided Jupyter Notebook.

1.  **Data Loading and Preprocessing**: The raw data was loaded from a `.data` file, and descriptive column names were assigned. Irrelevant features (like the `id` column) were dropped, and the categorical diagnosis labels ('M', 'B') were encoded into numerical values (1 and 0).
2.  **Exploratory Data Analysis (EDA)**: A visual analysis was performed to understand the distribution of features, check for class imbalance, and identify correlations between features using histograms and a correlation heatmap.
3.  **Data Splitting and Scaling**: The dataset was split into training (80%) and testing (20%) sets. All numerical features were then standardized using `StandardScaler` to ensure all models performed optimally.
4.  **Model Selection & Training**: The following classification models were trained and evaluated:
    * Logistic Regression
    * Support Vector Machine (SVM)
    * Decision Tree
    * Random Forest
    * K-Nearest Neighbors (KNN)
5.  **Hyperparameter Tuning**: The top-performing models (SVM and Random Forest) were fine-tuned using `GridSearchCV` to find their optimal parameters and further improve their performance.

---

## 📈 Results and Conclusion

### Model Performance

The initial model comparison revealed that **Support Vector Machine (SVM)** and **Random Forest** were the top performers. After hyperparameter tuning, the final evaluation on the test set confirmed the SVM model as the most effective.

* **Tuned SVM F1-Score**: 0.9756
* **Tuned Random Forest F1-Score**: 0.9630

### Confusion Matrix

A confusion matrix for the final SVM model shows its prediction accuracy on the test set.

****

The matrix clearly shows the model's high rate of correct predictions (True Positives and True Negatives) and its minimal number of errors, which is critical for a medical diagnosis application.

### Conclusion

The project successfully identified the **Support Vector Machine (SVM)** as the most suitable model for this breast cancer diagnosis task. Its high F1-score and low error rate on unseen data demonstrate its reliability. While other models also performed well, the tuned SVM provides the best balance of precision and recall, making it a strong candidate for real-world applications.

---

## 🛠️ Technologies and Libraries

* Python
* Google Colab
* Pandas
* Numpy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 🧑‍💻 Author

**Anu Thomson** - [GitHub Profile](https://github.com/AnuThmsn) - [LinkedIn Profile](https://www.linkedin.com/in/anu-thomson/)
