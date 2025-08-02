# Hotel Booking Cancellation Predictor

This project addresses a binary classification problem based on a private Kaggle competition involving hotel booking data. The objective was to predict the likelihood of a booking being canceled, evaluated using ROC-AUC as the performance metric.

The final model achieved a cross-validated ROC-AUC score of **0.95**, demonstrating high discriminative power between canceled and non-canceled bookings.

## 🧠 Model Overview

A **CatBoostClassifier** was trained using numerical and categorical features extracted from the dataset. Key features of the pipeline include:

- **Bayesian hyperparameter optimization** using `skopt` and `BayesSearchCV`
- **K-Fold cross-validation** (Stratified, 5 folds)
- **One-hot encoding** for categorical variables
- **MinMaxScaler** normalization for numerical features
- **Early stopping** during training to prevent overfitting

## 🛠️ Tools & Libraries

- Python 3
- Pandas, NumPy
- CatBoost
- scikit-learn
- scikit-optimize (`skopt`)
- ROC-AUC evaluation

## 📂 Dataset

The dataset is private and provided through a restricted Kaggle competition. It includes a training set with labels and a separate test set for final evaluation. To request access, please follow the dataset instructions in `./dataset/README.md`.

## 📈 Performance

| Metric       | Score  |
|--------------|--------|
| ROC-AUC (CV) | 0.95   |

## 📑 Notebook

The full implementation of data preparation, model training, tuning, and evaluation is available in [`hotel_cancellation_predictor.ipynb`](hotel_cancellation_predictor.ipynb).

## 📤 Outputs

Final predictions on the test set are saved to `test_predictions.csv` for submission.

---

*This project was completed independently as part of a private machine learning challenge.*
