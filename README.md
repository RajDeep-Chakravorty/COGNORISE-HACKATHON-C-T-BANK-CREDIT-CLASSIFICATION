# Credit Risk Classification Model

## Overview

This repository contains code for building a credit risk classification model. The model predicts the creditworthiness of individuals based on various features such as account information, credit history, purpose, savings account status, employment status, and others.

## Files

- `Credit_Risk_Classification.ipynb`: Jupyter Notebook containing the code for data preprocessing, exploratory data analysis, model training, evaluation, and hyperparameter tuning.
- `C&T train dataset.csv`: Training dataset containing labeled data for model training.
- `C&T test dataset.csv`: Testing dataset containing labeled data for evaluating model performance.
- `README.md`: This file, containing information about the repository.

## Libraries Used

- `pyforest`: For importing all necessary libraries at once.
- `pygwalker`: For performing exploratory data analysis and visualization.
- `pycaret`: For building and comparing machine learning models.
- `scikit-learn`: For various machine learning algorithms, data preprocessing, and evaluation metrics.
- `xgboost`, `lightgbm`: For gradient boosting algorithms.
- `matplotlib`, `seaborn`: For data visualization.
- `warnings`: For ignoring warnings during execution.

## Usage

1. **Clone the Repository**: Clone this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/<username>/credit-risk-classification.git
   ```

2. **Install Dependencies**: Ensure you have all the necessary libraries installed. You can install them using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**: Open and run the `Credit_Risk_Classification.ipynb` notebook in Jupyter or any compatible environment. Follow the instructions and execute each cell to preprocess the data, train the model, and evaluate its performance.

## Data

The dataset consists of two CSV files:

- `C&T train dataset.csv`: Training dataset containing labeled data for model training.
- `C&T test dataset.csv`: Testing dataset containing labeled data for evaluating model performance.

## Results

The notebook demonstrates the following:

- Data preprocessing techniques including handling missing values, encoding categorical variables, and handling outliers.
- Exploratory data analysis (EDA) to understand the distribution of features and their relationship with the target variable.
- Training and evaluation of multiple machine learning models including Random Forest, Decision Tree, Logistic Regression, Naive Bayes, MLP, Support Vector Machine, and Gradient Boosting.
- Hyperparameter tuning using GridSearchCV to improve model performance.

## Conclusion

Based on the evaluation metrics, the Gradient Boosting model achieved the highest accuracy of 68.75% on the testing dataset. However, further optimization and fine-tuning of models could potentially improve performance.

## Contributors

- [Rajdeep Chakravorty](https://github.com/RajDeep-Chakravorty)
Feel free to contribute by opening issues, proposing new features, or submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/RajDeep-Chakravorty/COGNORISE-HACKATHON-C-T-BANK-CREDIT-CLASSIFICATION/blob/main/LICENSE) file for details.
