# Predicting Annual Medical Insurance Premiums with Linear Regression

An end-to-end linear regression project that estimates the **annual premium** an insurer should charge a member, using demographic, lifestyle and clinical attributes. Linear regression was chosen because underwriting and compliance teams need to see how much each factor contributes to the price.

## Business problem

A health insurer wants a transparent, explainable pricing model. A black-box model is hard to justify to regulators and customers, while a linear model shows how much each factor (age, BMI, smoking, chronic conditions) adds to the premium.

- **Target:** `annual_premium` (USD, continuous), so this is a **regression** problem.
- **Success metrics:** R², adjusted R², MAE, RMSE.

## Data

`insurance_premium.csv`: 2,500 members, 12 columns.

| Type | Columns |
|---|---|
| Numeric | `age`, `bmi`, `children`, `chronic_conditions`, `annual_income`, `blood_pressure`, `cholesterol` |
| Categorical | `sex`, `smoker`, `region`, `exercise_freq` |
| Target | `annual_premium` |

Missing values: `bmi` 6%, `annual_income` 5%, `blood_pressure` 4%, `cholesterol` 4%, `exercise_freq` 3%, `smoker` 2%.

> **Data source:** _add where the dataset comes from (for example, a public dataset link or "synthetic teaching dataset")._

## Approach

1. **Data audit:** shape, types, summary statistics, missing-value counts.
2. **EDA:** target distribution (skewness 1.11), numeric histograms, category counts, premium by category, scatter plots coloured by smoker.
3. **Correlations (before and after cleaning):** age (0.45), BMI (0.25) and chronic conditions (0.22) correlate most with the premium.
4. **Outliers:** IQR rule, values capped (winsorised) at the bounds.
5. **Missing values:** mode for categorical columns, KNN imputation (k = 5) for numeric columns.
6. **Hypothesis tests:** Welch t-test, one-way ANOVA, chi-square.
7. **Encoding:** 0/1 for `sex` and `smoker`, one-hot with `drop_first=True` for `region` and `exercise_freq`.
8. **Split and scaling:** 60/20/20 train/validation/test, `StandardScaler` fitted on the training data.
9. **Assumption checks:** VIF, Durbin-Watson, Breusch-Pagan, Jarque-Bera, Q-Q plot, residual plots.
10. **Model and validation:** OLS, 5-fold cross-validation, Ridge, Lasso and Elastic Net.

## Results

### Hypothesis tests

| Feature | Test | Result |
|---|---|---|
| `smoker` | Welch t-test | Significant (mean premium about $39.9k vs $26.1k) |
| `exercise_freq` | ANOVA | Significant (p = 2.9e-07) |
| `sex` | Welch t-test | Not significant (p = 0.91) |
| `region` | ANOVA | Not significant (p = 0.62) |
| smoker vs region | Chi-square | Independent (p = 0.70) |

### Assumption checks

| Assumption | Check | Result |
|---|---|---|
| No multicollinearity | VIF | All below 1.6 |
| Independent errors | Durbin-Watson | 2.109 (pass) |
| Constant variance | Breusch-Pagan | p = 0.53 (pass) |
| Normal residuals | Jarque-Bera | p about 0 (**fails**) |

### Model performance

| Split | R² |
|---|---|
| Train | 0.884 |
| Validation | 0.848 |
| Test | 0.868 |

- **Adjusted R² (test):** 0.864
- **Test MAE:** $1,878 · **Test RMSE:** $2,658
- **5-fold CV R²:** mean 0.873 (std 0.025)

### Regularization

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear regression | 0.868 | $2,658 |
| Ridge (alpha = 1.0) | 0.868 | $2,658 |
| Lasso (alpha = 1.0) | 0.868 | $2,658 |
| Elastic Net (alpha = 1.0, l1_ratio = 0.5) | 0.773 | $3,487 |

Ridge and Lasso matched plain regression because at alpha = 1.0 the penalty is tiny compared with the coefficients (hundreds to thousands of dollars), so Lasso did not zero out any feature. Elastic Net shrank every coefficient to roughly two-thirds of its OLS value, which explains the lower score. Alpha should be tuned with cross-validation.

### Key drivers (coefficients on standardised features, USD per 1 standard deviation)

| Feature | Coefficient |
|---|---|
| `smoker` | +5,542 |
| `age` | +3,460 |
| `bmi` | +1,908 |
| `chronic_conditions` | +1,587 |
| `exercise_freq_regular` | −827 |

## Conclusion

Smoking is by far the strongest driver of the premium, followed by age, BMI and chronic conditions. The model explains about 87% of the variation on unseen data with an average error of about $1,900 (around 6.5% of the mean premium), and the train, validation and test scores are close, so there is no sign of overfitting.

## Limitations and next steps

- **Residuals are not normal** (Jarque-Bera rejects), so p-values and confidence intervals are less reliable. Predictions and coefficients are still usable.
- **The target was capped** at the IQR upper bound (about $47,500, 42 rows), so error figures apply to capped premiums and are optimistic for very high premiums.
- **Preprocessing was fitted on the full dataset** (outlier bounds, imputation) before the split. A scikit-learn `Pipeline` would remove this leakage.
- **Coefficients are per standard deviation**, not per raw unit. Refit on unscaled features if you need dollars per unit.
- Next steps: add a BMI × smoker interaction, try a log-transformed target, tune alpha with `RidgeCV` / `LassoCV`, and compare against a tree-based model.

## How to run

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
```

1. Put `insurance_premium.csv` in the same folder as the notebook.
2. In the notebook, change the data path from `/content/insurance_premium.csv` to `insurance_premium.csv`.
3. Run all cells: `jupyter notebook insurance_premium_linear_regression.ipynb`

## Repository contents

| File | Description |
|---|---|
| `insurance_premium_linear_regression.ipynb` | Full analysis, code and outputs |
| `insurance_premium.csv` | Dataset |
| `README.md` | This file |
