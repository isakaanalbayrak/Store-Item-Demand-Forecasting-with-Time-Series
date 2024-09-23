#####################################################
# Demand Forecasting
#####################################################

# Store Item Demand Forecasting Challenge

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

# Setting pandas options to display all columns and increase width for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

# Function to display basic information about the DataFrame
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

# Loading train, test, and sample submission datasets
train = pd.read_csv('datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('datasets/demand_forecasting/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('datasets/demand_forecasting/sample_submission.csv')

# Concatenating train and test datasets
df = pd.concat([train, test], sort=False)

#####################################################
# EDA (Exploratory Data Analysis)
#####################################################

# Checking the date range in the dataset
df["date"].min(), df["date"].max()

# Displaying basic information using check_df function
check_df(df)

# Unique number of stores and items
df[["store"]].nunique()
df[["item"]].nunique()

# Aggregating sales by store and item
df.groupby(["store"])["item"].nunique()
df.groupby(["store", "item"]).agg({"sales": ["sum"]})
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

df.head()


#####################################################
# FEATURE ENGINEERING
#####################################################

# Feature engineering for date-based features
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.isocalendar().week  # Updated to isocalendar() for compatibility
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

# Apply the feature engineering function
df = create_date_features(df)

# Aggregating sales data by store, item, and month
df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


########################
# Random Noise
########################

# Function to add random noise to the data
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# Lag/Shifted Features
########################

# Sorting the DataFrame by store, item, and date
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

# Showing how lagged features work
pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})

df.groupby(["store", "item"])['sales'].head()

# Function to create lag features for the dataset
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# Applying the lag features
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

# Checking the DataFrame after adding lag features
check_df(df)

########################
# Rolling Mean Features
########################

# Showing how rolling mean features work
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

# Function to create rolling mean features
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# Applying rolling mean features
df = roll_mean_features(df, [365, 546])

########################
# Exponentially Weighted Mean Features
########################

# Showing how exponentially weighted mean features work
pd.DataFrame({"sales": df["sales"].values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

# Function to create exponentially weighted mean features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

# Applying exponentially weighted mean features
alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

# Checking the DataFrame after adding exponentially weighted mean features
check_df(df)

########################
# One-Hot Encoding
########################

# Performing one-hot encoding on categorical variables
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

check_df(df)


########################
# Converting sales to log(1+sales)
########################

# Applying log transformation to the sales data
df['sales'] = np.log1p(df["sales"].values)

check_df(df)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# Custom SMAPE function for LightGBM evaluation
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# Custom LightGBM SMAPE function
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


########################
# Time-Based Validation Sets
########################

# Splitting the data into training and validation sets
train = df.loc[(df["date"] < "2017-01-01"), :]
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# Selecting features for the model
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

# Checking the shapes of training and validation sets
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# Creating LightGBM datasets
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

# Training the LightGBM model
model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

# Making predictions for the validation set
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# Calculating SMAPE for the validation set
smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Feature Importance
########################

# Function to plot feature importance
def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

# Plotting and showing feature importances
plot_lgb_importances(model, num=200)
plot_lgb_importances(model, num=30, plot=True)


# Get features with non-zero importance
feat_imp = plot_lgb_importances(model, num=200)
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values
imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

# Using the final dataset to train the model
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

# Preparing the test dataset
test = df.loc[df.sales.isna()]
X_test = test[cols]

# Final LightGBM model with best iteration
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# Training the final model
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

# Making predictions for the test set
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)


########################
# Submission File
########################

# Preparing the submission file
submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)

# Saving the submission file
submission_df.to_csv("submission_demand.csv", index=False)
