import os
import glob
import zipfile
import pandas as pd
import pvlib
from sklearn import linear_model, tree, neighbors, ensemble, decomposition,neural_network
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

# reads irradiance features
inpEndo = pd.read_csv(
    "Irradiance_features_intra-hour.csv",
    delimiter=",",
    parse_dates=True,
    index_col=0,
)
# reads sky image features
inpExo = pd.read_csv(
    "Sky_image_features_intra-hour.csv",
    delimiter=",",
    parse_dates=True,
    index_col=0,
)
# reads target
tar = pd.read_csv(
    "Target_intra-hour.csv",
    delimiter=",",
    parse_dates=True,
    index_col=0,
)

def run_forecast(target,horizon):
    cols = [
            "{}_{}".format(target,horizon),  # actual
            "{}_kt_{}".format(target,horizon),  # clear-sky index
            "{}_clear_{}".format(target,horizon),  # clear-sky model
            "elevation_{}".format(horizon)   # solar elevation 
        ]
    #这个是上面所对应的的数据，将所有的数据对应读入

    train = inpEndo[inpEndo.index.year <= 2015]
    train = train.join(inpExo[inpEndo.index.year <= 2015], how="inner")
    train = train.join(tar[tar.index.year <= 2015], how="inner")

    test = inpEndo[inpEndo.index.year == 2016]
    test = test.join(inpExo[inpEndo.index.year == 2016], how="inner")
    test = test.join(tar[tar.index.year == 2016], how="inner")
    
    feature_cols = inpEndo.filter(regex=target).columns.tolist()
    feature_cols_endo = inpEndo.filter(regex=target).columns.tolist()
    feature_cols.extend(inpExo.columns.unique().tolist())
        
    train = train[cols + feature_cols].dropna(how="any")
    test  = test[cols + feature_cols].dropna(how="any")

    train_X = train[feature_cols].values
    test_X  = test[feature_cols].values
    train_X_endo = train[feature_cols_endo].values
    test_X_endo  = test[feature_cols_endo].values

    train_y = train["{}_kt_{}".format(target,horizon)].values
    test_y = test["{}_kt_{}".format(target,horizon)].values
    elev_train = train["elevation_{}".format(horizon)].values
    elev_test  = test["elevation_{}".format(horizon)].values

    train_clear = train["{}_clear_{}".format(target,horizon)].values
    test_clear = test["{}_clear_{}".format(target,horizon)].values
  
    # train forecast models
    models = [
        # Ordinary Least-Squares (OLS)
        ["ols", linear_model.LinearRegression()],
        # Ridge Regression (OLS + L2-regularizer)
        ["ridge", linear_model.RidgeCV(cv=10)],
        # Lasso (OLS + L1-regularizer)
        ["lasso", linear_model.LassoCV(cv=10, n_jobs=-1, max_iter=10000)],
        #elastic net
        ["enet", linear_model.ElasticNetCV(cv=10,max_iter=10000)],
        #decision tree
        ["dec", tree.DecisionTreeRegressor(max_depth=8)],
        #randomforest
        ["rf", ensemble.RandomForestRegressor(n_estimators=200,max_depth=27,min_samples_leaf=60,criterion='mse',oob_score=True,max_features=None,min_samples_split=125)],
    ]
    
    for Xtra,Xtes,f in zip([train_X_endo,train_X],[test_X_endo,test_X],['endo','exo']):
        # normalize features
        scaler = StandardScaler()
        scaler.fit(Xtra)
        Xtra = scaler.transform(Xtra)
        # pca = decomposition.PCA(n_components="mle")
        # Xtra = pca.fit_transform(Xtra)
        Xtes = scaler.transform(Xtes)
        # Xtes = pca.transform(Xtes)

        for name, model in models:
            # train and forecast
            if name =="rf":
                min_samples_split = range(110,130)
                max_depth = range(20,35)
                max_features = np.arange(0.3,0.9,0.1)
                min_samples_leaf = range(50,70)
                parameters = {
                              "min_samples_split":min_samples_split,
                              "max_depth":max_depth,
                              "max_features":max_features,
                              "min_samples_leaf":min_samples_leaf}
                rf = model_selection.GridSearchCV(model,
                                                     parameters,
                                                     cv=5,
                                                     scoring="neg_mean_squared_error",
                                                     verbose=1,
                                                     refit=True,
                                                     n_jobs=-1)
                rf.fit(Xtra,train_y)
                train_pred = rf.predict(Xtra)
                test_pred = rf.predict(Xtes)
            if name == "dec":
                min_samples_split = range(100, 135)
                max_depth = range(2, 15)
                min_samples_leaf = range(45,60)
                parameters = {
                    "min_samples_split": min_samples_split,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf}
                dec = model_selection.GridSearchCV(model,
                                                     parameters,
                                                     cv=5,
                                                     scoring="neg_mean_squared_error",
                                                     verbose=1,
                                                     refit=True,
                                                     n_jobs=-1)
                dec.fit(Xtra,train_y)
                train_pred = dec.predict(Xtra)
                test_pred = dec.predict(Xtes)
            else:
                model.fit(Xtra,train_y)
                train_pred = model.predict(Xtra)
                test_pred = model.predict(Xtes)

            # limits forecasted kt to [0,1.1]
            train_pred[train_pred < 0] = 0
            test_pred[test_pred < 0] = 0
            train_pred[train_pred > 1.1] = 1.1
            test_pred[test_pred > 1.1] = 1.1

            # convert from kt [-] back to irradiance [W/m^2]
            train_pred *= train_clear
            test_pred *= test_clear

            # removes nighttime values (solar elevation < 5)
            train_pred[elev_train < 5] = np.nan
            test_pred[elev_test < 5] = np.nan
            
            train.insert(train.shape[1], "{}_{}_{}".format(target, name,f), train_pred)
            test.insert(test.shape[1], "{}_{}_{}".format(target, name,f), test_pred)
        
    # smart persistence forecast
    # uses the shortest backward average as the "current" kt value
    tmp = np.squeeze(train[["B({}_kt|5min)".format(target)]].values) * train_clear
    tmp[elev_train < 5] = np.nan
    train.insert(train.shape[1], "{}_sp".format(target), tmp)
    tmp = np.squeeze(test[["B({}_kt|5min)".format(target)]].values) * test_clear
    tmp[elev_test < 5] = np.nan
    test.insert(test.shape[1], "{}_sp".format(target), tmp)
    

    # saves forecasts
    # only keep essential forecast columns (true, clear-sky, and forecasted values)
    cols = train.columns[train.columns.str.startswith("{}".format(target))]
    train = train[cols]
    test = test[cols]
    
    # add metadata
    train.insert(train.shape[1], "dataset", "Train")
    test.insert(test.shape[1], "dataset", "Test")
    df = pd.concat([train, test], axis=0)
    df.insert(df.shape[1], "target", target)
    df.insert(df.shape[1], "horizon", horizon)
    df.to_hdf(
        os.path.join(
            "PCA",
            "forecasts_intra-hour_horizon={}_{}.h5".format(
                horizon, target
            ),
        ),
        "df",
        mode="w",
    )

# runs forecast for all variables and horizons
target  = ["ghi","dni"]
horizon = ["5min","10min","15min","20min","25min","30min"]
for t in target:
    for h in horizon:
        print("{} IH forecast for {}".format(h,t))
        run_forecast(t,h)

#三个预测的代码除了输入不一样之外，处理方法相似；