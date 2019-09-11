import os
import glob
import zipfile
import pandas as pd
import pvlib
from sklearn import ensemble, metrics, linear_model,model_selection,decomposition
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import sys
from tkinter import _flatten
import matplotlib.pyplot as plt

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

def run_forecast(target, horizon):
    cols = [
        "{}_{}".format(target, horizon),  # actual
        "{}_kt_{}".format(target, horizon),  # clear-sky index
        "{}_clear_{}".format(target, horizon),  # clear-sky model
        "elevation_{}".format(horizon)  # solar elevation
    ]

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
    test = test[cols + feature_cols].dropna(how="any")

    train_X = train[feature_cols].values
    test_X = test[feature_cols].values
    train_X_endo = train[feature_cols_endo].values
    test_X_endo = test[feature_cols_endo].values

    train_y = train["{}_kt_{}".format(target, horizon)].values
    test_y = test["{}_kt_{}".format(target,horizon)].values
    elev_train = train["elevation_{}".format(horizon)].values
    elev_test = test["elevation_{}".format(horizon)].values

    train_clear = train["{}_clear_{}".format(target, horizon)].values
    test_clear = test["{}_clear_{}".format(target, horizon)].values

    # train forecast models
    models = [
        ["ols", linear_model.LinearRegression()],
        # Ridge Regression (OLS + L2-regularizer)
        ["ridge", linear_model.RidgeCV(cv=10)],
        # Lasso (OLS + L1-regularizer)
        ["lasso", linear_model.LassoCV(cv=10, n_jobs=-1, max_iter=10000)],
        #elastic net
        ["enet", linear_model.ElasticNetCV(cv=10, max_iter=10000)],
        ]
    samples = [
        ["3", 0.03],
        ["5", 0.05],
        ["10", 0.1],
    ]


    for Xtra, Xtes, f in zip([train_X_endo, train_X], [test_X_endo, test_X], ['endo','exo']):

        scaler = StandardScaler()
        scaler.fit(Xtra)
        Xtra = scaler.transform(Xtra)
        Xtes = scaler.transform(Xtes)
        clf = cluster.KMeans(n_clusters=100)
        clf.fit(Xtra)
        # Xtra1 = clf.transform(Xtra)
        r = pd.DataFrame(clf.labels_).groupby(by=clf.labels_).groups
        indices =[]
        for number, sample in samples:
            for i in r:
                ind = random.sample(list(r[i]),int(sample*len(r[i])))
                indices.append(ind)
            indexes = list(_flatten(indices))
            new_Xtra = Xtra[indexes]
            new_train_y = train_y[indexes]
            new_elev_train = elev_train[indexes]
            new_train_clear = train_clear[indexes]

            for name, model in models:
                model.fit(new_Xtra,new_train_y)
                train_pred = model.predict(new_Xtra)
                test_pred = model.predict(Xtes)

                # limits forecasted kt to [0,1.1]
                train_pred[train_pred < 0] = 0
                test_pred[test_pred < 0] = 0
                train_pred[train_pred > 1.1] = 1.1
                test_pred[test_pred > 1.1] = 1.1

                # convert from kt [-] back to irradiance [W/m^2]
                train_pred *= new_train_clear
                test_pred *= test_clear

                # removes nighttime values (solar elevation < 5)
                train_pred[new_elev_train < 5] = np.nan
                test_pred[elev_test < 5] = np.nan
                test.insert(test.shape[1], "{}_{}_{}_{}".format(target, name, f, number), test_pred)

    # NAM forecast
    # tmp = new_train["nam_{}_{}".format(target, horizon)].values
    # tmp[new_elev_train < 5] = np.nan
    # new_train.insert(new_train.shape[1], "{}_nam".format(target), tmp)
    tmp = np.squeeze(test[["B({}_kt|5min)".format(target)]].values) * test_clear
    tmp[elev_test < 5] = np.nan
    test.insert(test.shape[1], "{}_sp".format(target), tmp)

    # saves forecasts
    # only keep essential forecast columns (true, clear-sky, and forecasted values)
    cols = test.columns[test.columns.str.startswith("{}".format(target))]
    # new_train = new_train[cols]
    test = test[cols]

    # add metadata
    # new_train.insert(new_train.shape[1], "dataset", "Train")
    test.insert(test.shape[1], "dataset", "Test")
    df = pd.DataFrame(test)
    # df = pd.concat([new_train, test], axis=0)
    df.insert(df.shape[1], "target", target)
    df.insert(df.shape[1], "horizon", horizon)
    df.to_hdf(
        os.path.join(
            "new",
            "forecasts_intra-hour_horizon={}_{}.h5".format(
                horizon, target
            ),
        ),
        "df",
        mode="w",
    )


# runs forecast for all variables and horizons
target = ["dni","ghi"]
horizon = ["5min", "10min", "15min", "20min", "25min", "30min"]
for t in target:
    for h in horizon:
        print("{} DA forecast for {}".format(h, t))
        run_forecast(t, h)
