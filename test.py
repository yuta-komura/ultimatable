import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from lib import indicator, message, repository

tf.compat.v1.disable_eager_execution()


def calc_sma(df: pd.DataFrame, use_columns: str):
    for i in range(800):
        df = indicator.sma(df=df, value=i + 1, use_columns=use_columns)
    return df


def calc_ema(df: pd.DataFrame, use_columns: str):
    for i in range(800):
        df = indicator.ema(df=df, value=i + 1, use_columns=use_columns)
    return df


def calc_rsi(df: pd.DataFrame, use_columns: str):
    for i in range(800):
        df = indicator.rsi(df=df, value=i + 1, use_columns=use_columns)
    return df


def calc_indicators(df: pd.DataFrame, use_columns: str):
    df = calc_sma(df=df, use_columns=use_columns)
    df = calc_ema(df=df, use_columns=use_columns)
    df = calc_rsi(df=df, use_columns=use_columns)
    return df


def learnig(shift):
    n = shift.shape[0]
    train_start = 0
    train_end = int(np.floor(0.8 * n))
    test_start = train_end + 1
    test_end = n

    data_train = shift.iloc[np.arange(train_start, train_end), :]
    data_test = shift.iloc[np.arange(test_start, test_end), :]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train_norm = scaler.transform(data_train)
    data_test_norm = scaler.transform(data_test)

    X_train = data_train_norm[:, 1:]
    y_train = data_train_norm[:, 0]
    X_test = data_test_norm[:, 1:]
    y_test = data_test_norm[:, 0]

    y_test = y_test.reshape(len(data_test), 1)
    test_inv = np.concatenate((y_test, X_test), axis=1)
    test_inv = scaler.inverse_transform(test_inv)

    n_stocks = X_train.shape[1]

    n_neurons_1 = 256
    n_neurons_2 = 128

    net = tf.compat.v1.InteractiveSession()

    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

    sigma = 1
    weight_initializer = tf.compat.v1.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    W_out = tf.Variable(weight_initializer([n_neurons_2, 1]))
    bias_out = tf.Variable(bias_initializer([1]))

    hi_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hi_2 = tf.nn.sigmoid(tf.add(tf.matmul(hi_1, W_hidden_2), bias_hidden_2))

    out = tf.transpose(tf.add(tf.matmul(hi_2, W_out), bias_out))
    mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))
    opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)
    net.run(tf.compat.v1.global_variables_initializer())

    epochs = 100
    for e in range(epochs):
        net.run(opt, feed_dict={X: X_train, Y: y_train})

    pred_test = net.run(out, feed_dict={X: X_test})

    pred_test = np.concatenate((pred_test.T, X_test), axis=1)
    pred_inv = scaler.inverse_transform(pred_test)

    mae_test = mean_absolute_error(test_inv, pred_inv)
    message.info("mae", mae_test)

    return pred_inv[len(pred_inv) - 1]


database = "tradingbot"

analysis_width = 1000

sql = "truncate future_ohlcv_1min_bitflyer_perp"
repository.execute(database=database, sql=sql)

sql = """
        select
            perp.date,
            perp.open,
            perp.high,
            perp.low,
            perp.close,
            perp.volume
        from
            ohlcv_1min_bitflyer_perp perp
        order by
            date
        """
basis = repository.read_sql(database=database, sql=sql)
basis = basis.set_index("date")

rule = "1min"
basis["open"] = basis["open"].resample(rule).first()
basis["high"] = basis["high"].resample(rule).max()
basis["low"] = basis["low"].resample(rule).min()
basis["close"] = basis["close"].resample(rule).last()
basis["volume"] = basis["volume"].resample(rule).sum()
basis = basis.dropna()
basis = basis.reset_index()

for i in range(len(basis) - 1 - analysis_width):

    feature = basis.copy()[i:i + analysis_width]

    analysis = feature.copy()
    del analysis["date"]
    analysis = calc_indicators(df=analysis, use_columns="open")
    analysis = calc_indicators(df=analysis, use_columns="high")
    analysis = calc_indicators(df=analysis, use_columns="low")
    analysis = calc_indicators(df=analysis, use_columns="close")

    shift_open = analysis.copy()
    shift_open["open"] = shift_open["open"].shift(-1)
    shift_open = shift_open.dropna()
    future_open = int(learnig(shift=shift_open)[0])

    shift_high = analysis.copy()
    shift_high["high"] = shift_high["high"].shift(-1)
    shift_high = shift_high.dropna()
    future_high = int(learnig(shift=shift_high)[1])

    shift_low = analysis.copy()
    shift_low["low"] = shift_low["low"].shift(-1)
    shift_low = shift_low.dropna()
    future_low = int(learnig(shift=shift_low)[2])

    shift_close = analysis.copy()
    shift_close["close"] = shift_close["close"].shift(-1)
    shift_close = shift_close.dropna()
    future_close = int(learnig(shift=shift_close)[3])

    shift_volume = analysis.copy()
    shift_volume["volume"] = shift_volume["volume"].shift(-1)
    shift_volume = shift_volume.dropna()
    future_volume = learnig(shift=shift_volume)[4]

    date = feature.iloc[len(feature) - 1]["date"]
    open = future_open
    high = future_high
    low = future_low
    close = future_close
    volume = future_volume

    sql = f"insert into future_ohlcv_1min_bitflyer_perp values('{date}',{open},{high},{low},{close},'{volume}')"
    repository.execute(database=database, sql=sql)

    basis.loc[basis["date"] == date, "open"] = open
    basis.loc[basis["date"] == date, "high"] = high
    basis.loc[basis["date"] == date, "low"] = low
    basis.loc[basis["date"] == date, "close"] = close
    basis.loc[basis["date"] == date, "volume"] = volume
