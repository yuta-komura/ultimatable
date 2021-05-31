from pprint import pprint

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from lib import indicator, message, repository

tf.compat.v1.disable_eager_execution()


def calc_sma(df: pd.DataFrame, use_columns: str):
    df = indicator.add_sma(df=df, value=22, use_columns=use_columns)
    df = indicator.add_sma(df=df, value=34, use_columns=use_columns)
    df = indicator.add_sma(df=df, value=55, use_columns=use_columns)
    return df


def calc_ema(df: pd.DataFrame, use_columns: str):
    df = indicator.add_ema(df=df, value=22, use_columns=use_columns)
    df = indicator.add_ema(df=df, value=34, use_columns=use_columns)
    df = indicator.add_ema(df=df, value=55, use_columns=use_columns)
    return df


def calc_rsi(df: pd.DataFrame, use_columns: str):
    df = indicator.add_rsi(df=df, value=22, use_columns=use_columns)
    df = indicator.add_rsi(df=df, value=34, use_columns=use_columns)
    df = indicator.add_rsi(df=df, value=55, use_columns=use_columns)
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

    epochs = 500
    for e in range(epochs):
        net.run(opt, feed_dict={X: X_train, Y: y_train})

    pred_test = net.run(out, feed_dict={X: X_test})

    pred_test = np.concatenate((pred_test.T, X_test), axis=1)
    pred_inv = scaler.inverse_transform(pred_test)

    mae_test = mean_absolute_error(test_inv, pred_inv)
    message.info("mae", mae_test)

    return pred_inv[len(pred_inv) - 1]


database = "tradingbot"

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
        limit 5000
        """
basis = repository.read_sql(database=database, sql=sql)

analysis = basis.copy()
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

pprint(future_open)
pprint(future_high)
pprint(future_low)
pprint(future_close)

pprint(basis.tail(1))


# fig = plt.figure(figsize=(24, 12), dpi=50)
# ax1 = fig.add_subplot(1, 1, 1)
# line1, = ax1.plot(test_inv[:, 0])
# line2, = ax1.plot(pred_inv[:, 0])
# fig.savefig("backtest_result.png")
# plt.show()
