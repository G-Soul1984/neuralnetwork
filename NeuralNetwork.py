import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
import random

np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)
def shuffle_in_unison(a, b):
    # перемешайте два массива одним и тем же способом
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

    def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                  test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
        """
    Загружает данные из источника Yahoo Finance, а также выполняет масштабирование, перетасовку, нормализацию и разделение.
 Параметры:
 бегущая строка (str/pd.DataFrame): бегущая строка, которую вы хотите загрузить, например, AAPL, TESL и т.д.
 n_steps (int): длина исторической последовательности (т.е. размер окна), используемая для прогнозирования, по умолчанию - 50
scale (bool): нужно ли масштабировать цены от 0 до 1, по умолчанию - True
shuffle (bool): нужно ли перетасовывать набор данных (как для обучения, так и для тестирования), по умолчанию - True
lookup_step (int): прогнозируемый шаг поиска в будущем, значение по умолчанию равно 1 (например, на следующий день).
 split_by_date (bool): независимо от того, разделяем ли мы набор данных на обучающие/тестовые по дате, установка
значения False приведет к случайному разделению наборов данных
test_size (float): соотношение для тестовых данных по умолчанию равно 0,2 (20% тестовых данных).
feature_columns (список): список функций, которые будут использоваться для подачи
        """
        # посмотрите, есть ли в тикере уже загруженные акции от yahoo finance
        if isinstance(ticker, str):
            # загрузите его из библиотеки yahoo_fin
            df = si.get_data(ticker)
        elif isinstance(ticker, pd.DataFrame):
            # уже загруженный, используйте его напрямую
            df = ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
        # это будет содержать все элементы, которые мы хотим вернуть из этой функции
        result = {}
        # мы также вернем сам исходный фрейм данных
        result['df'] = df.copy()
        # убедитесь, что переданные столбцы feature_columns существуют во фрейме данных
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."
        # добавить дату в виде столбца
        if "date" not in df.columns:
            df["date"] = df.index
        if scale:
            column_scaler = {}
            # масштабируйте данные (цены) от 0 до 1
            for column in feature_columns:
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler
            # добавьте экземпляры MinMaxScaler к возвращаемому результату
            result["column_scaler"] = column_scaler
        # добавьте целевой столбец (метку), переместив его на `lookup_step`
        df['future'] = df['adjclose'].shift(-lookup_step)
        # последние столбцы `lookup_step` содержат NaN в будущем столбце
        # возьми их, прежде чем бросать деньги
        last_sequence = np.array(df[feature_columns].tail(lookup_step))
        # drop NaNs
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=n_steps)
        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])
        # получите последнюю последовательность, добавив к последней последовательности `n_step` последовательность `lookup_step`
        # например, если n_steps=50 и lookup_step=10, длина last_sequence должна составлять 60 (то есть 50+10)
        # эта последняя последовательность будет использоваться для прогнозирования будущих цен на акции, которые недоступны в наборе данных
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        # добавить к результату
        result['last_sequence'] = last_sequence
        # постройте точки X и y
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)
        # преобразование в числовые массивы
        X = np.array(X)
        y = np.array(y)
        if split_by_date:
            # разделите набор данных на обучающие и тестовые наборы по дате (не произвольное разделение).
            train_samples = int((1 - test_size) * len(X))
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"] = X[train_samples:]
            result["y_test"] = y[train_samples:]
            if shuffle:
                # перетасуйте наборы данных для обучения (если задан параметр shuffle)
                shuffle_in_unison(result["X_train"], result["y_train"])
                shuffle_in_unison(result["X_test"], result["y_test"])
        else:
            # разбейте набор данных случайным образом
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                        test_size=test_size,
                                                                                                        shuffle=shuffle)
        # получите список данных набора тестов
        dates = result["X_test"][:, -1, -1]
        # извлекать тестовые объекты из исходного фрейма данных
        result["test_df"] = result["df"].loc[dates]
        # удалите дублирующиеся даты в тестовом фрейме данных
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # удалить даты из наборов данных для обучения/тестирования и преобразовать в float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
        return result
    # создание модели
    def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True),
                                            batch_input_shape=(None, sequence_length, n_features)))
                else:
                    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # добавляйте выпадение после каждого слоя
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model
    # обучение модели
    import os
    import time
    from tensorflow.keras.layers import LSTM

    # Window size or the sequence length
    N_STEPS = 50
    # Lookup step, 1 is the next day
    LOOKUP_STEP = 15
    # whether to scale feature columns & output price as well
    SCALE = True
    scale_str = f"sc-{int(SCALE)}"
    # whether to shuffle the dataset
    SHUFFLE = True
    shuffle_str = f"sh-{int(SHUFFLE)}"
    # whether to split the training/testing set by date
    SPLIT_BY_DATE = False
    split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
    # test ratio size, 0.2 is 20%
    TEST_SIZE = 0.2
    # features to use
    FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
    # date now
    date_now = time.strftime("%Y-%m-%d")
    ### model parameters
    N_LAYERS = 2
    # LSTM cell
    CELL = LSTM
    # 256 LSTM neurons
    UNITS = 256
    # 40% dropout
    DROPOUT = 0.4
    # whether to use bidirectional RNNs
    BIDIRECTIONAL = False
    ### training parameters
    # mean absolute error loss
    # LOSS = "mae"
    # huber loss
    LOSS = "huber_loss"
    OPTIMIZER = "adam"
    BATCH_SIZE = 64
    EPOCHS = 500
    # Amazon stock market
    ticker = "AMZN"
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
    {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"