import streamlit as st 
import os 
import pandas as pd 


st.write("Social sentiment stock eports")

# Determine path to sentiment trending csv files

path = './Data'

# Get reports from the directionary 
files = os.listdir(path)

#Extract the reports 
files_list = [os.path.splittext(file)[0].lower() for file in files]
st.write(files_list)

# Set numerical or alphabetical filing system 
files_alpha = set([filename[0] for filename in files_list]) 

sel_alpha = st.selectbox("Latest sentiment reports", list(files_alpha)) 
filtered_files = [file for file in files_list if file[0]==sel_alpha]
sel_filtered_file = st.selectbox("Chose report", filtered_files)
if sel_filtered_file:
    st.write()
    df = pd.read_csv(f"{path}//{sel_filtered_file}.csv")
    st.DataFram(df)

# Command: streamlit run app.py

# LSTM MODEL

def scale_array(features, target, train_proportion:float = 0.8, scaler: bool = True):
    x = np.array(features)
    y = np.array(target).reshape(-1,1)
    split = int(0.8 * len(x))
    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]

    if scaler:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)
    else:
        pass
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, scaler

def create_LSTM_model(
    train_set: np.ndarray,
    dropout: float = 0.2,
    layer_one_dropout: float = 0.6,
    number_layers: int = 4,
    optimizer: str = 'adam',
    loss: str = 'mean_squared_error'):
    model = Squential()
    number_units = X_train.shape[1]
    dropout_fraction = dropout
    model.add(LSTM(
        units=number_units,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1))
        )
    model.add(Dropout(layer_one_dropout))
    layer_counter = 1
    while layer_counter < (number_layers - 1):
        
        model.add(LSTM(units=number_units, return_sequences = True))
        model.add(Dropout(dropout_fraction))
        layer_counter+=1

        
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))

    
    model.add(Dense(1))

    
    model.compile(optimizer=optimizer, loss=loss)
    
    return model

def calculate_strategy_returns(prices_df, trading_threshold, shorting: bool = False):
    '''
    prices_df: pd.DataFrame containing an 'Actual' and 'Predicted' column representing actual and model-predicted prices respectively
    
    '''
     # Calculate actual daily returns
    prices_df['actual_returns'] = prices_df['Actual'].pct_change()
    # Create a 'last close' column
    prices_df['last_close'] = prices_df['Actual'].shift()
    # Calculate the predicted daily returns, by taking the predicted price as a proportion of the last close
    prices_df['predicted_returns'] = (prices_df['Predicted'] - prices_df['last_close'])/prices_df['last_close']

    # Actual signal = 1 if actual returns more than threshold,  -1 if less than threshold
    prices_df['actual_signal'] = 0
    prices_df.loc[prices_df['actual_returns'] > trading_threshold , 'actual_signal'] = 1
    if shorting == True:
        prices_df.loc[prices_df['actual_returns'] < -trading_threshold , 'actual_signal'] = -1

    # Strategy signal = 1 if predicted returns > threshold, -1 if less than threshold
    prices_df['strategy_signal'] = 0
    prices_df.loc[prices_df['predicted_returns'] > trading_threshold , 'strategy_signal'] = 1
    if shorting == True:
        prices_df.loc[prices_df['predicted_returns'] < -trading_threshold , 'strategy_signal'] = -1       

    # Compute strategy returns
    prices_df['strategy_returns'] = prices_df['actual_returns'] * prices_df['strategy_signal']
    
    return prices_df

def calculate_RMSE(y_actual, y_predicted):
    MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE

# Set target cumulative returns as a threshold for model to achieve.
target_cumulative_return = 1.01

# Set returns threshold for strategy to fire trading signal
trading_threshold = 0.00

# Set maximum numberof iterations to run
max_iter = 3

tickers = technicals.index.get_level_values('symbol').unique().to_list()
# Initialise list to hold tickers that have successfully trained models that achieve the target cumulative returns:
modelled_tickers = []
trading_signals = []

for ticker in tickers:
    print("="*50)
    print(f"Initialising training for {ticker}")

    # Create signal dataframe as a copy
    signal = technicals.copy().loc[ticker].dropna()
    
    # Create blank row for current trading day and append to end of dataframe
    most_recent_timestamp = signal.index.get_level_values('timestamp').max() + timedelta(minutes = 1)
    signal.loc[most_recent_timestamp, ['target']] = np.nan

    # # Create target
    signal['target'] = signal['close'] 

    # Shift indicators to predict current trading day close
    signal.iloc[:, :-1]  = signal.iloc[:, :-1].shift()

    # Drop first row with NaNs resulting from data shift
    signal = signal.iloc[1:, :]

    # Ensure all data is 'float' type while also dropping null values due to value shifts and unavailable NaN indicator data.
    signal = signal.astype('float')

    # Set features and target
    X = signal.iloc[:, :-1]
    y = signal['target']
      
    # Use predefined scale_array function to transform data and perform train/test split
    X_train, X_test, y_train, y_test, scaler = scale_array(X, y, train_proportion = 0.8)

    # Record start time
    start_time = time.time()
    
    # (Re)set iter_counter and strategy_cumulative_return to 0 
    strategy_cumulative_return = 0
    iter_counter = 0

    # While loop that repeatedly trains LSTM models to adjust weights until it can hit the target cumulative return. Loop stops if max_iter is hit or if returns are achieved on backtesting
    while strategy_cumulative_return < target_cumulative_return and iter_counter != max_iter:
        
        strategy_cumulative_return = 0
        # Start iteration counter
        iter_counter+=1

        # Create model if first iteration. Reset model if subsequent iterations
        model = create_LSTM_model(X_train,
                                  dropout=0.4,
                                  layer_one_dropout=0.6,
                                  number_layers=6
                                 )

        # Set early stopping such that each iteration stops running epochs if validation loss is not improving (i.e. minimising further)
        callback = EarlyStopping(
            monitor='val_loss',
            patience=20, mode='auto',
            restore_best_weights=True
        )

        # Print message to allow visual confirmation of iteration training is currently at.
        print("="*50)
        print(f"Training {ticker} model iteration {iter_counter} ...please wait.\n")

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=1000, batch_size=32,
            shuffle=False,
            validation_split = 0.1,  
            verbose = 0,
            callbacks = callback
        )
        # Print confirmation that current iteration has ended.
        print(f"Iteration {iter_counter} ended.")

        # Evaluate loss when predicting test data. Sliced out entry -1 as y_test[-1] target is NaN 
        model_loss = model.evaluate(X_test[:-1], y_test[:-1], verbose=0)
    
        # Make predictions
        predicted = model.predict(X_test)

        # Recover the original prices instead of the scaled version
        predicted_prices = scaler.inverse_transform(predicted)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Create a DataFrame of Real and Predicted values
        prices = pd.DataFrame({
            "Actual": real_prices.ravel(),
            "Predicted": predicted_prices.ravel()
        }, index = signal.index[-len(real_prices): ]) 

        # Use predefined calculate_strategy_returns function to calculate and append strategy returns column to 'prices' dataframe
        prices = calculate_strategy_returns(prices, trading_threshold, shorting = False)
        
        
        # Compute strategy cumulative returns
        strategy_cumulative_return = (1+prices['strategy_returns']).cumprod()[-1]
        
        rmse = calculate_RMSE(prices['Actual'], prices['Predicted'])
        
        # Print performance metrics of the model given the feature weights produced by current iteration
        print(f"LSTM Method iteration {iter_counter} for {ticker} - Performance")
        print("-"*50)
        print(f"Model loss on testing dataset: \n{model_loss:.4f}")
        print(f"RMSE: \n{rmse:.4f}")
        print(f"Cumulative return on testing dataset: \n{strategy_cumulative_return:.4f}")
    
    # Append ticker to modelled_tickers:
    modelled_tickers.append(ticker)
    
    if strategy_cumulative_return >= target_cumulative_return:
        print(f"Target cumulative returns achieved\n")
        # Calculate cumulative returns at their best and worst time points over time.
        min_return = (1+prices['strategy_returns']).cumprod().min()
        max_return = (1+prices['strategy_returns']).cumprod().max()

        
        # Print cumulative return performance
        print(f"From {prices.index.min()} to {prices.index.max()}, the cumulative return of the current model is {strategy_cumulative_return:.2f}.")
        print(f"At its lowest, the model recorded a cumulative return of {min_return:.2f}.")
        print(f"At its highest, the model recorded a cumulative return of {max_return:.2f}.")  
        
        # Convert model to json
        model_json = model.to_json()

        # Save model layout as json
        path = f"../Resources/LSTM_model_weights/{date.today()}"
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path)
        
        file_path = Path(f"../Resources/LSTM_model_weights/{date.today()}/{ticker}.json")
        with open(file_path, "w") as json_file:
            json_file.write(model_json)

        # Save weights
        model.save_weights(f"../Resources/LSTM_model_weights/{date.today()}/{ticker}.h5")
        
        # Append the trading signal predicted by model
        trading_signals.append(prices.loc[prices.index.max(), 'strategy_signal'])

    else:
        st.code(f"The LSTM model was not able to achieve the target cumulative returns on the testing dataset within {max_iter} iterations.\n")
        trading_signals.append(0)


st.code("*"*50)
st.code(f"Training completed.")


