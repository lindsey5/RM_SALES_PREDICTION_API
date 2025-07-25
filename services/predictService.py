import pandas as pd
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')

def loadDataset():
    # Load and process the sales data
    file_path = 'data/RM\'S COLLECTION SALES.xlsx'
    return pd.read_excel(file_path)

def create_features(df):
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week

    for lag in [14, 30, 60]:
        df[f'lag_{lag}'] = df['AMOUNT'].shift(lag)


    # Add difference features
    for lag in [30]:
        df[f'sales_diff_{lag}'] = df['AMOUNT'] - df['AMOUNT'].shift(lag)

    # Add rolling mean features
    for window in [30]:
        df[f'sales_rolling_mean_{window}'] = df['AMOUNT'].rolling(window).mean()

    return df

def forecast_next_days(model, historical_data, num_days=30):
    """
    Generate forecast for the next num_days
    """
    # Get the last date in historical data
    last_date = historical_data.index[-1]
    
    # Create date range for forecast
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                 periods=num_days, freq='D')
    
    # Initialize forecast dataframe
    forecast_df = pd.DataFrame(index=forecast_dates, columns=['AMOUNT'])
    
    # Combine historical and forecast dataframes for feature creation
    combined_df = pd.concat([historical_data, forecast_df])
    
    # Store predictions
    predictions = []
    
    for i, date in enumerate(forecast_dates):
        # Create features up to current forecast date
        current_combined = combined_df.loc[:date].copy()
        current_with_features = create_features(current_combined)
        
        # Get features for current date
        current_features = current_with_features.loc[date:date]
        
        # Check if we have all required features (no NaN values)
        feature_cols = [
            'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear',
            'lag_14', 'lag_30', 'lag_60',
            'sales_diff_30', 'sales_rolling_mean_30'
        ]
        
        # Handle missing lag features by using recent values
        for col in feature_cols:
            if pd.isna(current_features[col].iloc[0]):
                if 'lag_' in col:
                    lag_days = int(col.split('_')[1])
                    if len(combined_df.dropna()) >= lag_days:
                        # Use the lag value from available data
                        lag_value = combined_df.dropna()['AMOUNT'].iloc[-lag_days]
                        current_features[col] = lag_value
                    else:
                        # Use mean of available data
                        current_features[col] = combined_df.dropna()['AMOUNT'].mean()
                elif 'sales_diff_' in col:
                    # Use 0 for difference if lag not available
                    current_features[col] = 0
                elif 'rolling_mean_' in col:
                    # Use recent mean
                    window = int(col.split('_')[-1])
                    recent_data = combined_df.dropna()['AMOUNT'].tail(min(window, len(combined_df.dropna())))
                    current_features[col] = recent_data.mean()
        
        # Make prediction
        X_forecast = current_features[feature_cols]
        prediction = model.predict(X_forecast)[0]
        
        # Ensure prediction is positive
        prediction = max(prediction, 0)
        
        # Store prediction
        predictions.append(prediction)
        
        # Update the combined dataframe with the prediction
        combined_df.loc[date, 'AMOUNT'] = prediction

    return forecast_dates, predictions



def predict_future_sales():
    try:
        main_df = loadDataset()

        main_df['DATE'] = pd.to_datetime(main_df['DATE'])

        daily_sales = main_df.groupby('DATE')['AMOUNT'].sum().reset_index()
        daily_sales.set_index('DATE', inplace=True)
        daily_sales.sort_index(inplace=True)

        # Forecast future sales
        future_dates, predictions = forecast_next_days(model, daily_sales, num_days=30)

        return {
            'forecast':[float(pred) for pred in predictions],
            'dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'success': True,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }