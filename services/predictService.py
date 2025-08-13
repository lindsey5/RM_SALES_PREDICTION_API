import calendar
from datetime import datetime
import pandas as pd
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')

def loadDataset():
    """Load and process the sales data."""
    file_path = "data/RM'S COLLECTION SALES.xlsx"  # cleaner string
    return pd.read_excel(file_path)

def create_features(df):
    """Generate date and lag-based features."""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'DATE' in df.columns:
            df = df.set_index(pd.to_datetime(df['DATE']))
        else:
            raise ValueError("DataFrame must have a 'DATE' column or DatetimeIndex.")

    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    for lag in [14, 30, 60]:
        df[f'lag_{lag}'] = df['AMOUNT'].shift(lag)

    for lag in [30]:
        df[f'sales_diff_{lag}'] = df['AMOUNT'] - df['AMOUNT'].shift(lag)

    for window in [30]:
        df[f'sales_rolling_mean_{window}'] = df['AMOUNT'].rolling(window).mean()

    return df

def forecast_next_days(model, historical_data, year, month):
    df_last_date = pd.to_datetime(historical_data.index[-1])
    _, last_day = calendar.monthrange(year, month)
    forecast_end_date = datetime(year, month, last_day)
    forecast_start_date = df_last_date + pd.Timedelta(days=1)

    num_days = (forecast_end_date - forecast_start_date).days + 1
    forecast_dates = pd.date_range(start=forecast_start_date, periods=num_days)

    forecast_df = pd.DataFrame(index=forecast_dates, columns=['AMOUNT'])
    combined_df = pd.concat([historical_data, forecast_df])

    predictions = []
    feature_cols = [
        'dayofweek', 'quarter', 'month', 'year',
        'dayofyear', 'dayofmonth', 'weekofyear',
        'lag_14', 'lag_30', 'lag_60',
        'sales_diff_30', 'sales_rolling_mean_30'
    ]

    for date in forecast_dates:
        current_with_features = create_features(combined_df.loc[:date].copy())
        current_features = current_with_features.loc[date:date]

        for col in feature_cols:
            if pd.isna(current_features[col].iloc[0]):
                if 'lag_' in col:
                    lag_days = int(col.split('_')[1])
                    if combined_df['AMOUNT'].notna().sum() >= lag_days:
                        current_features[col] = combined_df['AMOUNT'].iloc[-lag_days]
                    else:
                        current_features[col] = combined_df['AMOUNT'].mean()
                elif 'sales_diff_' in col:
                    current_features[col] = 0
                elif 'rolling_mean_' in col:
                    window = int(col.split('_')[-1])
                    recent_data = combined_df['AMOUNT'].dropna().tail(min(window, combined_df['AMOUNT'].notna().sum()))
                    current_features[col] = recent_data.mean()

        X_forecast = current_features[feature_cols]
        prediction = max(model.predict(X_forecast)[0], 0)

        predictions.append({"date": date, "prediction": prediction})
        combined_df.loc[date, 'AMOUNT'] = prediction

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df[
        (predictions_df['date'].dt.month == month) &
        (predictions_df['date'].dt.year == year)
    ]
    return predictions_df

def predict_future_sales():
    try:
        main_df = loadDataset()
        main_df['DATE'] = pd.to_datetime(main_df['DATE'])
        daily_sales = main_df.groupby('DATE')['AMOUNT'].sum().reset_index()
        daily_sales.set_index('DATE', inplace=True)
        daily_sales.sort_index(inplace=True)

        month = datetime.now().month
        year = datetime.now().year
        predictions = forecast_next_days(model, daily_sales, year, month)

        return {'forecast': predictions.to_dict(orient='records'), 'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}
