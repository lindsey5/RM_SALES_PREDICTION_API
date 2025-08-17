import calendar
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from dateutil.relativedelta import relativedelta
from sklearn.calibration import LabelEncoder

# Load model
model = joblib.load('models/random_forest_model.pkl')
model2 = joblib.load('models/rm-collection-product-qty.pkl')

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


def create_global_features(data):
    item_encoder = LabelEncoder()
    data['ITEM_ENCODED'] = item_encoder.fit_transform(data['PRODUCT NAME'])
    data = data.sort_values(['PRODUCT NAME', 'YEAR', 'MONTH'])
    data['TIME_IDX'] = (data['YEAR'] - data['YEAR'].min()) * 12 + data['MONTH']

    item_stats = data.groupby('PRODUCT NAME').agg({
        'QUANTITY': ['mean', 'std', 'max', 'min'],
    })
    item_stats.columns = [f'ITEM_{c[0]}_{c[1]}' for c in item_stats.columns]
    item_stats = item_stats.reset_index()
    data = data.merge(item_stats, on='PRODUCT NAME', how='left')

    print(item_stats.columns)

    item_mean_map = data.set_index('PRODUCT NAME')['ITEM_QUANTITY_mean'].to_dict()

    lagged_list = []
    for item, item_df in data.groupby('PRODUCT NAME'):
        item_df = item_df.copy().sort_values(['YEAR', 'MONTH'])
        item_df['QTY_LAG1'] = item_df['QUANTITY'].shift(1)
        item_df['QTY_LAG2'] = item_df['QUANTITY'].shift(2)
        item_df['QTY_LAG3'] = item_df['QUANTITY'].shift(3)
        item_df['QTY_ROLLING_3'] = item_df['QUANTITY'].rolling(3, min_periods=1).mean()
        item_df['QTY_ROLLING_6'] = item_df['QUANTITY'].rolling(6, min_periods=1).mean()
        item_df['QTY_RELATIVE'] = item_df['QUANTITY'] / item_mean_map[item]
        lagged_list.append(item_df)

    data_with_lags = pd.concat(lagged_list, ignore_index=True)
    data_with_lags['MONTH_SIN'] = np.sin(2 * np.pi * data_with_lags['MONTH'] / 12)
    data_with_lags['MONTH_COS'] = np.cos(2 * np.pi * data_with_lags['MONTH'] / 12)
    data_with_lags['QUARTER_SIN'] = np.sin(2 * np.pi * data_with_lags['QUARTER'] / 4)
    data_with_lags['QUARTER_COS'] = np.cos(2 * np.pi * data_with_lags['QUARTER'] / 4)

    monthly_totals = data_with_lags.groupby(['YEAR', 'MONTH']).agg({
        'QUANTITY': 'sum'
    }).rename(columns={'QUANTITY': 'MARKET_QTY'}).reset_index()

    data_with_lags = data_with_lags.merge(monthly_totals, on=['YEAR', 'MONTH'], how='left')
    data_with_lags = data_with_lags.fillna(0)

    print(data_with_lags)

    return data_with_lags

def forecast_items_qty_sold(target_month=None, target_year=None):
    try:
        # Load and prepare data
        df = loadDataset()
        
        df['MONTH'] = df['DATE'].dt.month
        df['YEAR'] = df['DATE'].dt.year

        # Aggregate monthly data
        monthly_data = df.groupby(['PRODUCT NAME', 'YEAR', 'MONTH']).agg({
            'QUANTITY': 'sum',
            'AMOUNT': 'mean',
        }).reset_index()

        monthly_data['QUARTER'] = monthly_data['MONTH'].apply(lambda x: ((x - 1) // 3) + 1)
        
        # Create features (assuming this function exists)
        data = create_global_features(monthly_data)

        # Get last available date from data
        last_year = int(data['YEAR'].max())
        last_month = int(data[data['YEAR'] == last_year]['MONTH'].max())
        
        target_month = int(target_month)
        target_year = int(target_year)
        

        last_date = datetime(last_year, last_month, 1)
        target_date = datetime(target_year, target_month, 1)

        # Define feature columns
        feature_columns = [
            'ITEM_ENCODED','TIME_IDX','MONTH','QUARTER',
            'ITEM_QUANTITY_mean',	'ITEM_QUANTITY_std', 'ITEM_QUANTITY_max','ITEM_QUANTITY_min',
            'QTY_LAG1','QTY_LAG2','QTY_LAG3',
            'QTY_ROLLING_3','QTY_ROLLING_6','QTY_RELATIVE',
            'MONTH_SIN','MONTH_COS','QUARTER_SIN','QUARTER_COS',
        ]


        predictions_all_months = []
        current_date = last_date + relativedelta(months=1)

        # Make a copy of data to avoid modifying original
        forecast_data = data.copy()

        # Loop through months until target
        while current_date <= target_date:
            forecast_month = current_date.month
            forecast_year = current_date.year
            current_quarter = ((forecast_month - 1) // 3) + 1

            month_predictions = []

            # Group by item and make predictions
            for item, item_df in forecast_data.groupby('PRODUCT NAME'):
                item_df = item_df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)
                
                # Skip items with insufficient data
                if len(item_df) < 3: 
                    continue

                try:
                    # Get the most recent row for this item
                    last_row = item_df.iloc[-1].copy()
                    
                    # Create feature vector
                    feat = pd.DataFrame([last_row[feature_columns]])
                    
                    # Update time-based features for forecast month
                    feat['MONTH'] = forecast_month
                    feat['MONTH_SIN'] = np.sin(2 * np.pi * forecast_month / 12)
                    feat['MONTH_COS'] = np.cos(2 * np.pi * forecast_month / 12)
                    feat['QUARTER'] = current_quarter
                    feat['QUARTER_SIN'] = np.sin(2 * np.pi * current_quarter / 4)
                    feat['QUARTER_COS'] = np.cos(2 * np.pi * current_quarter / 4)

                    # Update lag features
                    feat['QTY_LAG1'] = item_df['QUANTITY'].iloc[-1]
                    feat['QTY_LAG2'] = item_df['QUANTITY'].iloc[-2] if len(item_df) >= 2 else item_df['QUANTITY'].iloc[-1]
                    feat['QTY_LAG3'] = item_df['QUANTITY'].iloc[-3] if len(item_df) >= 3 else feat['QTY_LAG2'].iloc[0]
                    
                    # Update rolling averages
                    feat['QTY_ROLLING_3'] = item_df['QUANTITY'].tail(min(3, len(item_df))).mean()
                    feat['QTY_ROLLING_6'] = item_df['QUANTITY'].tail(min(6, len(item_df))).mean()
                
                    # Make prediction
                    pred_qty = model2.predict(feat)[0]

                    month_predictions.append({
                        'item': str(item),
                        'predicted_month': f"{forecast_month:02d}",
                        'predicted_year': forecast_year,
                        'predicted_qty': round(pred_qty, 0),
                        'predicted_sales': round(float(item_df['AMOUNT'].iloc[-1]) * pred_qty, 2),
                    })

                    # Add prediction back to data for next iteration
                    new_row = last_row.copy()
                    new_row['YEAR'] = forecast_year
                    new_row['MONTH'] = forecast_month
                    new_row['QTY'] = pred_qty
                    new_row['QUARTER'] = current_quarter
                    
                    # Update TIME_IDX if it exists
                    if 'TIME_IDX' in new_row:
                        new_row['TIME_IDX'] = forecast_data['TIME_IDX'].max() + 1
                    
                    forecast_data = pd.concat([forecast_data, pd.DataFrame([new_row])], ignore_index=True)

                except Exception as e:
                    # Log individual item prediction errors but continue
                    print(f"Warning: Could not predict for item '{item}': {str(e)}")
                    continue

            predictions_all_months.extend(month_predictions)
            current_date += relativedelta(months=1)

        # Filter to target month only
        target_predictions = [
            p for p in predictions_all_months
            if int(p['predicted_month']) == target_month and int(p['predicted_year']) == target_year
        ]

        return {
            'forecast': target_predictions,
            'success': True,
            'month': f"{datetime(target_year, target_month, 1):%B %Y}",
            'total_items_predicted': len(target_predictions)
        }

    except Exception as e:
        return {
            'forecast': [],
            'success': False,
            'error': f'Forecasting failed: {str(e)}',
            'month': None
        }