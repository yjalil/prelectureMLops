from sklearn.preprocessing import MinMaxScaler
def passenger_scale(df):
    scaler = MinMaxScaler()
    df['passenger_count_scaled'] = scaler.fit_transform(df[['passenger_count']])
    return df
