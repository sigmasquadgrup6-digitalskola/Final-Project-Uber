from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from datetime import time

with open('uber_price_pred_model.pkl', 'rb') as fr:
    model = pickle.load(fr)
with open('scaler_sc_X.pkl', 'rb') as fr:
    scaler = pickle.load(fr)


# Weekend checker
def is_weekend(dt):
    return 1 if dt.weekday() >= 5 else 0  # 5 = Sabtu, 6 = Minggu

#One Hot Time Category
def encode_hour_category(hour):
    return [
        1 if 9 <= hour < 16 else 0,
        1 if hour < 6 else 0,
        1 if 16 <= hour < 19 else 0,
        1 if 6 <= hour < 9 else 0,
        1 if 19 <= hour <= 23 else 0]



def run_uber_ML():
    passenger_count = st.number_input("Jumlah Penumpang", 1, 6, key='1')
    distance_km = st.number_input("Jarak perjalanan (KM)", min_value= 0.10, max_value = 999.00, 
                                  step = 0.10, value = 3.00, format='%.2f', key='2')
    tanggal = st.date_input('Tanggal Pick up', max_value = date.today(), min_value= date(2009,1,1), 
                            value = date(2015,10,15))
    waktu = st.time_input('Waktu Pick up', value = time(15,00))
    
    dt = datetime.datetime.combine(tanggal, waktu)
    hour = dt.hour

    weekend = is_weekend(dt)
    hour_ohe = encode_hour_category(hour)

# 12 Fitur sesuai training
    features = np.array([[
        passenger_count,
        distance_km,
        hour,
        dt.day,
        dt.month,
        dt.year,
        weekend,
        *hour_ohe
    ]])
    
    if st.button("Prediksi tarif"):
        try:
            features_scaled = scaler.transform(features)

            pred = model.predict(features_scaled)[0]
            st.success(f"Tarif diperkirakan: **${pred:.2f} USD**")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Pastikan urutan fitur sudah sesuai model.")
        
if __name__ =='__main__':
    run_uber_ML()

