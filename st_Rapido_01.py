#Import Libraries
import pandas as pd
import mysql.connector
import pprint
import sqlalchemy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVR
import pickle, pathlib, datetime
from datetime import date, timedelta
import streamlit as st


#Streamlit Structure


st.markdown(
    """
    <div style="background-color:#7E33FF;padding:15px;border-radius:5px">
        <h2 style="color:white;text-align:center;">
            Rapido: Intelligent Mobility Insights
        </h2>
        <p style="color:white;text-align:center;">
            Ride Patterns, Cancellations & Fare Forecasting
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

#Load Data Sets
bookings_raw        = pickle.load(open(f"cache/bookings_raw_20260125.pkl", "rb"))
customers_raw        = pickle.load(open(f"cache/customers_raw_20260123.pkl",   "rb"))
drivers_raw          = pickle.load(open(f"cache/drivers_raw_20260125.pkl",     "rb"))
location_demand_raw  = pickle.load(open(f"cache/location_demand_raw_20260123.pkl",     "rb"))
time_raw             = pickle.load(open(f"cache/time_raw_20260123.pkl",     "rb"))
cancel_count         = pickle.load(open(f"cache/cancel_count_20260125.pkl",     "rb"))

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Plots", "Model Evaluation", "Predictions"])

if page == "Home":
    #st.title("Rapido: Intelligent Mobility Insights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
         st.metric("Completed Rides", 68346)
    with col2:
         st.metric("Cancelled Rides", 23284)
    with col3:
         st.metric("Incomplete Rides", 8370)
    with col4:
         st.metric("Avg Fare per KM", "₹25.03")

    # Vehicle type counts
    vehicle_counts = {"Auto": 1697, "Cab": 1683, "Bike": 1620}

    st.markdown(
    "<h8 style='text-align: left; color: #FFF;'>Vehicle Type Metrics</h8>",
            unsafe_allow_html=True)

    # Display metrics side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Auto Rides", vehicle_counts["Auto"])

    with col2:
        st.metric("Cab Rides", vehicle_counts["Cab"])

    with col3:
        st.metric("Bike Rides", vehicle_counts["Bike"])

    st.write("Data Insights summary - Numerical")
    st.dataframe(bookings_raw.describe())
    st.write("Data Insights summary - Object")
    st.dataframe(bookings_raw.describe(include="object"))
    

elif page == "Plots":
    st.title("Visualizations")
    tab1, tab2, tab3 = st.tabs(["General Plot Insights", "Long Distance Insights","Reliability & Loyalty"])
    #st.write("EDA and model plots go here")
    with tab1:
    # Ride volume by hour
        @st.cache_data
        def get_ride_volume_plot(bookings_raw):
        # Aggregate once
            Ride_vol_by_hour=(bookings_raw.groupby("hour_of_day").size().reset_index(name="booking_count"))
            fig1 = px.bar(
                        Ride_vol_by_hour, 
                        x="hour_of_day", 
                        y="booking_count",
                        title="<b>Ride Volume by Hour</b>",
                        labels={"hour_of_day": "Hour of Day", "booking_count": "Number of Bookings"},
                        template="plotly_white",  # Clean, professional look
                        color="booking_count",    # adds a color gradient based on volume
                        color_continuous_scale="Viridis"
                    )
        #Display
            fig1.update_layout(
                xaxis_tickmode='linear', # Shows every hour index (1, 2, 3...)
                ) 
            return fig1
        fig1 = get_ride_volume_plot(bookings_raw)
        st.plotly_chart(fig1, use_container_width=True)

    #Ride volume by city
        @st.cache_data
        def get_ride_city_plot(bookings_raw):
            ride_vol_by_city=(bookings_raw.groupby("city").size().reset_index(name="ride_volume"))
            fig2 = px.bar(
                        ride_vol_by_city,
                        x="city",
                        y="ride_volume",
                        title="<b>Ride Volume by City</b>",
                        labels={"city":"City","ride_volume":"Ride Volume"},
                        color="ride_volume",
                        )
            return fig2
        fig2=get_ride_city_plot(bookings_raw)
        #Display
        st.plotly_chart(fig2,use_container_width=True)

    #Ride Volume by Day of the week
        ride_vol_day=(bookings_raw.groupby("day_of_week").size().reset_index(name="ride_volume"))
        fig3=px.bar(
                    ride_vol_day,
                    x="day_of_week",
                    y="ride_volume",
                    title="<b>Ride Volume by Day of week</b>",
                    labels={"day_of_week":"Day","ride_volume":"Volume"},
                    color="ride_volume",
                    color_continuous_scale="Viridis"
                )
        #Display
        st.plotly_chart(fig3,use_container_width=True)

    #Heat Map
        cancel_data = bookings_raw[bookings_raw['booking_status'] == 'Cancelled']
        pivot = cancel_data.pivot_table(index='city', columns='hour_of_day', values='booking_status', aggfunc='count').fillna(0)
        fig4=px.imshow(
                        pivot,
                        labels=dict(x="Hour of Day", y="City", color="Cancellation"),
                        x=pivot.columns,
                        y=pivot.index,
                        color_continuous_scale="Reds",
                        aspect="auto",
                        text_auto=True
                        )

        fig4.update_layout(
                            title="<b>Cancellation Heatmap (City vs Hour)</b>",
                            xaxis_nticks=24
                            )
    #Display
        st.plotly_chart(fig4, use_container_width=True)

    #Fare vs Distance
        fig5 = px.scatter(
                        bookings_raw, 
                        x='ride_distance_km', 
                        y='booking_value',
                        title="<b>Fare vs Distance</b>",
                        labels={'ride_distance_km': 'Ride Distance (km)', 'booking_value': 'Fare (Value)'},
                        opacity=0.5,           # Similar to alpha=0.8
                        template="plotly_white",
                        hover_data=['city','vehicle_type']    #see the city and Vehicle Type when hovering over a dot
                        )
        #Display
        st.plotly_chart(fig5, use_container_width=True)

    #Cancellations Insights
        fig6= px.bar(cancel_count,
            x="city",
            y="count",
            hover_name="weather_condition",
            color="traffic_level",
            title="<b>Cancellations Insights</b>"
                    )
        #Display
        st.plotly_chart(fig6, use_container_width=True)

    #Drivers Ratings by City
        df=drivers_raw.reset_index()
        fig7 = px.bar(
            df,
            x="driver_city",
            y="avg_driver_rating",
            hover_data=["avg_driver_rating","driver_experience_years","driver_id"],
            color="vehicle_type",
            title="Driver Ratings by City",
            category_orders={"vehicle_type": ["Auto", "Cab", "Bike"]},
            barmode="group"
        )

        fig7.update_xaxes(title="Driver City")
        fig7.update_yaxes(title="Avg Driver Rating")
        st.plotly_chart(fig7,use_container_width=True)
    #Rush Hours
        fig8= px.line(time_raw, x="day_of_week",y="hour_of_day",hover_data="peak_time_flag",title="Rush Hours",
                    labels={"day_of_week":"Day","hour_of_day":"Hours"})

    #Display
        st.plotly_chart(fig8,use_container_width=True)
    
    with tab2:
        st.header("Analysis & Plots")
        #@st.cache_data
        #def load_data():
        df= bookings_raw.copy()
            #return df

        # Add metrics at the top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trips", len(df))
        with col2:
            long_dist_count = df['long_distance_flag'].sum()
            st.metric("Long Distance Trips", long_dist_count)
        with col3:
            long_dist_pct = (long_dist_count / len(df)) * 100
            st.metric("Long Distance %", f"{long_dist_pct:.1f}%")
        with col4:
            avg_distance = df[df['ride_distance_km'] > 15]["ride_distance_km"].mean()
            st.metric("Avg Distance (km)", f"{avg_distance:.2f}")

        # Plot 1: Box Plot - Ride Distance Distribution by Long Distance Flag
        st.subheader("📊 Ride Distance Distribution")
        fig9 = px.box(df, 
                        x='long_distance_flag', 
                        y='ride_distance_km',
                        color='long_distance_flag',
                        labels={'long_distance_flag': 'Long Distance Flag', 
                                'ride_distance_km': 'Ride Distance (km)'},
                        color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'},
                        title='Distribution of Ride Distances by Long Distance Flag')

        fig1.update_xaxes(tickmode='array', tickvals=[0, 1], ticktext=['Short Distance', 'Long Distance'])
        st.plotly_chart(fig9, use_container_width=True)

    with tab3:
         #fig10=st.header("Driver Reliability")
         drivers_RL=drivers_raw
         drivers_RL['driver_reliability_score'] = (drivers_RL['acceptance_rate'] * (1 - drivers_RL['delay_rate']))
         drivers_RL["cust_loyalty_score"]=(drivers_RL["accepted_rides"] - drivers_RL["incomplete_rides"]) / drivers_RL["total_assigned_rides"]
         fig10=px.bar(
                drivers_RL,
                x="driver_city",
                y="acceptance_rate",
                color="driver_reliability_score",
                barmode="group",
                hover_data=["vehicle_type","incomplete_rides"],
                title="Driver Reliability Analysis"
                )
         st.plotly_chart(fig10, use_container_width=True)

         #st.header("Customer Loyalty")
         #Customer Loyalty Score - ratio of completed rides to total assigned rides
         fig11= px.bar(
                        drivers_RL,
                        x="driver_city",
                        y="cust_loyalty_score",
                        color="vehicle_type",
                        barmode="group",
                        hover_data=["driver_experience_years","avg_driver_rating","driver_reliability_score"],
                        labels={"driver_city":"City","cust_loyalty_score":"Loyalty Score"},
                        title="Customer Loyalty Analysis"
                        )
         st.plotly_chart(fig11, use_container_width=True)
         


elif page == "Model Evaluation":
    st.title("Model Evaluation Results")
    tab1, tab2, tab3, tab4 = st.tabs(["Ride Outcome Prediction\n(Multi-Class Classification)", "Customer Cancellation Risk Model (Binary Classification","Driver Delay Prediction Model (Binary Classification)", "Fare Prediction Model (Regression)"])
            #st.write("Classification and regression metrics tables go here")
    with tab1:
         with open(f"cache/LogisticRegression_df_20260127.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/RandomForest_df_20260127.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/GradientBoosting_df_20260127.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/DecisionTree_df_20260127.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/SVC_df_20260127.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         LogisticRegression_df_20260127=pickle.load(open(f"cache/LogisticRegression_df_20260127.pkl", "rb"))
         RandomForest_df_20260127=pickle.load(open(f"cache/RandomForest_df_20260127.pkl", "rb"))
         GradientBoosting_df=pickle.load(open(f"cache/GradientBoosting_df_20260127.pkl", "rb"))
         DecisionTree_df=pickle.load(open(f"cache/DecisionTree_df_20260127.pkl", "rb"))
         SVC_df=pickle.load(open(f"cache/SVC_df_20260127.pkl", "rb"))

         st.write("1. LogisticRegression Model")
         st.table(LogisticRegression_df_20260127)

         st.write("2. RandomForestClassifier Model")
         st.table(RandomForest_df_20260127)

         st.write("3. GradientBoostingClassifier Model")
         st.table(GradientBoosting_df)

         st.write("4. DecisionTree Model")
         st.table(DecisionTree_df)

         st.write("5. Linear SVC Model")
         st.table(SVC_df)

    with tab2:
         with open(f"cache/cust_cancel_dfs/Cust_LogisticRegression_20260128.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/cust_cancel_dfs/Cust_RandomForest_20260128.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/cust_cancel_dfs/Cust_GradientBoosting_20260128.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/cust_cancel_dfs/Cust_DT_20260128.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         Cust_LogisticRegression=pickle.load(open(f"cache/cust_cancel_dfs/Cust_LogisticRegression_20260128.pkl", "rb"))
         Cust_RandomForest=pickle.load(open(f"cache/cust_cancel_dfs/Cust_RandomForest_20260128.pkl", "rb"))
         Cust_GradientBoosting=pickle.load(open(f"cache/cust_cancel_dfs/Cust_GradientBoosting_20260128.pkl", "rb"))
         DecisionTree_df=pickle.load(open(f"cache/cust_cancel_dfs/Cust_DT_20260128.pkl", "rb"))
         

         st.write("1. LogisticRegression Model")
         st.table(Cust_LogisticRegression)

         st.write("2. RandomForestClassifier Model")
         st.table(Cust_RandomForest)

         st.write("3. GradientBoostingClassifier Model")
         st.table(Cust_GradientBoosting)

         st.write("4. DecisionTree Model")
         st.table(DecisionTree_df)

    with tab3:
         with open(f"cache/Driver_delay_DFs/Driver_LogisticRegression_df.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/Driver_delay_DFs/Driver_RF_df.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/Driver_delay_DFs/Driver_GB_df.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/Driver_delay_DFs/Driver_DT_df.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         Driver_LogisticRegression=pickle.load(open(f"cache/Driver_delay_DFs/Driver_LogisticRegression_df.pkl", "rb"))
         Driver_RF=pickle.load(open(f"cache/Driver_delay_DFs/Driver_RF_df.pkl", "rb"))
         Driver_GB=pickle.load(open(f"cache/Driver_delay_DFs/Driver_GB_df.pkl", "rb"))
         Driver_DT=pickle.load(open(f"cache/Driver_delay_DFs/Driver_DT_df.pkl", "rb"))
         

         st.write("1. LogisticRegression Model")
         st.table(Driver_LogisticRegression)

         st.write("2. RandomForestClassifier Model")
         st.table(Driver_RF)

         st.write("3. GradientBoostingClassifier Model")
         st.table(Driver_GB)

         st.write("4. DecisionTree Model")
         st.table(Driver_DT)

    with tab4:
         with open(f"cache/Fare_pred_results.pkl", "rb") as file:
                        loaded_data = pickle.load(file)
         df_results=pickle.load(open(f"cache/Fare_pred_results.pkl", "rb"))
      
         st.write("Model Performance Comparison")
         st.table(df_results)

         st.write("### Prediction Accuracy")

         with open(f"cache/Fare_pred_model/Fare_pred_LogisticRegression_20260129.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/Fare_pred_model/Fare_pred_LinearSVR_20260129.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/Fare_pred_model/Fare_pred_RandomForest_20260129.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/Fare_pred_model/Fare_x_test_20260129.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         with open(f"cache/Fare_pred_model/Fare_y_test_20260129.pkl", "rb") as file:
                        loaded_data = pickle.load(file)

         model14=pickle.load(open(f"cache/Fare_pred_model/Fare_pred_LogisticRegression_20260129.pkl", "rb"))
         model15=pickle.load(open(f"cache/Fare_pred_model/Fare_pred_LinearSVR_20260129.pkl", "rb"))
         model16=pickle.load(open(f"cache/Fare_pred_model/Fare_pred_RandomForest_20260129.pkl", "rb"))
         x_test=pickle.load(open(f"cache/Fare_pred_model/Fare_x_test_20260129.pkl", "rb"))
         y_test=pickle.load(open(f"cache/Fare_pred_model/Fare_y_test_20260129.pkl", "rb"))

         preds = {
                    "LinearRegression": model14.predict(x_test),
                    "Liner SVR": model15.predict(x_test),
                    "RandomForest": model16.predict(x_test)
                }
        # Combine predictions into one long-form dataframe for Plotly
         plot_list = []
         for name, pred in preds.items():
            temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': pred, 'Model': name})
            plot_list.append(temp_df)
         full_plot_df = pd.concat(plot_list)

         fig_scatter = px.scatter(full_plot_df, x='Actual', y='Predicted', color='Model',
                    marginal_x="histogram", opacity=0.5, title="Combined Model Comparison")
         fig_scatter.add_shape(type="line", line=dict(dash='dash'), 
                            x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
         st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Predictions":
     st.title("Ride Outcome prediction")
     option=st.radio("Choose Model",("Booking Status Prediction", "Fare Prediction","Customer Cancellation Risk","Driver Delay Prediction"))
     with open("cache/GB_model_BKstatus.pkl", "rb") as file:
        model3 = pickle.load(file)
        #model2=models_bundle["B_Status_RandomForest"]
     with open ("cache/Fare_pred_model/Fare_pred_RandomForest_20260129.pkl","rb") as f:
        model16=pickle.load(f)
     with open ("cache/merged_df.pkl","rb") as f:
        merged_df=pickle.load(f)
     with open ("cache/cust_cancel_dfs/Cust_RandomForest_20260128.pkl","rb") as f:
        model7=pickle.load(f)
     with open ("cache/Driver_delay_DFs/Driver_RF_model.pkl","rb") as f:
        model11=pickle.load(f)

            # ---Common UI Inputs ---
     day_of_week = st.selectbox(
        "Day Of Week",
         ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],key="day_of_week_booking")
     is_weekend = st.selectbox("Is Weekend",["Yes","no"])
     hour_of_day = st.number_input("Hours", min_value=0.0, step=0.1, format="%.2f")
     if option != "Driver Delay Prediction":
      vehicle_type = st.selectbox(
                                "Vehicle Type",
                            ["Bike","Auto","Cab"])

     #if option != "Driver Delay Prediction":
     ride_distance_km = st.number_input("Distance", min_value=0.0, step=0.1, format="%.2f")
     if option != "Driver Delay Prediction":
      estimated_ride_time_min = st.number_input("Est Time", min_value=0.0, step=0.1, format="%.2f")
     if option != "Driver Delay Prediction":
      actual_ride_time_min = estimated_ride_time_min     
     traffic_level=st.radio("Traffic Level", ["High", "Low", "Medium"])
     weather_condition= st.radio("Weather", ["Clear", "Heavy Rain", "Rain"])
     if option != "Driver Delay Prediction":
      base_fare = st.number_input("Base Fare", min_value=0.0, step=0.1, format="%.2f")
     if option != "Driver Delay Prediction":
      surge_multiplier = st.number_input("surge multiplier", min_value=0.0, step=0.1, format="%.2f")
                 #Assign Numeric Values
     if day_of_week=="Monday":
            day_of_week=1
     elif day_of_week=="Tuesday":
            day_of_week=5
     elif day_of_week=="Wednesday":
            day_of_week=6
     elif day_of_week=="Thursday":
            day_of_week=4
     elif day_of_week=="Friday":
            day_of_week=0
     elif day_of_week=="Saturday":
            day_of_week=2
     elif day_of_week=="Sunday":
            day_of_week=3

     if is_weekend=="Yes":
            is_weekend=1
     else:
            is_weekend=0
                
     if option != "Driver Delay Prediction":
      if vehicle_type=="Auto":
            vehicle_type=0
      elif vehicle_type=="Bike":
            vehicle_type=1
      else: vehicle_type=2
                
     if traffic_level=="High":
            traffic_level=0
     elif traffic_level=="Low":
            traffic_level=1
     else: traffic_level=2        
            
     if weather_condition=="Clear":
            weather_condition=0
     elif weather_condition=="Heavy Rain":
            weather_condition=1
     else: weather_condition=2

     if traffic_level < 1 & weather_condition > 0:
           bad_conditions = 1
     else: bad_conditions=0

     
     if option == "Booking Status Prediction":
            # --- UI Inputs ---
            
            pickup_location=st.selectbox(
                                "Pickup loc",
                            ["Loc_1", "Loc_2", "Loc_3", "Loc_4", "Loc_5", "Loc_6", "Loc_7", "Loc_8", "Loc_9", 
                             "Loc_10", "Loc_11", "Loc_12", "Loc_13", "Loc_14", "Loc_15", "Loc_16", "Loc_17", 
                             "Loc_18", "Loc_19", "Loc_20", "Loc_21", "Loc_22", "Loc_23", "Loc_24", "Loc_25", 
                             "Loc_26", "Loc_27", "Loc_28", "Loc_29", "Loc_30", "Loc_31", "Loc_32", "Loc_33", 
                             "Loc_34", "Loc_35", "Loc_36", "Loc_37", "Loc_38", "Loc_39", "Loc_40", "Loc_41", 
                             "Loc_42", "Loc_43", "Loc_44", "Loc_45", "Loc_46", "Loc_47", "Loc_48", "Loc_49", "Loc_50"])
            drop_location=st.selectbox(
                                "Drop loc",
                            ["Loc_1", "Loc_2", "Loc_3", "Loc_4", "Loc_5", "Loc_6", "Loc_7", "Loc_8", "Loc_9", 
                            "Loc_10", "Loc_11", "Loc_12", "Loc_13", "Loc_14", "Loc_15", "Loc_16", "Loc_17", 
                            "Loc_18", "Loc_19", "Loc_20", "Loc_21", "Loc_22", "Loc_23", "Loc_24", "Loc_25", 
                            "Loc_26", "Loc_27", "Loc_28", "Loc_29", "Loc_30", "Loc_31", "Loc_32", "Loc_33", 
                            "Loc_34", "Loc_35", "Loc_36", "Loc_37", "Loc_38", "Loc_39", "Loc_40", "Loc_41", 
                            "Loc_42", "Loc_43", "Loc_44", "Loc_45", "Loc_46", "Loc_47", "Loc_48", "Loc_49", "Loc_50"])
            booking_value = st.number_input("Booking Value", min_value=0.0, step=0.1, format="%.2f")
             
            # --- Save Button ---
            if st.button("Submit"):
                st.cache_data.clear()
                st.cache_resource.clear()

                pickup_location = int(pickup_location.split('_')[1]) - 1
                drop_location= int(drop_location.split('_')[1]) - 1
                if ride_distance_km>15:
                    long_distance_flag = 1
                else: 
                    long_distance_flag = 0
                fare_per_KM=booking_value/ ride_distance_km+0.001
                
                fare_per_Min = booking_value / actual_ride_time_min+0.001
                if hour_of_day in [8,9,10,17,18,19,20]:
                    is_rush_hour = 1
                else: is_rush_hour = 0

                #time_diff = actual_ride_time_min - estimated_ride_time_min
                distance_time_ratio = ride_distance_km / actual_ride_time_min + 0.001
                surge_impact = base_fare * surge_multiplier
                traffic_weather_product=traffic_level * weather_condition
                traffic_surge_product=traffic_level * surge_multiplier

                # Store data in an array (list)
                BKStatus_data = [[
                                day_of_week,               
                                is_weekend,                 
                                hour_of_day,               
                                pickup_location,           
                                drop_location,           
                                vehicle_type,             
                                ride_distance_km,         
                                estimated_ride_time_min,  
                                traffic_level,              
                                weather_condition,          
                                base_fare,       
                                surge_multiplier,
                                booking_value,        
                                long_distance_flag,
                                fare_per_KM,      
                                fare_per_Min,             
                                is_rush_hour,               
                                distance_time_ratio,      
                                surge_impact,            
                                traffic_weather_product,    
                                traffic_surge_product 
                ]]
                
                result1=model3.predict(BKStatus_data)

                if result1==0:
                    st.write("Cancelled")
                else : 
                    st.write("Completed")

     if option == "Fare Prediction":       
            booking_status=st.selectbox("Booking Status", ["Completed","Cancelled","Incomplete"])
            cancellation_rate = st.number_input("cancellation_rate(Enter value between 0-1. Value > 0.3 is High)", min_value=0.0, step=0.1, format="%.2f") 
            # --- Save Button ---
            if st.button("Submit"):
                st.cache_data.clear()
                st.cache_resource.clear()
                
                

                if booking_status=="Completed":
                    booking_status=1
                elif booking_status=="Incomplete":
                    booking_status=2
                else: booking_status=0

                if ride_distance_km>15:
                    long_distance_flag = 1
                else: 
                    long_distance_flag = 0

                fare_per_KM=((base_fare*surge_multiplier)/ ride_distance_km)+0.001
                fare_per_Min = ((base_fare*surge_multiplier) / actual_ride_time_min)+0.001

                if hour_of_day in [8,9,10,17,18,19,20]:
                    is_rush_hour = 1
                else: is_rush_hour = 0

                time_diff = actual_ride_time_min - estimated_ride_time_min
                distance_time_ratio = ride_distance_km / actual_ride_time_min + 0.001
                surge_impact = base_fare * surge_multiplier
                traffic_weather_product=traffic_level * weather_condition
                if cancellation_rate > 0.3:
                    high_cancellation_history=1
                else: high_cancellation_history=0

                if surge_multiplier > 1.5:
                    high_surge=1
                else: high_surge=0

                threshold = merged_df['fare_per_KM'].quantile(0.75)

                if fare_per_KM > threshold:
                    high_price_per_km=1
                else: high_price_per_km=0
                #high_price_per_km = (fare_per_KM > fare_per_KM.quantile(0.75)).astype(int)

                # Store data in an array (list)
                Fare_data = [[
                    day_of_week,
                    is_weekend,
                    hour_of_day,
                    vehicle_type,
                    ride_distance_km,
                    estimated_ride_time_min,
                    traffic_level,
                    weather_condition,
                    base_fare,
                    surge_multiplier,
                    booking_status,
                    long_distance_flag,
                    fare_per_KM,
                    fare_per_Min,
                    is_rush_hour,
                    time_diff,
                    distance_time_ratio,
                    surge_impact,
                    bad_conditions,
                    traffic_weather_product,
                    high_cancellation_history,
                    high_surge,
                    high_price_per_km
                ]]
                
                result2=model16.predict(Fare_data)

                st.write("Predicted Fare is:",result2)
     
     if option == "Customer Cancellation Risk":
         st.write("Work in Progress")

     if option == "Driver Delay Prediction":
        #st.write("Work in progress")
            booking_value = st.number_input("Booking Value", min_value=0.0, step=0.1, format="%.2f")
            booking_status=st.selectbox("Booking Status", ["Completed","Cancelled","Incomplete"])
            driver_id = st.number_input("Driver ID", min_value=0, step=1)
            total_assigned_rides = st.number_input("Total Assigned Rides", min_value=0, step=1)
            accepted_rides = st.number_input("Accepted rides", min_value=0, step=1)
            incomplete_rides = st.number_input("incomplete rides", min_value=0, step=1)
            delay_count = st.number_input("Delay Count", min_value=0, step=1)
            acceptance_rate = st.number_input("Acceptance rate", min_value=0.0, step=0.1, format="%.2f")
            avg_pickup_delay_min = st.number_input("Avg pickup delay min", min_value=0, step=1)

            # --- Save Button ---
            if st.button("Submit"):
                st.cache_data.clear()
                st.cache_resource.clear()
                
                if booking_status=="Completed":
                    booking_status=1
                elif booking_status=="Incomplete":
                    booking_status=2
                else: booking_status=0

                if ride_distance_km>15:
                    long_distance_flag = 1
                else: 
                    long_distance_flag = 0
                
                # Store data in an array (list)
                Driver_data = [[
                        day_of_week,           
                        is_weekend,              
                        hour_of_day,             
                        traffic_level,           
                        weather_condition,       
                        booking_value,         
                        booking_status,          
                        driver_id,              
                        long_distance_flag,      
                        total_assigned_rides,    
                        accepted_rides,          
                        incomplete_rides,        
                        delay_count,             
                        acceptance_rate,       
                        avg_pickup_delay_min 
                        ]]
                
                result3=model11.predict(Driver_data)

                st.write("Driver Delay Prediction:",result3)

                if result3==0:
                    st.write("No Delay is Expected")
                else : 
                    st.write("Delay is Expected")

