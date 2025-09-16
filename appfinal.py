

from prophet import Prophet
import pandas as pd

import pandas as pd
import streamlit as st
import pickle





def main():
    # Set the app title

    # Apply custom CSS styling
    st.markdown(
        """
        <style>
            /* Change the font family and color of the heading */
            .title-wrapper {
                font-family: 'Arial', sans-serif;
                color: blue;
            }
            

        </style>
        <h1 class="title-wrapper">Prophet Model Deployment G-3</h1>
        """,
        unsafe_allow_html=True
    )

    df = pd.read_csv("crude-oil-price.csv")
    Newdf = df.copy()
    Newdf['date'] = pd.to_datetime(Newdf.date, format="%Y-%m-%dT%H:%M:%S")
    dfm=pd.DataFrame(data=Newdf.iloc[:,1])

    # Add a date input widget
    date_input = st.date_input("Enter a date:", value=None, min_value=None, max_value=None)


    if st.button("Predict"):
        # Prepare the input data for prediction
        input_date = pd.to_datetime(date_input)
        input_data = pd.DataFrame({'ds': [input_date]})

        # Generate predictions
        with open('prophet.pkl', 'rb') as file:
            loadmodel = pickle.load(file)

        predictions = loadmodel.predict(input_data)

        # Extract price, upper range, and lower range values
        price = predictions['yhat'].values[0]
        upper_range = predictions['yhat_upper'].values[0]
        lower_range = predictions['yhat_lower'].values[0]

        # Display the predictions
        st.subheader('Predicted Price')
        st.write("Forecasted Oil Price(USD/BBL):", price)
        #st.write("Upper Range(USD/BBL):", upper_range)
        #st.write("Lower Range(USD/BBL):", lower_range)
        # Generate future dates for prediction
        future_dates = loadmodel.make_future_dataframe(periods=365)  # Adjust the number of periods as needed
        future_dates = future_dates[future_dates['ds'] <= input_date]

        predictions_plot = loadmodel.predict(future_dates)
        predictions_df = predictions_plot[predictions_plot['ds'] <= input_date]
        predictions_df['year'] = predictions_df['ds'].dt.year

        # Visualize the predicted graph
        st.subheader("Predicted Graph")
        chart_data = pd.concat([df['price'], predictions_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]])
        chart_data.set_index('ds', inplace=True)
        st.line_chart(chart_data)


if __name__ == "__main__":
    main()

