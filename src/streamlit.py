import streamlit as st
import requests

st.title("Video Game Sales Prediction")
st.subheader("Anjay gurinjay :sunglasses:🤙")

# Create form of input
with st.form(key = "video_game_sales"):
    na_sales_input = st.number_input(
        label = "1.\tInsert Sales value from NA Region:",
        min_value = 0,
        max_value = 999,
        help = "Value range from 0 to 999"
    )

    eu_sales_input = st.number_input(
        label = "2.\tInsert Sales value from EU Region:",
        min_value = 0,
        max_value = 999,
        help = "Value range from 0 to 999"
    )

    jp_sales_input = st.number_input(
        label = "3.\tInsert Sales value from JP Region:",
        min_value = 0,
        max_value = 999,
        help = "Value range from 0 to 999"
    )

    other_sales_input = st.number_input(
        label = "4.\tInsert Sales value from Other Region:",
        min_value = 0,
        max_value = 999,
        help = "Value range from 0 to 999"
    )

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Create dict data input
        raw_data = {
            "NA_Sales": na_sales_input,
            "EU_Sales": eu_sales_input,
            "JP_Sales": jp_sales_input,
            "Other_Sales": other_sales_input
        }

        with st.spinner("Sending data to server ..."):
            res = requests.post("http://api:8080/predict", json = raw_data).json()

        # parse result
        if res["error_msg"] != "":
            st.error("Error occurs while predicting: {}".format(res["error_msg"]))

        else:
            st.success("Predicted Global Sales based on data: {}".format(res["res"]))