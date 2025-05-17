# Refactored Streamlit App for Better Responsiveness
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from calendar import month_name

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Retail Analytics & Forecasting")

# --- HEADER IMAGE ---
st.markdown("""
""", unsafe_allow_html=True)

# --- Cached Loading Functions ---
@st.cache_data
def load_dashboard_data():
    df = pd.read_csv("synthetic_online_retail_data.xls")
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

@st.cache_data
def load_forecast_data():
    return pd.read_csv("encoded_data.csv")

@st.cache_resource
def load_model_scaler():
    model = joblib.load("sales_forecast_model.pkl")
    scaler = joblib.load("sales_forecast_scaler.pkl")
    return model, scaler

@st.cache_resource
def initialize_resources():
    df_dash = load_dashboard_data()
    df_forecast = load_forecast_data()
    model, scaler = load_model_scaler()
    return df_dash, df_forecast, model, scaler

# --- Load all resources once ---
df_dash, df_forecast, model, scaler = initialize_resources()

# --- Lookup tables ---
product_encoded_table = df_forecast[['product_name', 'product_encoded']].drop_duplicates()
city_encoded_table    = df_forecast[['city', 'city_encoded']].drop_duplicates()

# --- Tabs ---
tab1, tab2 = st.tabs(["\U0001F4CA Retail Dashboard", "\U0001F4C8 Sales Profit Forecasting"])

# --- Dashboard Tab ---
with tab1:
    st.title("\U0001F6CD\ufe0f Online Retail Dashboard")

    # Sidebar Filters
    st.sidebar.header("\U0001F50D Filters")
    cities  = df_dash['city'].dropna().unique().tolist()
    genders = df_dash['gender'].dropna().unique().tolist()

    selected_city   = st.sidebar.selectbox("City for 'Sales by City'", ["All"] + cities)
    selected_gender = st.sidebar.selectbox("Gender for 'Spend by Age Group'", ["All"] + genders)

    tab1a, tab1b, tab1c = st.tabs(["\U0001F4C8 Sales & Trends", "\U0001F9CD\u200D\u2642\ufe0f Customers", "\U0001F9E9 Products"])

    with tab1a:
        st.subheader("Monthly Sales Trend")
        monthly_sales = df_dash.set_index('order_date').resample('M')['price'].sum()
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        monthly_sales.plot(marker='o', linestyle='-', color='green', ax=ax1)
        ax1.set_title("Monthly Sales Trend")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Total Sales")
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("Top 10 Sales by City")
        data_city = df_dash if selected_city == "All" else df_dash[df_dash['city'] == selected_city]
        city_sales = data_city.groupby('city')['price'].sum().nlargest(10)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        city_sales.plot(kind='barh', color='teal', ax=ax2)
        ax2.set_title("Top 10 Sales by City")
        st.pyplot(fig2)

    with tab1b:
        st.subheader("Top 10 Customers by Spend")
        top_customers = df_dash.groupby('customer_id')['price'].sum().nlargest(10)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        top_customers.plot(kind='bar', color='skyblue', ax=ax3)
        st.pyplot(fig3)

        st.subheader("Spend by Age Group")
        age_df = df_dash.copy()
        if selected_gender != "All":
            age_df = age_df[age_df['gender'] == selected_gender]
        age_df['age_group'] = pd.cut(age_df['age'], bins=[0, 18, 25, 35, 45, 60, 100],
                                     labels=['<18','18-25','26-35','36-45','46-60','60+'])
        age_spend = age_df.groupby('age_group')['price'].sum()
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        age_spend.plot(kind='bar', color='coral', ax=ax4)
        ax4.set_title("Spend by Age Group")
        st.pyplot(fig4)

    with tab1c:
        st.subheader("Top 10 Products by Quantity")
        top_products = df_dash.groupby('product_name')['quantity'].sum().nlargest(10)
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        top_products.plot(kind='bar', color='orange', ax=ax5)
        st.pyplot(fig5)

        st.subheader("Average Review Score by Category")
        avg_review = df_dash.groupby('category_name')['review_score'].mean().sort_values()
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        avg_review.plot(kind='barh', color='purple', ax=ax6)
        st.pyplot(fig6)

    st.markdown("---")
    st.markdown("Made By Ahmed Sheikh ❤️")

# --- Forecasting Tab ---
with tab2:
    st.title("\U0001F4C8 Sales Profit Forecasting")
    st.markdown("Fill in all fields and click **Forecast Profit** to see current and next-month profit estimates.")

    with st.form("forecast_form"):
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input("Quantity", min_value=0.0, format="%.2f")
        with col2:
            price = st.number_input("Price", min_value=0.0, format="%.2f")

        col3, col4 = st.columns(2)
        with col3:
            discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, format="%.1f")
            discounted_price = price * (1 - discount / 100)
        with col4:
            product_name = st.selectbox("Product Name", product_encoded_table['product_name'])
            product_encoded = product_encoded_table.query("product_name == @product_name")['product_encoded'].iloc[0]

        col5, col6 = st.columns(2)
        with col5:
            city = st.selectbox("City", city_encoded_table['city'])
            city_encoded = city_encoded_table.query("city == @city")['city_encoded'].iloc[0]
        with col6:
            season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: ["Winter", "Spring", "Summer", "Fall"][x-1])

        col7, col8 = st.columns(2)
        with col7:
            order_month = st.selectbox("Order Month", list(range(1,13)), format_func=lambda x: month_name[x])
        with col8:
            order_day = st.number_input("Order Day", min_value=1, max_value=31, step=1)

        col9, col10 = st.columns(2)
        with col9:
            order_weekday = st.selectbox("Order Weekday", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        with col10:
            order_year = st.number_input("Order Year", min_value=2000, max_value=2100, step=1)

        col11, col12 = st.columns(2)
        with col11:
            last_month_profit = st.number_input("Last Month Profit", format="%.2f")
        with col12:
            avg_last_3_months_profit = st.number_input("Avg. Last 3 Months Profit", format="%.2f")

        col13, col14 = st.columns(2)
        with col13:
            month_over_month_change = st.number_input("Month-over-Month Change", format="%.4f")
        with col14:
            cumulative_sales_to_date = st.number_input("Cumulative Sales to Date", format="%.2f")

        submitted = st.form_submit_button("\U0001F4CA Forecast Profit")

        if submitted:
            row = dict(
                quantity=quantity,
                price=discounted_price,
                product_encoded=product_encoded,
                city_encoded=city_encoded,
                last_month_profit=last_month_profit,
                avg_last_3_months_profit=avg_last_3_months_profit,
                month_over_month_change=month_over_month_change,
                cumulative_sales_to_date=cumulative_sales_to_date,
                season=season,
                order_month=order_month,
                order_day=order_day,
                order_weekday=order_weekday,
                order_year=order_year
            )
            df_input = pd.DataFrame([row])
            scaled_input = scaler.transform(df_input)
            first_pred = model.predict(scaled_input)[0]

            next_row = row.copy()
            next_row['order_month'] = (order_month % 12) + 1
            next_row['order_year'] += 1 if next_row['order_month'] == 1 else 0
            next_row['last_month_profit'] = first_pred
            next_row['avg_last_3_months_profit'] = (avg_last_3_months_profit * 2 + first_pred)/3
            next_row['month_over_month_change'] = ((first_pred - last_month_profit)/last_month_profit if last_month_profit else 0.0)
            next_row['cumulative_sales_to_date'] = cumulative_sales_to_date + first_pred
            next_row['season'] = ((next_row['order_month'] % 12 + 3) // 3)

            df_next = pd.DataFrame([next_row])
            scaled_next = scaler.transform(df_next)
            second_pred = model.predict(scaled_next)[0]

            st.balloons()
            st.markdown(f"""
                <div style='background-color:#1f4788; padding:20px; border-radius:12px; color:white; text-align:center;'>
                  <h3>\U0001F4B0 Profit Forecast</h3>
                  <p style='font-size:24px;'>For <strong>{month_name[order_month]}</strong>: 
                    <span style='color:#00ffcc;'>{first_pred:.2f}</span></p>
                  <p style='font-size:20px;'>Next Month (<strong>{month_name[next_row['order_month']]}</strong>): 
                    <span style='color:#ffd700;'>{second_pred:.2f}</span></p>
                </div>
            """, unsafe_allow_html=True)
            st.toast(f"Forecast done for {month_name[order_month]} & {month_name[next_row['order_month']]}", icon="\U0001F4C8")
