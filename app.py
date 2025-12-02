import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import yfinance as yf
import datetime
from datetime import timedelta
import time
import plotly.graph_objects as go


# Set page configuration
st.set_page_config(
    page_title="Demand zone scanner for indian stock market | RBR and DBR Pattern Scanner",
    page_icon="🔍",
    layout="wide",
)

# Hide Streamlit style
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Load configuration from YAML file
try:
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Configuration file 'config.yaml' not found. Please create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading configuration: {e}")
    st.stop()

# Create authenticator object
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config.get("preauthorized", {"emails": []}),  # Use .get() for optional field
)

# Render the login widget
name, authentication_status, username = authenticator.login("main")


# Helper function for RMA calculation
def rma(series, length):
    """Calculate Running Moving Average (RMA)"""
    alpha = 1 / length
    return series.ewm(alpha=alpha, adjust=False).mean()


# Function to calculate True Range (TR) and ATR
def calculate_atr(data, length=14):
    """
    Calculate Average True Range (ATR) and related metrics.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLC data
    length (int): Period for ATR calculation
    
    Returns:
    pandas.DataFrame: DataFrame with ATR, Range, and Body columns added
    """
    data = data.copy()  # Avoid modifying original DataFrame
    
    data["previous_close"] = data["Close"].shift(1)
    data["tr1"] = abs(data["High"] - data["Low"])
    data["tr2"] = abs(data["High"] - data["previous_close"])
    data["tr3"] = abs(data["Low"] - data["previous_close"])
    data["TR"] = data[["tr1", "tr2", "tr3"]].max(axis=1)
    data["ATR"] = rma(data["TR"], length).round(2)
    data["Range"] = (data["High"] - data["Low"]).round(2)
    data["Body"] = abs(data["Close"] - data["Open"]).round(2)
    
    # Clean up temporary columns
    data.drop(columns=["tr1", "tr2", "tr3", "previous_close"], inplace=True)
    
    return data


def capture_ohlc_data(data, i):
    """
    Captures the OHLC data for the 10 rows before and 15 rows after the pattern.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the OHLC data.
    i (int): The index of the first candle in the pattern.

    Returns:
    pandas.DataFrame: The OHLC data for the surrounding candles.
    """
    start_index = max(0, i - 10)
    end_index = min(len(data), i + 15)
    ohlc_data = data.iloc[start_index:end_index].copy()
    return ohlc_data


def find_patterns(data, interval):
    """
    Find RBR and DBR patterns in the data.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLC and ATR data
    interval (str): Time interval string
    
    Returns:
    list: List of dictionaries containing pattern information
    """
    patterns = []
    
    # Need at least 4 candles to check the pattern
    if len(data) < 5:
        return patterns
    
    for i in range(len(data) - 2, 3, -1):  # Leave room for future candles check
        try:
            # Calculate TR and ATR values with rounding
            tr_value_3 = round(data["TR"].iloc[i - 2], 2)
            atr_value_3 = round(data["ATR"].iloc[i - 2], 2)
            tr_value_2 = round(data["TR"].iloc[i - 1], 2)
            atr_value_2 = round(data["ATR"].iloc[i - 1], 2)
            tr_value_1 = round(data["TR"].iloc[i], 2)
            atr_value_1 = round(data["ATR"].iloc[i], 2)

            # Calculate Body and Range values
            third_candle_range = data["Range"].iloc[i - 2]
            third_candle_body_size = data["Body"].iloc[i - 2]
            second_candle_range = data["Range"].iloc[i - 1]
            second_candle_body_size = data["Body"].iloc[i - 1]
            first_candle_range = data["High"].iloc[i] - data["Close"].iloc[i - 1]
            first_candle_body_size = data["Body"].iloc[i]

            # Fourth candle calculations (candle before leg-in)
            fourth_candle_open = min(data["Open"].iloc[i - 3], data["Close"].iloc[i - 3])
            fourth_candle_close = max(data["Open"].iloc[i - 3], data["Close"].iloc[i - 3])

            third_candle_open = min(data["Open"].iloc[i - 2], data["Close"].iloc[i - 2])
            third_candle_close = max(data["Open"].iloc[i - 2], data["Close"].iloc[i - 2])
            third_candle_body_size_calc = third_candle_close - third_candle_open

            # Calculate overlap between fourth and third candles
            overlap = max(0, min(fourth_candle_close, third_candle_close) - max(
                fourth_candle_open, third_candle_open
            ))

            # Determine proximal_line and stop_loss
            if data["Body"].iloc[i - 1] > 0.50 * data["Range"].iloc[i - 1]:
                proximal_line = max(data["Open"].iloc[i - 1], data["Close"].iloc[i - 1])
            else:
                proximal_line = data["High"].iloc[i - 1]

            # Determine pattern type and stop loss
            if data["Open"].iloc[i - 2] > data["Close"].iloc[i - 2]:
                pattern_name_is = "DBR"
                stop_loss = min(
                    data["Low"].iloc[i], data["Low"].iloc[i - 1], data["Low"].iloc[i - 2]
                )
            else:
                pattern_name_is = "RBR"
                stop_loss = min(data["Low"].iloc[i], data["Low"].iloc[i - 1])

            # Find the last consecutive bullish candle that meets the condition
            j = i
            while (
                j < len(data)
                and data["Close"].iloc[j] > data["Open"].iloc[j]
                and data["TR"].iloc[j] > data["ATR"].iloc[j]
                and data["Body"].iloc[j] > 0.5 * data["Range"].iloc[j]
            ):
                j += 1
            
            if j < len(data):
                first_candle_range = data["High"].iloc[j] - data["Close"].iloc[i - 1]

            # Avoid division by zero
            if third_candle_body_size_calc == 0 or second_candle_body_size == 0:
                continue

            # Check if there's enough data after index i
            if i + 1 >= len(data):
                continue

            # Base conditions for pattern validation
            base_conditions = (
                tr_value_3 > atr_value_3
                and tr_value_2 < atr_value_2
                and tr_value_1 > atr_value_1
                and third_candle_range >= 2 * second_candle_range
                and first_candle_range >= 4 * second_candle_range
                and first_candle_range >= 2 * third_candle_range
                and overlap / third_candle_body_size_calc <= 0.5
                and third_candle_body_size >= 0.5 * third_candle_range
                and second_candle_body_size <= 0.5 * second_candle_range  # Fixed: boring candle should have small body
                and first_candle_body_size >= 0.5 * data["Range"].iloc[i]
                and data["Open"].iloc[i] > max(data["Open"].iloc[i - 1], data["Close"].iloc[i - 1])
            )

            # Zone validation - check future price action
            future_lows = data["Low"].iloc[i + 1:]
            if len(future_lows) == 0:
                continue
                
            min_future_low = future_lows.min()
            
            zone_valid = (
                round(min_future_low) > round(proximal_line)
                or (
                    round(min_future_low) > round(stop_loss)
                    and any(
                        round(data["High"].iloc[k]) < round(proximal_line)
                        for k in range(i + 1, len(data))
                    )
                )
            )

            # Additional condition for DBR pattern
            if data["Open"].iloc[i - 2] > data["Close"].iloc[i - 2]:
                condition = (
                    data["Open"].iloc[i] < data["Open"].iloc[i - 2]
                    and base_conditions
                    and zone_valid
                )
            else:
                condition = base_conditions and zone_valid

            if condition:
                # Format dates based on interval
                if interval in ["1d", "1wk", "1mo", "3mo"]:
                    date_3 = data.index[i - 2].strftime("%d %b %Y")
                    date_1 = data.index[i].strftime("%d %b %Y")
                else:
                    date_3 = data.index[i - 2].strftime("%d %b %Y %H:%M:%S")
                    date_1 = data.index[i].strftime("%d %b %Y %H:%M:%S")

                time_frame = interval
                latest_closing_price = round(data["Close"].iloc[-1], 2)

                # Calculate zone distance
                if round(min_future_low) > round(proximal_line):
                    zone_distance = (
                        (latest_closing_price - proximal_line) / proximal_line * 100
                    )
                else:
                    zone_distance = 0

                ohlc_data = capture_ohlc_data(data, i)

                patterns.append(
                    {
                        "date_3": date_3,
                        "date_1": date_1,
                        "Proximal_line": round(proximal_line, 2),
                        "Distal_line": round(stop_loss, 2),
                        "Eod_close": latest_closing_price,
                        "Pattern_name": pattern_name_is,
                        "Time_frame": time_frame,
                        "zone_distance": round(zone_distance, 2),
                        "OHLC_data": ohlc_data,
                    }
                )
        except (IndexError, KeyError, ZeroDivisionError) as e:
            # Skip this iteration if any error occurs
            continue

    return patterns


def plot_candlestick(
    symbol,
    legin_date,
    legout_date,
    proximal_line,
    distal_line,
    time_frame,
    pattern_name,
    eod_close,
    ohlc_data,
):
    """
    Create a candlestick chart with pattern annotations.
    
    Parameters:
    symbol (str): Stock ticker symbol
    legin_date: Date of leg-in candle
    legout_date: Date of leg-out candle
    proximal_line (float): Proximal line price
    distal_line (float): Distal line (stop loss) price
    time_frame (str): Time frame interval
    pattern_name (str): Name of the pattern (RBR/DBR)
    eod_close (float): End of day closing price
    ohlc_data (DataFrame): OHLC data for the pattern
    
    Returns:
    plotly.graph_objects.Figure: Candlestick chart figure or None if error
    """
    try:
        # Validate ohlc_data has enough rows
        if len(ohlc_data) < 11:
            return None

        stock_data = []
        annotations = []

        # Determine date format based on time_frame
        date_format = (
            "%Y-%m-%d"
            if time_frame in ["1d", "1wk", "1mo", "3mo"]
            else "%Y-%m-%d %H:%M:%S"
        )

        # Collect data and create annotations
        for idx, (index, row) in enumerate(ohlc_data.iterrows()):
            formatted_date = index.strftime(date_format)
            stock_data.append(
                {
                    "time": formatted_date,
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                }
            )

            if idx == 8 and len(ohlc_data) > 8:  # Leg-in candle
                annotations.append(
                    {
                        "x": formatted_date,
                        "y": row["High"],
                        "text": f"legin datetime <br> {formatted_date}",
                        "showarrow": True,
                        "arrowhead": 2,
                        "arrowsize": 1,
                        "arrowwidth": 2,
                        "arrowcolor": "blue",
                        "ax": -20,
                        "ay": -30,
                        "font": dict(size=12, color="black"),
                        "bgcolor": "rgba(255, 255, 0, 0.5)",
                    }
                )
            elif idx == 10 and len(ohlc_data) > 10:  # Leg-out candle
                annotations.append(
                    {
                        "x": formatted_date,
                        "y": row["High"],
                        "text": f"legout datetime <br> {formatted_date}",
                        "showarrow": True,
                        "arrowhead": 2,
                        "arrowsize": 1,
                        "arrowwidth": 2,
                        "arrowcolor": "blue",
                        "ax": -20,
                        "ay": -30,
                        "font": dict(size=12, color="black"),
                        "bgcolor": "rgba(255, 255, 0, 0.5)",
                    }
                )

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=[item["time"] for item in stock_data],
                    open=[item["open"] for item in stock_data],
                    high=[item["high"] for item in stock_data],
                    low=[item["low"] for item in stock_data],
                    close=[item["close"] for item in stock_data],
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                    increasing_fillcolor="#26a69a",
                    decreasing_fillcolor="#ef5350",
                    line_width=0.5,
                )
            ]
        )

        # Add annotations
        fig.update_layout(annotations=annotations)

        # Determine high and low prices dynamically (index 9 is the boring candle)
        if len(ohlc_data) > 10:
            index_9 = ohlc_data.index[9]
            open_9 = ohlc_data.loc[index_9, "Open"]
            close_9 = ohlc_data.loc[index_9, "Close"]
            high_9 = ohlc_data.loc[index_9, "High"]
            low_9 = ohlc_data.loc[index_9, "Low"]

            if abs(open_9 - close_9) > 0.50 * (high_9 - low_9):
                high_price = max(open_9, close_9)
            else:
                high_price = high_9

            # Determine low price based on pattern type
            if (
                ohlc_data.loc[ohlc_data.index[8], "Open"]
                > ohlc_data.loc[ohlc_data.index[8], "Close"]
            ):
                low_price = min(
                    ohlc_data.loc[ohlc_data.index[8], "Low"],
                    low_9,
                    ohlc_data.loc[ohlc_data.index[10], "Low"],
                )
            else:
                low_price = min(low_9, ohlc_data.loc[ohlc_data.index[10], "Low"])
        else:
            high_price = proximal_line
            low_price = distal_line

        total_risk = high_price - low_price
        minimum_target = (total_risk * 5) + high_price

        # Add rectangular shape for the zone
        shape_start = ohlc_data.index[8].strftime(date_format) if len(ohlc_data) > 8 else stock_data[0]["time"]
        shape_end = ohlc_data.index[-1].strftime(date_format)

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=shape_start,
            y0=low_price,
            x1=shape_end,
            y1=high_price,
            fillcolor="green",
            opacity=0.2,
            layer="below",
            line=dict(width=0),
        )

        # Add annotations for entry and stop loss
        if len(ohlc_data) > 17:
            specified_index = ohlc_data.index[17].strftime(date_format)
        elif len(ohlc_data) > 0:
            specified_index = ohlc_data.index[-1].strftime(date_format)
        else:
            return None

        fig.add_annotation(
            x=specified_index,
            y=high_price,
            text=f"{symbol} entry at: {high_price:.2f}",
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=12, color="black"),
            align="center",
            valign="bottom",
            yshift=10,
        )

        fig.add_annotation(
            x=specified_index,
            y=low_price,
            text=f"Stoploss at : {low_price:.2f}",
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=12, color="black"),
            align="center",
            valign="top",
            yshift=-10,
        )

        # Add annotation for minimum_target
        fig.add_annotation(
            x=ohlc_data.index[-1].strftime(date_format),
            y=minimum_target,
            text=f"<b>1:5 Target: {minimum_target:.2f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=-20,
            ay=-30,
            font=dict(size=12, color="black"),
            bgcolor="rgba(0, 255, 0, 0.3)",
        )

        # Custom tick labels with line breaks
        custom_ticks = [item["time"].replace(" ", "<br>") for item in stock_data]

        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                type="category",
                ticktext=custom_ticks,
                tickvals=[item["time"] for item in stock_data],
                tickangle=0,
                tickmode="array",
                tickfont=dict(size=10),
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=100, b=40),
            legend=dict(x=0, y=1.0),
            font=dict(size=12),
            height=600,
            width=800,
            dragmode=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_fixedrange=True,
            yaxis_fixedrange=True,
        )

        # Add styled HTML text annotation header
        header_text = (
            f"<span style='padding-right: 20px;'><b>📊 Chart:</b> {symbol}</span>"
            f"<span style='padding-right: 20px;'><b>🕒 Close:</b> ₹ {eod_close}</span>"
            f"<span style='padding-right: 20px;'><b>🟢 Zone:</b> {pattern_name}</span>"
            f"<span><b>⏳ Time frame:</b> {time_frame}</span>"
        )

        fig.add_annotation(
            x=0.5,
            y=1.20,
            text=header_text,
            showarrow=False,
            align="center",
            xref="paper",
            yref="paper",
            font=dict(size=17, color="white"),
            bgcolor="rgba(0, 0, 0, 0.8)",
            borderpad=4,
            width=800,
            height=50,
            valign="middle",
        )
        
        return fig

    except Exception as e:
        st.error(f"Error occurred while creating chart for {symbol}: {e}")
        return None


def download_stock_data(ticker, period, interval):
    """
    Download stock data from Yahoo Finance.
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Data period
    interval (str): Data interval
    
    Returns:
    pandas.DataFrame: Stock data or empty DataFrame if error
    """
    try:
        data = yf.download(
            ticker + ".NS", period=period, interval=interval, progress=False
        )
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return pd.DataFrame()


def process_2h_data(ticker, period):
    """
    Process data for 2-hour interval by resampling from 1-hour data.
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Data period
    
    Returns:
    pandas.DataFrame: Resampled 2-hour data
    """
    try:
        df = yf.download(ticker + ".NS", period=period, interval="1h", progress=False)
        
        if df.empty:
            return pd.DataFrame()
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.round(2)
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Reset index to work with groupby
        df = df.reset_index()
        
        # Find the datetime column name
        datetime_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
        
        if datetime_col is None:
            datetime_col = df.columns[0]
        
        # Resample to 2H
        df_agg = df.groupby(
            pd.Grouper(key=datetime_col, freq="2H", origin="start")
        ).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            }
        )
        
        # Remove NaN rows
        df_final = df_agg.dropna(how="all")
        
        return df_final
        
    except Exception as e:
        return pd.DataFrame()


# Initialize session state for authentication if not exists
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None


# Main Application Logic
if st.session_state["authentication_status"] is True:
    authenticator.logout("Logout", "main")
    st.write(f'Welcome *{st.session_state["name"]}*')
    
    st.markdown(
        "<h1 style='text-align: center;'> Welcome to demand zone scanner </h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<img src='https://yt3.googleusercontent.com/OfYP3wK5_zJVC8xSogqLBU9MzCDuzitraf3vc8L9hrecWZvRX8wo5hqgel_eQyhS_Z7jbp881x8=s160-c-k-c0x00ffffff-no-rj' width='200' style='display: block; margin: 0 auto;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style='background-color: #E8F2FC; padding: 10px; border-radius: 5px;'>
        Remember - "Making money in the stock market is as easy as making a cup of tea but <strong>your trade should be process oriented not profit oriented.</strong>" <br> <span>- Dr. Ravi R Kumar
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<h5 style='text-align: center;'>Try scanner👇</h5>", unsafe_allow_html=True)

    st.write(
        "To also make finding demand zone as easy as making a cup of tea. And don't forget to join our telegram [channel](https://t.me/dr_ravi_r_kumar_course_notes)."
    )
    st.markdown("___________")

    # Get user input for tickers
    ticker_option = st.radio(
        "Select ticker option:",
        ["Custom Symbol", "Nifty50stocks", "Bhavcopy Stocks"],
        horizontal=True,
    )

    tickers = []
    
    if ticker_option == "Custom Symbol":
        user_tickers = st.text_input("Enter tickers (comma-separated):", "")
        if user_tickers.strip():
            tickers = [ticker.strip().upper() for ticker in user_tickers.split(",") if ticker.strip()]
    elif ticker_option == "Bhavcopy Stocks":
        hot_stocks = [
            "INFY", "MPHASIS", "ASTRAL", "ITC", "INDIACEM", "ASIANPAINT",
            "BERGEPAINT", "LALPATHLAB", "COROMANDEL", "AUBANK", "HCLTECH",
            "COLPAL", "GLENMARK", "TITAN",
        ]
        tickers = st.multiselect("Select hot stocks:", hot_stocks, default=hot_stocks)
    else:  # Nifty50stocks
        predefined_tickers = [
            "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
            "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL",
        ]
        tickers = st.multiselect(
            "Select predefined tickers:", predefined_tickers, default=predefined_tickers
        )

    # Get user input for interval
    valid_intervals = ["1m", "5m", "15m", "30m", "1h", "2h", "1d", "1wk", "1mo", "3mo"]
    interval = st.selectbox(
        "Select interval:", valid_intervals, index=valid_intervals.index("2h")
    )

    # Set the default period based on the selected interval
    period_map = {
        "1m": "7d",
        "5m": "1mo",
        "15m": "1mo",
        "30m": "1mo",
        "1h": "2y",
        "2h": "1y",
    }
    period = period_map.get(interval, "2y")

    # Add a button to scan for patterns
    find_patterns_button = st.button("🔍 Scan Demand Zone")

    if find_patterns_button:
        if not tickers:
            st.warning("Please enter at least one ticker symbol.")
        else:
            # Create an empty DataFrame to store the patterns
            patterns_df = pd.DataFrame(
                columns=[
                    "Ticker", "Eod_close", "zone_distance", "Legin_date", "Legout_date",
                    "Proximal_line", "Distal_line", "Pattern_name", "Time_frame",
                ]
            )
            
            chart_figures = []
            zone_distances = []
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            any_patterns_found = False
            valid_tickers_count = 0

            for i, ticker in enumerate(tickers):
                progress_percent = (i + 1) / len(tickers)
                progress_bar.progress(progress_percent)
                progress_text.text(
                    f"🔍 Scanning Demand Zone {i + 1} of {len(tickers)} - {progress_percent * 100:.2f}% Complete"
                )

                try:
                    if interval == "2h":
                        data = process_2h_data(ticker, period)
                    else:
                        data = download_stock_data(ticker, period, interval)

                    if data.empty:
                        continue

                    valid_tickers_count += 1
                    data = data.round(2)
                    data = calculate_atr(data)
                    patterns = find_patterns(data, interval)

                    if patterns:
                        any_patterns_found = True

                        for pattern in patterns:
                            fig = plot_candlestick(
                                ticker,
                                pattern["date_3"],
                                pattern["date_1"],
                                pattern["Proximal_line"],
                                pattern["Distal_line"],
                                pattern["Time_frame"],
                                pattern["Pattern_name"],
                                pattern["Eod_close"],
                                pattern["OHLC_data"],
                            )
                             
                            if fig is not None:
                                chart_figures.append(fig)
                                zone_distances.append(pattern["zone_distance"])

                            # Append pattern data to patterns_df
                            new_row = pd.DataFrame([{
                                "Ticker": ticker,
                                "Eod_close": pattern["Eod_close"],
                                "zone_distance": pattern["zone_distance"],
                                "Legin_date": pattern["date_3"],
                                "Legout_date": pattern["date_1"],
                                "Proximal_line": pattern["Proximal_line"],
                                "Distal_line": pattern["Distal_line"],
                                "Pattern_name": pattern["Pattern_name"],
                                "Time_frame": pattern["Time_frame"],
                            }])
                            patterns_df = pd.concat([patterns_df, new_row], ignore_index=True)

                except Exception as e:
                    continue

            # Sort and format results
            if not patterns_df.empty:
                patterns_df = patterns_df.sort_values(by="zone_distance", ascending=True)
                patterns_df["zone_distance"] = patterns_df["zone_distance"].apply(
                    lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                )
                patterns_df = patterns_df.reset_index(drop=True)

            # Sort chart figures by zone distance
            if zone_distances and chart_figures:
                sorted_figures = sorted(zip(zone_distances, chart_figures), key=lambda x: x[0])
                sorted_chart_figures = [fig for _, fig in sorted_figures]
            else:
                sorted_chart_figures = []

            progress_bar.empty()
            progress_text.empty()

            if any_patterns_found:
                st.success(
                    "Scanning is complete. Below is the table data and chart view of stocks that match Ravi R. Kumar's Zone Validation rules."
                )
                
                tab1, tab2 = st.tabs(["📁 Data", "📈 Chart"])

                with tab1:
                    st.markdown("**Table View**")
                    st.dataframe(patterns_df, use_container_width=True, hide_index=True)

                with tab2:
                    st.markdown("**Chart View**")
                    for fig in sorted_chart_figures:
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=False)
            else:
                if valid_tickers_count > 0:
                    st.warning(
                        """When validating the demand zone pattern for the given time frame interval, we didn't find any Rally Boring Rally or Drop Boring Rally pattern that fulfills these Ravi R Kumar zone validation rules:

1. The boring candle to leg-in candle ratio should be 1:2, and the boring candle to leg-out candle ratio should be 1:4. This means 'one two ka four' should be there.
2. The leg-in candle TR should be greater than the boring candle TR, which should be less than ATR, and the leg-out candle TR should be greater than its ATR.
3. There should be a white area in the zone.
4. The candle body behind the leg-in candle should not cover more than 50% of the leg-in candle; when checking these parameters, the zone becomes invalid.
5. Leg-out candle formation - In the Drop Boring Rally pattern, the leg-out candle opening should not be greater than the leg-in candle opening.

Dr. Ravi R Kumar says he has 14 zone validation rules. If any zone passes these basic zone validations, then you don't need to check other parameters. As per our analysis, no zone was fulfilling the basic validation rules."""
                    )
                else:
                    st.error("No valid tickers found. Please check your ticker symbols.")

elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")
else:  # authentication_status is None
    st.warning("Please enter your username and password")
