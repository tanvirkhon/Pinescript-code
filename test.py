import pandas as pd
import numpy as np
import dontshare_config as ds
import nice_funcs as n
from eth_account.signers.local import LocalAccount
import eth_account
import time
import schedule
import datetime
import json
import usdt_dominance  # Import the USDT dominance module

# Parameters
symbol = 'SOL'
timeframe = '5m'
max_positions = 1
leverage = 2
stop_loss_atr_multiplier_high_volatility = 2.5
stop_loss_atr_multiplier_low_volatility = 2.0
trailing_stop_atr_multiplier_high_volatility = 2.0
trailing_stop_atr_multiplier_low_volatility = 1.5
risk_to_reward_ratio = 1.05
cooldown_minutes = 10
short_term_ma_period = 5
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 20
volatility_threshold = 0.5
pullback_threshold = 0.01

# File path for saving state
state_file_path = 'trading_bot_state.json'
usdt_dominance_value = None  # Initialize USDT dominance value

# Function to save state
def save_state(file_path, entry_px1, trailing_stop_price, cumulative_roi, take_profit_price, initial_stop_loss_price, last_trade_time, usdt_dominance_value):
    state = {
        'entry_px1': entry_px1,
        'trailing_stop_price': trailing_stop_price,
        'cumulative_roi': cumulative_roi,
        'take_profit_price': take_profit_price,
        'initial_stop_loss_price': initial_stop_loss_price,
        'last_trade_time': last_trade_time.isoformat() if last_trade_time else None,
        'usdt_dominance_value': usdt_dominance_value
    }
    with open(file_path, 'w') as file:
        json.dump(state, file)

# Function to load state
def load_state(file_path):
    try:
        with open(file_path, 'r') as file:
            state = json.load(file)
            entry_px1 = state.get('entry_px1', None)
            trailing_stop_price = state.get('trailing_stop_price', None)
            cumulative_roi = state.get('cumulative_roi', 0.0)
            take_profit_price = state.get('take_profit_price', None)
            initial_stop_loss_price = state.get('initial_stop_loss_price', None)
            last_trade_time = datetime.datetime.fromisoformat(state.get('last_trade_time')) if state.get('last_trade_time') else None
            usdt_dominance_value = state.get('usdt_dominance_value', None)
            return entry_px1, trailing_stop_price, cumulative_roi, take_profit_price, initial_stop_loss_price, last_trade_time, usdt_dominance_value
    except FileNotFoundError:
        return None, None, 0.0, None, None, None, None

# Initialize variables
entry_px1, trailing_stop_price, cumulative_roi, take_profit_price, initial_stop_loss_price, last_trade_time, usdt_dominance_value = load_state(state_file_path)

# API Secret
secret = ds.private_key

# Function to safely execute API calls with rate limiting
def safe_api_call(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        print(f"--- Error Encountered ---")
        print(f"Error during API call: {e}")
        print(f"Function: {func.__name__}")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        time.sleep(1)
        return safe_api_call(func, *args, **kwargs)

# Define moving average function
def EMA(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().values

# Define ATR function
def ATR(high, low, close, period):
    high_low = high - low
    high_close = np.abs(high[1:] - close[:-1])
    low_close = np.abs(low[1:] - close[:-1])
    true_range = np.maximum.reduce([high_low[1:], high_close, low_close])
    true_range = np.concatenate(([np.nan], true_range))  # Adjust the length
    atr = pd.Series(true_range).rolling(period).mean().values
    return atr

# Define RSI function
def RSI(data, period):
    delta = pd.Series(data).diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

# Function to update USDT dominance
def update_usdt_dominance():
    global usdt_dominance_value
    usdt_dominance_value = usdt_dominance.get_current_usdt_dominance_coingecko()
    usdt_dominance.save_usdt_dominance(usdt_dominance_value)  # Save current dominance
    print(f"Updated USDT Dominance: {usdt_dominance_value:.2f}")

# Main bot function implementing the simplified strategy
def bot():
    global last_trade_time, entry_px1, trailing_stop_price, cumulative_roi, take_profit_price, initial_stop_loss_price

    try:
        account1 = eth_account.Account.from_key(secret)
        
        result = safe_api_call(n.get_position_andmaxpos, symbol, account1, max_positions)
        positions1, im_in_pos, mypos_size, pos_sym1, entry_px1, pnl_perc1, long1, num_of_pos = result[:8]

        lev, pos_size = safe_api_call(n.adjust_leverage_size_signal, symbol, leverage, account1)
        pos_size = pos_size / 2 

        # Get historical data
        snapshot_data = safe_api_call(n.get_ohlcv2, symbol, timeframe, 500)
        df = n.process_data_to_df(snapshot_data)

        # Rename columns to ensure they match the expected names
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

        # Convert columns to numeric types
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])

        # Calculate indicators
        df['ma20'] = EMA(df['Close'].values, 20)
        df['ma25'] = EMA(df['Close'].values, 25)
        df['ma30'] = EMA(df['Close'].values, 30)
        df['ma35'] = EMA(df['Close'].values, 35)
        df['ma40'] = EMA(df['Close'].values, 40)
        df['ma45'] = EMA(df['Close'].values, 45)
        df['ma50'] = EMA(df['Close'].values, 50)
        df['ma55'] = EMA(df['Close'].values, 55)
        df['ma60'] = EMA(df['Close'].values, 60)
        df['ma65'] = EMA(df['Close'].values, 65)
        df['ma70'] = EMA(df['Close'].values, 70)
        df['ma200'] = EMA(df['Close'].values, 200)
        df['short_term_ma'] = EMA(df['Close'].values, short_term_ma_period)

        df['ATR'] = ATR(df['High'].values, df['Low'].values, df['Close'].values, 14)
        df['RSI'] = RSI(df['Close'].values, rsi_period)

        close = df['Close'].values[-1]
        ma_values = [df['ma20'].values[-1], df['ma25'].values[-1], df['ma30'].values[-1], df['ma35'].values[-1], df['ma40'].values[-1], df['ma45'].values[-1], df['ma50'].values[-1], df['ma55'].values[-1], df['ma60'].values[-1], df['ma65'].values[-1], df['ma70'].values[-1]]
        ma200 = df['ma200'].values[-1]
        short_term_ma = df['short_term_ma'].values[-1]
        atr = df['ATR'].values[-1]
        rsi = df['RSI'].values[-1]

        # Calculate recent ATR for dynamic adjustment
        recent_atr = df['ATR'].values[-10:].mean()
        if recent_atr > volatility_threshold:
            stop_loss_atr_multiplier = stop_loss_atr_multiplier_high_volatility
            trailing_stop_atr_multiplier = trailing_stop_atr_multiplier_high_volatility
        else:
            stop_loss_atr_multiplier = stop_loss_atr_multiplier_low_volatility
            trailing_stop_atr_multiplier = trailing_stop_atr_multiplier_low_volatility

        current_time = datetime.datetime.now()

        print(f'--- Current Market Data ---')
        print(f'Close Price: {close:.2f}')
        print(f'MA values: {ma_values}')
        print(f'MA 200: {ma200:.2f}')
        print(f'Short-term MA: {short_term_ma:.2f}')
        print(f'ATR: {atr:.2f}')
        print(f'RSI: {rsi:.2f}')
        print(f'--- Current Position Data ---')
        print(f'We are in {num_of_pos} positions and max pos is {max_positions}')
        print(f'Current PNL percentage: {pnl_perc1:.2f}%')
        print(f'Cumulative ROI: {cumulative_roi:.2f}%')
        print(f'Leverage: {lev}')
        print(f'Entry Price: {entry_px1}')

        if last_trade_time:
            time_since_last_trade = (current_time - last_trade_time).total_seconds() / 60
            print(f'Time since last trade exit: {time_since_last_trade:.2f} minutes')

        # Use USDT dominance value to decide trade direction
        if usdt_dominance_value is None:
            update_usdt_dominance()

        usdt_dominance_positive = usdt_dominance.is_usdt_dominance_positive()
        print(f'USDT Dominance is positive: {usdt_dominance_positive}')

        # Define trade direction based on USDT dominance
        long_trade_allowed = not usdt_dominance_positive
        short_trade_allowed = usdt_dominance_positive

        if not im_in_pos and (not last_trade_time or time_since_last_trade > cooldown_minutes):
            print(f'--- Checking Entry Conditions ---')

            if long_trade_allowed:
                all_ma_above_ma200 = all(ma > ma200 for ma in ma_values)
                rsi_not_overbought = rsi <= rsi_overbought
                close_to_short_term_ma = abs(close - short_term_ma) / short_term_ma <= pullback_threshold
                print(f'All MAs above MA 200: {all_ma_above_ma200}')
                print(f'RSI <= {rsi_overbought}: {rsi_not_overbought}')
                print(f'Close to short-term MA: {close_to_short_term_ma}')

                if all_ma_above_ma200 and rsi_not_overbought and close_to_short_term_ma:
                    entry_px1 = close
                    initial_stop_loss_price = entry_px1 - stop_loss_atr_multiplier * atr
                    take_profit_price = entry_px1 + (entry_px1 - initial_stop_loss_price) * risk_to_reward_ratio
                    trailing_stop_price = initial_stop_loss_price  # Start trailing stop at initial stop loss

                    # Debug prints
                    print(f'Entry Price: {entry_px1}')
                    print(f'ATR: {atr}')
                    print(f'Initial Stop Loss Price: {initial_stop_loss_price}')
                    print(f'Take Profit Price: {take_profit_price}')
                    print(f'Trailing Stop Price: {trailing_stop_price}')

                    pos_size = float(pos_size)
                    limit_px = float(close)

                    print(f'--- Placing Long Order ---')
                    safe_api_call(n.cancel_all_orders, account1)
                    order_result = safe_api_call(n.limit_order, symbol, True, pos_size, limit_px, False, account1)
                    print(f'Placed long order for {pos_size} at {limit_px}')
                    print(f'Order Result: {order_result}')
                    last_trade_time = current_time

                    # Save state after entering a trade
                    save_state(state_file_path, entry_px1, trailing_stop_price, cumulative_roi, take_profit_price, initial_stop_loss_price, last_trade_time, usdt_dominance_value)
                else:
                    print('Not entering long trade: conditions for long entry not met')

            if short_trade_allowed:
                all_ma_below_ma200 = all(ma < ma200 for ma in ma_values)
                rsi_not_oversold = rsi >= rsi_oversold
                close_to_short_term_ma = abs(close - short_term_ma) / short_term_ma <= pullback_threshold
                print(f'All MAs below MA 200: {all_ma_below_ma200}')
                print(f'RSI >= {rsi_oversold}: {rsi_not_oversold}')
                print(f'Close to short-term MA: {close_to_short_term_ma}')

                if all_ma_below_ma200 and rsi_not_oversold and close_to_short_term_ma:
                    entry_px1 = close
                    initial_stop_loss_price = entry_px1 + stop_loss_atr_multiplier * atr
                    take_profit_price = entry_px1 - (initial_stop_loss_price - entry_px1) * risk_to_reward_ratio
                    trailing_stop_price = initial_stop_loss_price  # Start trailing stop at initial stop loss

                    # Debug prints
                    print(f'Entry Price: {entry_px1}')
                    print(f'ATR: {atr}')
                    print(f'Initial Stop Loss Price: {initial_stop_loss_price}')
                    print(f'Take Profit Price: {take_profit_price}')
                    print(f'Trailing Stop Price: {trailing_stop_price}')

                    pos_size = float(pos_size)
                    limit_px = float(close)

                    print(f'--- Placing Short Order ---')
                    safe_api_call(n.cancel_all_orders, account1)
                    order_result = safe_api_call(n.limit_order, symbol, False, pos_size, limit_px, False, account1)
                    print(f'Placed short order for {pos_size} at {limit_px}')
                    print(f'Order Result: {order_result}')
                    last_trade_time = current_time

                    # Save state after entering a trade
                    save_state(state_file_path, entry_px1, trailing_stop_price, cumulative_roi, take_profit_price, initial_stop_loss_price, last_trade_time, usdt_dominance_value)
                else:
                    print('Not entering short trade: conditions for short entry not met')

        elif im_in_pos:
            current_price = float(df.iloc[-1]['Close'])

            # Determine if the current position is long or short
            is_long_position = long1

            # Update trailing stop price if the new one is higher
            if is_long_position:
                new_trailing_stop_price = current_price - trailing_stop_atr_multiplier * atr
                if trailing_stop_price is None:
                    trailing_stop_price = initial_stop_loss_price
                else:
                    trailing_stop_price = max(trailing_stop_price, new_trailing_stop_price)
            else:
                new_trailing_stop_price = current_price + trailing_stop_atr_multiplier * atr
                if trailing_stop_price is None:
                    trailing_stop_price = initial_stop_loss_price
                else:
                    trailing_stop_price = min(trailing_stop_price, new_trailing_stop_price)

            print(f'--- Position Management ---')
            print(f'Take Profit Price: {take_profit_price:.2f}')
            print(f'Initial Stop Loss Price: {initial_stop_loss_price:.2f}')
            print(f'Trailing Stop Price: {trailing_stop_price:.2f}')
            print(f'Current Price: {current_price:.2f}')

            # Save state after updating trailing stop price
            save_state(state_file_path, entry_px1, trailing_stop_price, cumulative_roi, take_profit_price, initial_stop_loss_price, last_trade_time, usdt_dominance_value)

            if (is_long_position and current_price >= take_profit_price) or (not is_long_position and current_price <= take_profit_price):
                safe_api_call(n.cancel_all_orders, account1)
                safe_api_call(n.close_all_positions, account1)
                print(f'--- Closing Position ---')
                print(f'Closed position at {close} due to reaching take profit price of {take_profit_price:.2f}')
                last_trade_time = current_time
                cumulative_roi += pnl_perc1  # Update cumulative ROI
                save_state(state_file_path, None, None, cumulative_roi, None, None, last_trade_time, usdt_dominance_value)  # Clear state
            elif (is_long_position and current_price <= initial_stop_loss_price) or (not is_long_position and current_price >= initial_stop_loss_price):
                safe_api_call(n.cancel_all_orders, account1)
                safe_api_call(n.close_all_positions, account1)
                print(f'--- Closing Position ---')
                print(f'Closed position at {close} due to close price below initial stop loss price')
                last_trade_time = current_time
                cumulative_roi += pnl_perc1  # Update cumulative ROI
                save_state(state_file_path, None, None, cumulative_roi, None, None, last_trade_time, usdt_dominance_value)  # Clear state
            elif (is_long_position and current_price <= trailing_stop_price) or (not is_long_position and current_price >= trailing_stop_price):
                safe_api_call(n.cancel_all_orders, account1)
                safe_api_call(n.close_all_positions, account1)
                print(f'--- Closing Position ---')
                print(f'Closed position at {close} due to close price below trailing stop price')
                last_trade_time = current_time
                cumulative_roi += pnl_perc1  # Update cumulative ROI
                save_state(state_file_path, None, None, cumulative_roi, None, None, last_trade_time, usdt_dominance_value)  # Clear state

    except Exception as e:
        print(f'--- Error Encountered ---')
        print(f'Error: {e}')
        print('*** maybe internet connection lost... sleeping 30 and retrying')
        time.sleep(30)

# Update USDT dominance at the start
update_usdt_dominance()
schedule.every(15).minutes.do(update_usdt_dominance)  # Schedule the USDT dominance update every 15 minutes

bot()
schedule.every(30).seconds.do(bot)

while True:
    try:
        schedule.run_pending()
        time.sleep(10)
    except Exception as e:
        print(f'--- Error in Main Loop ---')
        print(f'Error: {e}')
        print('*** maybe internet connection lost... sleeping 30 and retrying')
        time.sleep(30)
