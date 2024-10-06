# from decimal import Decimal
# import pandas as pd

# def get_index_eod(date):
#     ticker = 'spx'
#     year = date[:4]
#     eod_file = f"/data/thetadata/indexes/{ticker}/eod/{year}.csv.gz"
#     eods = pd.read_csv(eod_file, compression='gzip')

#     # Convert 'date' column to string to ensure compatibility
#     eods['date'] = eods['date'].astype(str)

#     # Filter by date
#     eod = eods[eods['date'] == date]

#     # Convert 'date' column to datetime
#     eod.loc[:, 'date'] = pd.to_datetime(eod['date'], format='%Y%m%d')

#     return eod

# def get_stock_close(interval, date, intervals):
#     ticker = 'spy'
#     # Add SPY close at interval
#     ohlc = pd.read_csv(
#         f"/data/thetadata/stocks/{ticker}/{interval}/{date}.csv.gz", compression='gzip')
#     for index, row in ohlc.iterrows():
#         ms_of_day = row['ms_of_day']
#         close_value = row['close']
#         if ms_of_day in intervals:
#             intervals[ms_of_day]['underlying_close'] = close_value
#         # Optional: else clause to handle non-existent keys
#     return intervals

# def get_option_sd_strikes(date):
#     underlying_ticker = 'spx'
#     options_ticker = 'spxw'
#     eod = get_index_eod(date)
#     price = eod['open'].values[0] * 1000
#     strikes = pd.read_csv(
#         f"/data/thetadata/options/{options_ticker}/strikes/{date}.csv.gz", compression='gzip')
#     atm_strike = strikes.loc[(strikes.sub(Decimal(str(price))).abs().idxmin())]
#     atm_strike = atm_strike.iloc[0].strike
#     print(f"atm strike: {atm_strike}")

#     range_value = 50 * 1000
#     strikes_within_range = strikes[(
#         strikes['strike'] >= atm_strike - range_value) & (strikes['strike'] <= atm_strike + range_value)]
#     return strikes_within_range


# def get_option_quotes(date):
#     ticker = 'spxw'
#     interval = '1s'
#     df = pd.read_csv(
#         f"/data/thetadata/options/{ticker}/0dte/{interval}/{date}.csv.gz", compression='gzip')

#     # Add columns
#     df['mid'] = round((df['bid'] + df['ask']) / 2, 4)
#     # df['vbid'] = round((df['bid'] * df['bid_size']) /2, 4)
#     # df['vask'] =  round((df['ask'] * df['ask_size']) /2, 4)

#     # Drop columns
#     df.drop(columns=['expiration', 'root', 'bid_exchange', 'bid_condition',
#             'ask_exchange', 'ask_condition', 'date'], inplace=True)
#     df.drop(columns=['bid', 'bid_size', 'ask', 'ask_size'], inplace=True)
#     df.drop(columns=['Unnamed: 0'], inplace=True)

#     df.drop_duplicates(inplace=True)

#     # Check each interval has 2 quotes
#     start_time = 34200000  # 09:30 in milliseconds
#     end_time = 57600000    # 16:00 in milliseconds
#     for ms_of_day in range(start_time, end_time + 1, 1000):
#         rows = df[df['ms_of_day'] == ms_of_day]
#         if len(rows) != 42:
#             print(f"{ms_of_day} only has {len(rows)} quotes")

#     # Pivot bid/ask from separate rows to columns
#     pivot_df = df.pivot_table(index=['ms_of_day', 'strike'], columns='right', values='mid', aggfunc='first')
#     pivot_df.columns = ['call_mid', 'put_mid'] # Rename the columns
#     pivot_df = pivot_df.reset_index() # Reset the index
#     pivot_df['time_of_day'] = pd.to_datetime(pivot_df['ms_of_day'], unit='ms').dt.time
#     pivot_df

#     print(f"Number of quotes loaded: {len(pivot_df)}")
#     return pivot_df


# def interval_option_chains(quotes):
#     intervals= {}
#     grouped= quotes.groupby(['ms_of_day', 'right'])
#     for (interval, right), group in grouped:
#         chain= group[['strike', 'mid', 'vbid', 'vask']].set_index('strike').to_dict('index')
#         if interval not in intervals:
#             intervals[interval]= {'P': {}, 'C': {}}
#         intervals[interval][right]= chain
#     return intervals  # {ms : { P: strikes, C, strikes }}

# def check(intervals):
#    for interval in intervals.keys():
#     if 'underlying_close' not in intervals[interval]:
#         # expect last interval to be missing
#         print(f"underlying_close value is missing for interval {interval}")


# def load(date):
#     strikes= get_option_sd_strikes(date)
#     quotes= get_option_quotes(date, strikes)
#     intervals= interval_option_chains(quotes) # {ms : { P: strikes, C, strikes }}
#     intervals= get_stock_close(date, intervals) # Add SPY close
#     check(intervals)
#     return intervals
