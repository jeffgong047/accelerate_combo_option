import wrds 
import re 
import yfinance as yf

conn = wrds.Connection()
libraries = conn.list_libraries()
#use re to find the library that contains the word option 
option_libraries = [lib for lib in libraries if re.search(r'option', lib)]
selected_library = "optionm_all"
print(selected_library)
print('tables: ', conn.list_tables(library = selected_library))
print(conn.describe_table(library = selected_library, table = 'opprcd2023'))



#     opt_stats_filename = os.path.join(input_dir, 'option_price_'+st + ".xlsx")
#     opt_df_original = pd.read_excel(opt_stats_filename)
#     print("Original columns in opt_df_original:", opt_df_original.columns)  # Print original columns
#     column_mapping = {
#     'secid': 'Security ID',
#     'date': 'The Date of this Price',
#     'symbol': 'Option Symbol',
#     'symbol_flag': '0=Old option notation 1=New OSI symbol',
#     'exdate': 'Expiration Date of the Option',
#     'last_date': 'Date on Which the Option Last Traded',
#     'cp_flag': 'C=Call, P=Put',
#     'strike_price': 'Strike Price of the Option Times 1000 (strike_price)',
#     'best_bid': 'Highest Closing Bid Across All Exchanges',
#     'best_offer': 'Lowest Closing Ask Across All Exchanges',
#     'volume': 'Volume',
#     'open_interest': 'Open Interest for the Option',
#     'impl_volatility': 'Implied Volatility of the Option',
#     'delta': 'Delta of the Option',
#     'gamma': 'Gamma of the Option',
#     'vega': 'Vega/Kappa of the Option',
#     'theta': 'Theta of the Option',
#     'optionid': 'Unique ID for the Option Contract',
#     'cfadj': 'Cumulative Adjustment Factor',
#     'am_settlement': 'AM Settlement',
#     'contract_size': 'Contract Size',
#     'ss_flag': 'Settlement Flag: 0=Standard, 1=Non-std, E=Non-std Exp. Date',
#     'forward_price': 'Forward Price',
#     'expiry_indicator': 'Expiry Indicator',
#     'root': 'Root of the Option Symbol',
#     'suffix': 'Suffix of the Option Symbol',
#     'div_convention': 'Method of Incorporating Dividends Into the Option Calculations',
#     'exercise_style': '(A)merican, (E)uropean, or ?',
#     'am_set_flag': 'AM Settlement Flag',
#     'cusip': 'CUSIP Number',
#     'ticker': 'Ticker Symbol',
#     'sic': 'SIC Code',
#     'index_flag': 'Index Flag',
#     'exchange_d': 'Exchange Designator',
#     'class': 'Class Designator',
#     'issue_type': 'The Type of Security',
#     'industry_group': 'Industry Group',
#   'issuer': 'Description of the Issuing Company'
#}
  #  opt_df_original.rename(columns=column_mapping, inplace=True)
  #  print("Columns after renaming:", opt_df_original.columns)  # Print columns after renaming


# Input data
security_ids = [101594, 101375, 102265, 102796, 103042, 102968, 211899, 103879, 105329, 105759,
                106276, 106203, 106566, 102936, 103125, 107318, 107616, 107430, 107525, 108161,
                108948, 109224, 111459, 109869, 111469, 135419, 111668, 111861, 111860, 104533]
tickers = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC', 
           'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 
           'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM']


security_ids = [security_ids[0]]
for idx, security_id in enumerate(security_ids):
    sql = f"SELECT * FROM optionm_all.opprcd2023 WHERE secid={security_id}"
    print("Executing SQL query:", sql)  # Debugging line
    df = conn.raw_sql(sql)
    print(len(df))
    name = df.iloc[0]['symbol'].split(' ')[0]
    print(name, tickers[idx])
    # assert name == tickers[idx]
    df.to_excel(f'option_price_{name}_all_year.xlsx')

