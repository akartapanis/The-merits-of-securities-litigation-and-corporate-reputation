
#%%
'''
Import relevant libraries
'''

import pandas as pd
from scipy.stats import ttest_1samp
import sys

sys.path.append(r'E:\Dropbox\Python\Custom Modules')
import Antonis_Modules as ak

pd.set_option('display.max_columns', 100,
              'display.width', 1000)

data_loc = 'E:/Dropbox/Projects/Litigation Reputation/3. Data'
wrds_loc = 'G:/WRDS data'


#%%
'''
Load Main Data and bring in the needed vars
'''

main_df = (pd
           .read_parquet(f'{data_loc}/2. Processed Data/29b. New sued sample composition for ERC test.gzip')
           .assign(FilingYear=lambda x: x['FILING_DATE'].dt.year)
           )


# %%
'''
Function to calculate abnormal returns using a user defined window 
'''

# DSI
dsi_df = (pd
          .read_parquet('G:/WRDS data/zip files/202106/crsp_dsi_20210621.gzip', columns=['date', 'vwretd'])
          .assign(tr_day=lambda x: range(0, len(x)),
                  date=lambda x: pd.to_datetime(x['date']))
          )

# CRSP - Compustat linktable
linktable_ts_df = (pd
                   # Import Linktable and keep relevant flags
                   .read_parquet('G:/Wrds data/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                   .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                   .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                           linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                           gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                   .query('gvkey in @main_df.gvkey')
                   .rename(columns={'lpermno': 'permno'})
                   .filter(['gvkey', 'permno', 'linkdt', 'linkenddt'])
                   # We know that it is already only one permno per gvkey per date
                   .pipe(ak.create_ts_v2, 'linkdt', 'linkenddt')
                   )

# DSF
dsf_df = (pd
          .read_parquet('G:/WRDS data/zip files/202106/crsp_dsf_20210621.gzip', columns=['permno', 'date', 'ret', 'prc', 'shrout'])
          .assign(date=lambda x: pd.to_datetime(x['date']),
                  permno=lambda x: pd.to_numeric(x['permno'], downcast='integer'))
          .merge(linktable_ts_df, on=['permno', 'date'], how='inner')
          .drop(['permno'], axis=1)
          )


def get_car(input_df, date_of_int, out_name, pre_window, post_window):

	# Create a temporary dataframe
	temp_df = input_df.copy()

	# Expand by 10 days, bring in trading day number and keep the earliest (This is the day of or if during
	# non-trading days, then the first day after)
	temp_df = (temp_df
	           .assign(start=lambda x: x[date_of_int],
	                   end=lambda x: x[date_of_int] + pd.DateOffset(days=10))
	           .reset_index(drop=True)
	           # Create panel dataframe
	           .pipe(ak.create_ts_v2, 'start', 'end')
	           # Bring in trading days
	           .merge(dsi_df, on='date', how='left')
	           .dropna(subset=['tr_day'])
	           .sort_values(by=['MSCAD_ID', 'gvkey', 'date'])
	           .drop_duplicates(subset=['MSCAD_ID', 'gvkey'], keep='first')
	           )

	# Create panel data that contains relevant trading days based on pre-specified window
	car_df = pd.DataFrame()
	for i in range(pre_window, post_window + 1):
		temp1_df = (temp_df
		            .copy()
		            .filter(['MSCAD_ID', 'gvkey', 'tr_day'])
		            .assign(tr_day=lambda x: x['tr_day'] + i,
		                    rel_day=i)
		            )
		car_df = car_df.append(temp1_df, sort=False)

	car_df = (car_df
	          .merge(dsi_df, on='tr_day', how='left')
	          .merge(dsf_df, on=['gvkey', 'date'], how='inner')
	          .sort_values(by=['MSCAD_ID', 'date'])
	          .assign(ar=lambda x: x['ret'] - x['vwretd'])
	          .groupby(['MSCAD_ID', 'gvkey'], as_index=False).agg(car=('ar', 'sum'))
	          .rename(columns={'car': out_name})
	          )

	input_df = input_df.merge(car_df, on=['MSCAD_ID', 'gvkey'], how='left')

	return input_df



def delay_to_filing_trdays(row):

	cpe = row["LOSS_END_DATE"].date()
	fil_date = row["FILING_DATE"].date()

	temp_df = (dsi_df
	           .query('@cpe <= date and date <= @fil_date')
	           )

	# Deduct one to get the right length (i.e., if on the same day, without deducting one, the length will be 1, but we know it should
	# be zero
	delay = len(temp_df) - 1

	return delay


# %%
'''
Let's first get CAR for different windows
'''


main_df = (main_df
           .filter(['MSCAD_ID', 'gvkey', 'CASESTATUS', 'LOSS_END_DATE', 'FILING_DATE', 'Dismissed', 'SettledUnder05MVEor50M',
                    'SettledOver05MVEor50M' ])
           .reset_index(drop=True)
           .pipe(get_car, 'LOSS_END_DATE', 'CPE_CAR_m10_m2', -10, -2)
           .pipe(get_car, 'LOSS_END_DATE', 'CPE_CAR_m1_p1', -1, 1)
           .pipe(get_car, 'LOSS_END_DATE', 'CPE_CAR_p2_p10', 2, 10)
           .pipe(get_car, 'LOSS_END_DATE', 'CPE_CAR_p11_p60', 11, 60)
           .pipe(get_car, 'FILING_DATE', 'FilDate_CAR_m10_m2', -10, -2)
           .pipe(get_car, 'FILING_DATE', 'FilDate_CAR_m1_p1', -1, 1)
           .pipe(get_car, 'FILING_DATE', 'FilDate_CAR_p2_p10', 2, 10)
           .pipe(get_car, 'FILING_DATE', 'FilDate_CAR_p11_p60', 11, 60)
           .assign(delay_to_filing_trdays=lambda x: x.apply(lambda y: delay_to_filing_trdays(y), axis=1))
           )


'''
Winsorize
'''