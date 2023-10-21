
#%%
'''
Import relevant libraries
'''

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression
import sys

sys.path.append(r'E:\Dropbox\Python\Custom Modules')
import Antonis_Modules as ak

pd.set_option('display.max_columns', 100,
              'display.width', 1000)

wrds_loc = 'G:/WRDS data'

os.chdir('E:/Dropbox/Projects/Litigation Reputation')


#%%
'''
Import Data
'''

# Main sample
sample_df = (pd
             .read_parquet(r'./3. Data/2. Processed Data/1. Main Sample_20220428.gzip')
			 .sort_values(by=['MSCAD_ID', 'gvkey', 'Post'])
             .assign(CASESTATUS=lambda x: np.where(x['CASESTATUS'] == 'Dismissed w/o Prejudice', 'Dismissed', x['CASESTATUS']))
             .assign(CASESTATUS=lambda x: x['CASESTATUS'].fillna('Control'))
             )

# Cases in final sample
in_final_df = pd.read_excel(r'3. Data/2. Processed Data/13. Cases in final sample_20220428.xlsx')
sample_df = sample_df.merge(in_final_df, on=['MSCAD_ID', 'gvkey'], how='inner')

# Keep only one instance
sample_df = sample_df[sample_df['Post'] == 0]

# DSI
dsi_df = (pd
          .read_parquet(f'{wrds_loc}/zip files/202106/crsp_dsi_20210621.gzip', columns=['date', 'vwretd'])
          .assign(tr_day=lambda x: range(0, len(x)),
                  date=lambda x: pd.to_datetime(x['date']))
          )

# DSF
rel_permno_set = set(sample_df['permno'].unique())
dsf_df = (pd
          .read_parquet(f'{wrds_loc}/zip files/202106/crsp_dsf_20210621.gzip', columns=['permno', 'date', 'ret', 'shrout', 'prc'])
          .assign(date=lambda x: pd.to_datetime(x['date']),
                  MVE=lambda x: (x['prc'].abs() * x['shrout']) / 1000) # Divide by 1000 to get it in M
          .query('date >= "1991-01-01" and permno in @rel_permno_set')
          .drop(['prc', 'shrout'], axis=1)
          )
del rel_permno_set


#%%
'''
Function to calculate abnormal returns on a daily basis around window of interest (centered on filing day or CPE)
'''

def get_daily_ar(input_df, date_of_int, out_name, post_window):

	# Create a temporary dataframe
	temp_df = input_df.copy()

	# Expand by 10 days, bring in trading day number and keep the earliest (This is the day of or if during
	# non-trading days, then the first day after)
	temp_df = (temp_df
	           .assign(start=lambda x: x[date_of_int],
	                   end=lambda x: x[date_of_int] + pd.DateOffset(days=10))
	           .reset_index(drop=True)
	           )

	temp_df = ak.create_ts(temp_df, 'start', 'end')
	temp_df = (temp_df
	           .merge(dsi_df, on='date', how='left')
	           .dropna(subset=['tr_day'])
	           .sort_values(by=['MSCAD_ID', 'permno', 'date'])
	           .drop_duplicates(subset=['MSCAD_ID', 'permno'], keep='first')
	           )

	# Daiy returns for window of interest
	car_df = pd.DataFrame()
	for i in range(-post_window, post_window + 1):
		temp1_df = (temp_df
					.copy()
					.filter(['MSCAD_ID', 'permno', 'tr_day'])
		            .assign(tr_day=lambda x: x['tr_day'] + i,
                            rel_day=i)
		            )
		car_df = car_df.append(temp1_df, sort=False)

	car_df = (car_df
	          .merge(dsi_df, on='tr_day', how='left')
	          .merge(dsf_df, on=['permno', 'date'], how='left')
	          .sort_values(by=['MSCAD_ID', 'date'])
	          .assign(ar=lambda x: x['ret'] - x['vwretd'],
	                  car=lambda x: x.groupby(['MSCAD_ID', 'permno'])['ar'].cumsum(),
	                  type=out_name)
	          .filter(['MSCAD_ID', 'permno', 'date', 'rel_day', 'type', 'ar', 'car']))

	# Get MVE the day prior to the CAR accumulation period
	temp_df = (temp_df
			   .filter(['MSCAD_ID', 'permno', 'tr_day'])
	           .assign(tr_day=lambda x: x['tr_day'] - post_window - 1)
	           .merge(dsi_df, on='tr_day', how='left')
	           .merge(dsf_df, on=['permno', 'date'], how='left')
			   .filter(['MSCAD_ID', 'permno', 'MVE'])
	           .rename(columns={'MVE': 'MVE_prior_to_CAR_start'})
	           )
	car_df = car_df.merge(temp_df, on=['MSCAD_ID', 'permno'], how='left')

	return car_df


#%%
'''
Get returns for variables of interest
'''

post_window = 2
loss_end_df = get_daily_ar(sample_df[sample_df['CASESTATUS'] != 'Control'], 'LOSS_END_DATE', 'Loss End Date', post_window)
filing_df = get_daily_ar(sample_df[sample_df['CASESTATUS'] != 'Control'], 'FILING_DATE', 'Filing Date', post_window)

filing_df = (filing_df
             .merge(sample_df.query('CASESTATUS != "Control"').filter(['MSCAD_ID', 'LOSS_END_DATE', 'FILING_DATE']),
                    on=['MSCAD_ID'], how='left')
             )

car_df = loss_end_df.append(filing_df, sort=False)

# Let's get in case status
car_df = pd.merge(car_df, sample_df[sample_df['CASESTATUS'] != 'Control'][['MSCAD_ID', 'CASESTATUS', 'Sued_sample']], on='MSCAD_ID',
                  how='left')
car_df = car_df.rename(columns={'CASESTATUS': 'Classification'})


car_df[car_df['car'].notnull()].groupby('rel_day')['MSCAD_ID'].count()


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Regression Analyses
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# Main sample
sample_df = (pd
             .read_parquet(r'./3. Data/2. Processed Data/1. Main Sample_20220428.gzip')
             .assign(CASESTATUS=lambda x: np.where(x['CASESTATUS'] == 'Dismissed w/o Prejudice', 'Dismissed', x['CASESTATUS']))
             .assign(CASESTATUS=lambda x: x['CASESTATUS'].fillna('Control'),
                     PostScore=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['Score_t'].shift(-1),
                     PercChgLogScore=lambda x: (np.log(x['PostScore']) - np.log(x['Score_t'])) / np.log(x['Score_t']),
                     log_max_dmgs=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['log_max_dmgs'].shift(-1),
                     FilingYear=lambda x: x['FILING_DATE'].dt.year,
                     LogClassPeriod=lambda x: np.log(  ( (x['LOSS_END_DATE'] - x['LOSS_START_DATE']) / np.timedelta64(1, 'D') ) + 1) )
             .query('Sued_sample == 1 and Post == 0')
             )

# Summarize Class Period End Returns
cpe_sum_df = (car_df
			  # Keep in only returns around filing day and summarize at the case level (i.e., overall CAR)
              .query('type == "Loss End Date"')
              .groupby(['MSCAD_ID', 'permno'], as_index=False).agg(CPE_car=('ar', 'sum'))
              )

# Compustat to get data just prior to filing date
comp_df = (pd
           .read_parquet(f'{wrds_loc}/zip files/202106/comp_funda_20210621.gzip',
                         columns=['indfmt', 'datafmt', 'popsrc', 'consol', 'curcd', 'prcc_f', 'csho', 'gvkey', 'datadate', 'fyear',
                                  'at', 'ceq'])
           .query('indfmt == "INDL" and datafmt == "STD" and popsrc == "D" and consol == "C" and curcd == "USD"')
           .assign(datadate=lambda x: pd.to_datetime(x['datadate']),
                   gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'),
                   log_at=lambda x: np.log(x['at']),
                   btm=lambda x: x['ceq'] / (x['prcc_f'] * x['csho']) )
            .assign(btm=lambda x: np.where(x['btm'] < 0, 0, x['btm']),
                    log_btm=lambda x: np.log(x['btm'] + 1))
           .filter(['gvkey', 'fyear', 'datadate', 'log_at', 'log_btm'])
           )
comp_df = (comp_df
           .merge(comp_df[['gvkey', 'fyear', 'datadate']].assign(fyear=lambda x: x['fyear'] - 1), on=['gvkey', 'fyear'],
                  how='left', suffixes=['', '_ny'])
           .assign(datadate_ny=lambda x: x['datadate_ny'].fillna(x['datadate'] + pd.DateOffset(years=1)))
           )
comp1_df = (sample_df
			.filter(['MSCAD_ID', 'gvkey', 'FILING_DATE'])
            .merge(comp_df, on=['gvkey'], how='left')
            .query('datadate <= FILING_DATE and FILING_DATE < datadate_ny')
            .drop(['datadate_ny', 'fyear'], axis=1)
            )

# Import analyst data
crsp_ibes_df = pd.read_stata(f'{wrds_loc}/ibes_crsp_20190705.dta', columns=['ibtic', 'permno'])
ibes_df = (pd
           .read_parquet(f'{wrds_loc}/zip files/202104/ibes_statsum_epsus_20210407.gzip')
           # Keep quarterly issued this quarter; Announcement cannot come before period end; Forecast needs to come
           # before Announcement date
           .dropna(subset=['fpedats', 'anndats_act', 'statpers'])
           .assign(fpedats=lambda x: pd.to_datetime(x['fpedats']),
                   anndats_act=lambda x: pd.to_datetime(x['anndats_act']),
                   statpers=lambda x: pd.to_datetime(x['statpers']))
           .query('fpi in ["6", "7"] and fpedats < anndats_act and statpers < anndats_act and curcode == "USD"')
           .dropna(subset=['actual', 'medest'])
           # Keep the latest per period
           .sort_values(by=['ticker', 'fpedats', 'statpers'])
           .drop_duplicates(subset=['ticker', 'fpedats'], keep='last')
           # Keep vars of interest
           .filter(['ticker', 'fpedats', 'numest'])
           .rename(columns={'ticker': 'ibtic',
                            'fpedats': 'datadate'})
           .merge(crsp_ibes_df, on=['ibtic'], how='inner')
           .drop(['ibtic'], axis=1)
           )

# Now finally filing day returns and bring in all relevant data
fil_sum_df = (car_df
			  # Keep in only returns around filing day and summarize at the case level (i.e., overall CAR)
              .query('type == "Filing Date"')
              .groupby(['MSCAD_ID', 'permno', 'MVE_prior_to_CAR_start'], as_index=False).agg(car=('ar', 'sum'))
			  # Bring in additional vars (Filing Year, Max Damages etc.)
			  .merge(sample_df[['MSCAD_ID', 'gvkey', 'CASESTATUS', 'log_max_dmgs', 'FilingYear', 'PercChgLogScore',
                                'LogClassPeriod']],
                     on=['MSCAD_ID'], how='left')
              .merge(cpe_sum_df, on=['MSCAD_ID', 'permno'], how='left')
              .merge(comp1_df, on=['MSCAD_ID', 'gvkey'], how='left')
			  .merge(ibes_df, on=['permno', 'datadate'], how='left')
			  .assign(numest=lambda x: x['numest'].fillna(0),
                      log_numest=lambda x: np.log(x['numest'] + 1))
              # Reset index
              .reset_index(drop=True)
              )

'''
Winsorize
'''