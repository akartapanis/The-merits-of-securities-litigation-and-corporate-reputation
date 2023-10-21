
#%%
'''
Import Libraries
'''

from datetime import datetime
import numpy as np
import os
import pandas as pd
import pandasql as ps
import platform
import statsmodels.api as sm
import sys

if platform.system() == 'Windows':
	sys.path.append('E:/Dropbox/Python/Custom Modules')
elif platform.system() == 'Darwin':
	sys.path.append('/users/antoniskartapanis/Dropbox/Python/Custom Modules')

import Antonis_Modules as ak

pd.set_option('display.max_columns', 999,
              'display.width', 1000)

os.chdir(r'E:\Dropbox\Projects\Litigation Reputation\3. Data')
wrds_loc = 'G:/WRDS data'


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
										        INITIAL FILES FOR LITIGATION FILINGS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''
Lawsuits and 10b-5 SEC Accounting
'''

# Lawsuits
sued_df = pd.read_excel('1. Raw Data/1. Sued_sample_20211104.xlsx')

# For those cases with multiple filings, let's find the sum of settlements
# (if the same amount, then don't double count)
sued_df = (sued_df
           .drop_duplicates(subset=['GVKEY', 'FILING_DATE', 'SETTLEMENT_AMOUNT'])
           .assign(SETTLEMENT_AMOUNT=lambda x: x.groupby(['GVKEY', 'FILING_DATE'])['SETTLEMENT_AMOUNT'].transform('sum') )
           .drop_duplicates(subset=['GVKEY', 'FILING_DATE'])
           .query('FILING_DATE <= "2015-12-31"')
           .assign(PreClassStart=lambda x: x['LOSS_START_DATE'] - pd.offsets.DateOffset(months=24),
                   PostDispositionEnd=lambda x: x['DISPOSITION_DATE'] + pd.offsets.DateOffset(months=24) )
           # Fix a case that has settled based on Stanford
           .assign(CASESTATUS=lambda x: np.where(x['MSCAD_ID']==604091, 'Settled', x['CASESTATUS']),
                   SETTLEMENT_AMOUNT=lambda x: np.where(x['MSCAD_ID']==604091, 7750000, x['SETTLEMENT_AMOUNT']) )
           # One case with missing settlement let's fix it
           .assign(SETTLEMENT_AMOUNT=lambda x: np.where(x['MSCAD_ID'] == 50461, 3500000, x['SETTLEMENT_AMOUNT']) )
           .rename(columns={'GVKEY': 'gvkey'})
           # Exclude cases that are not settled or dismissed (i.e. Awards; transferred to MDL etc)
           .query('CASESTATUS in ["Settled", "Dismissed", "Dismissed w/o Prejudice"]')
           )


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
										         Get Post CPE MVE
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

linktable_df = (pd
                # Import Linktable and keep relevant flags
                .read_parquet(f'{wrds_loc}/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                        gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                # Keep relevant ones
                .query('gvkey in @sued_df.gvkey.unique()')
                # Create panel data
                .pipe(ak.create_ts_v2, 'linkdt', 'linkenddt', 'D')
                .filter(['gvkey', 'lpermno', 'date'])
                .rename(columns={'lpermno': 'permno'})
                )

dsf_df = (pd
          .read_parquet(f'{wrds_loc}/zip files/202106/crsp_dsf_20210621.gzip', columns=['date', 'permno', 'shrout', 'prc'])
          .assign(date=lambda x: pd.to_datetime(x['date']),
                  MVE=lambda x: x['shrout'] * x['prc'].abs())
          .merge(linktable_df, on=['permno', 'date'], how='inner')
          # If multiple with the same gvkey on a given day, keep the lowest permno
          .sort_values(by=['gvkey', 'date', 'permno'])
          .drop_duplicates(subset=['gvkey', 'date'], keep='first')
          .filter(['gvkey', 'date', 'MVE'])
          )

# Create a panel with 15 days following class period end or lowest return date for control firms
post_df = (pd
           .concat([sued_df.assign(date=lambda x: x['LOSS_END_DATE'] + pd.DateOffset(days=i)) for i in range(1, 16)])
           .filter(['MSCAD_ID', 'gvkey', 'date'])
           .sort_values(by=['MSCAD_ID', 'gvkey', 'date'])
           .reset_index(drop=True)
           # Bring in dsf data
           .merge(dsf_df, on=['gvkey', 'date'], how='inner')
           # Keep first post MVE
           .sort_values(by=['MSCAD_ID', 'gvkey', 'date'])
           .drop_duplicates(subset=['MSCAD_ID', 'gvkey'], keep='first')
           .filter(['MSCAD_ID', 'gvkey', 'MVE'])
           .rename(columns={'MVE': 'PostMVE'})
           )

# Ensure that they have a Post MVE
sued_df = sued_df.merge(post_df, on=['MSCAD_ID', 'gvkey'], how='inner')


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
										         Create Settlement Outcome Indicators
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


sued_df = (sued_df
           .assign(Settled=lambda x: np.where(x['CASESTATUS'] == 'Settled', 1, 0),
                   Dismissed=lambda x: np.where(x['Settled'] == 0, 1, 0),
                   SettledOver5M=lambda x: np.where( (x['Settled'] == 1) & (x['SETTLEMENT_AMOUNT'] > 5000000), 1, 0),
                   SettledUnder5M=lambda x: np.where( (x['Settled'] == 1) & (x['SETTLEMENT_AMOUNT'] <= 5000000), 1, 0),
                   SettleToMVE=lambda x: x['SETTLEMENT_AMOUNT'] / (x['PostMVE'] * 1000), # settlement in actual $, PostMVE in $K,
                   SettledOver05MVE=lambda x: np.select( [(x['SettleToMVE'] >= 0.005) & (x['Settled'] == 1), x['SettleToMVE'].isnull()],
                                                         [1, np.nan], 0),
                   SettledUnder05MVE=lambda x: np.select( [(x['SettledOver05MVE'] == 0) & (x['Settled'] == 1), x['SettleToMVE'].isnull()],
                                                          [1, np.nan], 0),
                   SettledOver05MVEor50M=lambda x: np.where( ( (x['SETTLEMENT_AMOUNT'] > 50000000) | (x['SettledOver05MVE'] == 1) ) &
                                                             (x['Settled'] == 1), 1, 0),
                   SettledUnder05MVEor50M=lambda x: np.where( (x['SettledOver05MVEor50M'] != 1) & (x['Settled'] == 1), 1, 0)
                   )
           )

sued_df.to_parquet(f'{os.getcwd()}/2. Processed Data/29b. New sued sample composition for ERC test.gzip', index=False, compression='gzip')