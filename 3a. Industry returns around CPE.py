

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
import sys
from tqdm import tqdm

if platform.system() == 'Windows':
	sys.path.append('E:/Dropbox/Python/Custom Modules')
elif platform.system() == 'Darwin':
	sys.path.append('/users/antoniskartapanis/Dropbox/Python/Custom Modules')

import Antonis_Modules as ak

pd.set_option('display.max_columns', 999,
              'display.width', 1000)

if platform.system() == 'Darwin':
	os.chdir('/Users/antoniskartapanis/Dropbox/Projects/Litigation Reputation/3. Data')
elif platform.system() == 'Windows':
	os.chdir(r'E:\Dropbox\Projects\Litigation Reputation\3. Data')
	wrds_loc = 'G:/WRDS data/zip files/202106'


#%%
'''
Main sample and dates of interest
'''

main_df = (pd
           .read_parquet(r'./2. Processed Data/1. Main Sample_20220428.gzip',
                         columns=['MSCAD_ID', 'gvkey', 'Post', 'Sued_sample', 'LOSS_END_DATE', 'LowestRetDate'])
           .query('Post == 1')
           # For sued firms use CPE and for control firm lowest return date
           .assign(end_date=lambda x: np.where(x['Sued_sample']==1, x['LOSS_END_DATE'], x['LowestRetDate']),
                   start_date=lambda x: x['end_date'] - pd.DateOffset(months=1) + pd.DateOffset(days=1))
           .drop(['LOSS_END_DATE', 'LowestRetDate', 'Sued_sample', 'Post'], axis=1)
           .reset_index(drop=True)
           )


#%%
'''
Identify permno and industry
'''


# LinkTable
linktable_df = (pd
                # Import Linktable and keep relevant flags
                .read_parquet('{}/crsp_ccmxpf_linktable_20210621.gzip'.format(wrds_loc))
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt']))
                # Replace missing end date with file date and keep only relevant columns past 1990
                .assign(linkenddt=lambda x: x['linkenddt'].fillna(pd.to_datetime('2021-06-21')))
                .filter(['gvkey', 'lpermno', 'linkdt', 'linkenddt'])
                .query('linkenddt >= "1990-01-01"')
                # If start date prior to 1990, then make it 1990
                .assign(start_temp=pd.to_datetime('1990-01-01'),
                        linkdt=lambda x: np.where(x['linkdt'].dt.year < 1990,
                                                  x['start_temp'],
                                                  x['linkdt']))
                .drop(['start_temp'], axis=1)
                # Take care of permno
                .rename(columns={'lpermno': 'permno'})
                .assign(permno=lambda x: pd.to_numeric(x['permno'], downcast='integer'))
                # Numeric gvkey and keep only those in our sample
                .assign(gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                .query('gvkey in @main_df.gvkey.unique()')
                )

# Create a timeseries and merge
linktable_ts_df = (ak
                   .create_ts_v2(linktable_df, 'linkdt', 'linkenddt', 'D')
                   .rename(columns={'date': 'end_date'})
                   )

main_df = (main_df
           .merge(linktable_ts_df, on=['gvkey', 'end_date'], how='left')
           .sort_values(by=['MSCAD_ID', 'gvkey', 'permno'])
           .drop_duplicates(subset=['MSCAD_ID', 'gvkey'], keep='first')
           )

del linktable_df, linktable_ts_df


# Bring in historical data from CRSP
dsenames_df = (pd
               .read_parquet('{}/crsp_dsenames_20210621.gzip'.format(wrds_loc),
                             columns=['permno', 'namedt', 'nameendt', 'siccd'])
               .assign(namedt=lambda x: pd.to_datetime(x['namedt']),
                       nameendt=lambda x: pd.to_datetime(x['nameendt']),
                       sic2=lambda x: x['siccd']//100)
               .query('nameendt >= "1990-01-01"')
               .drop(['siccd'], axis=1)
               # If start date prior to 1990, then make it 1990
               .assign(start_temp=pd.to_datetime('1990-01-01'),
                       namedt=lambda x: np.where(x['namedt'].dt.year < 1990,
                                                 x['start_temp'],
                                                 x['namedt']))
               .drop(['start_temp'], axis=1)
               )

dsenames_ts_df = (ak
                  .create_ts_v2(dsenames_df, 'namedt', 'nameendt', 'D')
                  .rename(columns={'date': 'end_date'})
                  )

main_df = (main_df
           .merge(dsenames_ts_df, on=['permno', 'end_date'], how='left')
           .assign(sic2=lambda x: pd.to_numeric(x['sic2'], downcast='integer'))
           )

del dsenames_df


#%%
'''
Create a panel and bring in all potential matches
'''


# Load Data
dsi_df = (pd
          .read_parquet('{}/crsp_dsi_20210621.gzip'.format(wrds_loc), columns=['date', 'vwretd'])
          .assign(date=lambda x: pd.to_datetime(x['date']))
          )
dsf_df = (pd
          .read_parquet('{}/crsp_dsf_20210621.gzip'.format(wrds_loc),
                        columns=['permno', 'ret', 'date'])
          .assign(date=lambda x: pd.to_datetime(x['date']))
          .merge(dsi_df, on=['date'], how='left')
          .assign(ar=lambda x: x['ret'] - x['vwretd'])
          .drop(['ret', 'vwretd'], axis=1)
          .dropna(subset=['ar'])
          )
del dsi_df


def get_industry_car(input_df):

	return (ak
	        .create_ts_v2(input_df, 'start_date', 'end_date', 'D')
	        # Bring in returns --- We first bring in all permnos that have the same sic2 on that date and then we bring in returns
	        .merge(dsenames_ts_df.rename(columns={'end_date': 'date'}), on=['date', 'sic2'], how='left', suffixes=['_orig', ''])
	        .merge(dsf_df, on=['permno', 'date'], how='left')
	        # Calculate Cumulative Abnormal Returns for each firm --- Remember we care about MSCAD_ID-gvkey as each MSCAD_ID has multiple
	        # firms given the entropy balance design
	        .sort_values(by=['MSCAD_ID', 'gvkey', 'date', 'permno'])
	        .groupby(['MSCAD_ID', 'gvkey', 'permno'], as_index=False)['ar'].sum()
	        # Now across all industry firms, find average and median
	        .groupby(['MSCAD_ID', 'gvkey'], as_index=False).agg(AvgIndustryCAR=('ar', 'mean'),
	                                                            MedianIndustryCAR=('ar', 'median'))
	        )


ind_ar_df = get_industry_car(main_df)
main_df = main_df.merge(ind_ar_df, on=['MSCAD_ID', 'gvkey'], how='left')

del dsf_df, dsenames_ts_df


#%%
'''
Export
'''

main_df.to_parquet(f'{os.getcwd()}/2. Processed Data/9d. Industry Returns around CPE or Lowest Ret Date.gzip', index=False,
                   compression='gzip')

