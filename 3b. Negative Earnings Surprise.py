

#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
													Import relevant libraries
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import os
import pandas as pd
import pandasql as ps
import sys

sys.path.append('E:/Dropbox/Python/Custom Modules')
import Antonis_Modules as ak

pd.set_option('display.max_columns', 100,
              'display.width', 1000)

os.chdir(r'E:/Dropbox/Projects/Litigation Reputation/3. Data/2. Processed Data')
wrds_loc = 'G:/wrds data/zip files/202106'


# Main sample
main_df = pd.read_parquet('1. Entropy Sample_20220428.gzip')


# Also add sued firms that are not in the main sample --- they will be used for the drop coverage analysis
sued_df = (pd
           .read_excel('../1. Raw Data/1. Sued_sample_20211104.xlsx', usecols=['MSCAD_ID', 'GVKEY', 'LOSS_END_DATE'])
           .rename(columns={'GVKEY': 'gvkey'})
           .query('MSCAD_ID not in @main_df.MSCAD_ID')
           .assign(Sued_sample=1,
                   Post=1)
           .dropna()
           )

main_df = (pd
           .concat([main_df, sued_df], ignore_index=True)
           )
del sued_df


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
		Identify if there was an earnings announcement during the 10 days leading to CPE or Lowest Return Date for Control Firms
						Also look at whether it's a negative earnings announcement using only ibq
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Load quarterly compustat
compq_df = (pd
            .read_parquet(f'{wrds_loc}/comp_fundq_20210621.gzip',
                          columns=['gvkey', 'indfmt', 'datafmt', 'popsrc', 'consol', 'curcdq', 'rdq', 'datadate', 'ibq', 'fyearq',
                                   'fqtr'])
            # Usual filters
            .query('indfmt == "INDL" and datafmt == "STD" and popsrc == "D" and consol == "C" and curcdq == "USD"')
            .drop_duplicates(subset=['gvkey', 'rdq'])
            # Fix Datadate format
            .assign(rdq=lambda x: pd.to_datetime(x['rdq']),
                    datadate=lambda x: pd.to_datetime(x['datadate']),
                    gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'),
                    fyq=lambda x: (x['fyearq']-1) * 4 + x['fqtr'])
            .filter(['gvkey', 'rdq', 'datadate', 'fyq', 'ibq'])
            .dropna(subset=['gvkey', 'rdq'])
            .assign(date=lambda x: x['rdq'])
            )

# Identify negative new using prior year's same quarter's earnings
compq_df = (compq_df
            .merge(compq_df[['gvkey', 'fyq', 'ibq']].assign(fyq=lambda x: x['fyq'] + 4), on=['gvkey', 'fyq'], how='left',
                   suffixes=['', '_pq4'])
            .assign(NegSurpriseIbq=lambda x: np.where(x['ibq'] < x['ibq_pq4'], 1, 0))
            .assign(NegSurpriseIbq=lambda x: np.where(x[['ibq', 'ibq_pq4']].isnull().max(axis=1), np.nan, x['NegSurpriseIbq']))
            .filter(['gvkey', 'date', 'rdq', 'datadate', 'NegSurpriseIbq'])
            .drop_duplicates()
            )

# Keep only one observation per Case-gvkey; create a panel dataset for the 10 days leading up to and including loss end date (or
# date with lowest return for the control firms) and bring in compustat data if rdq date within that period (rdq date has been renamed
# to date)
panel_df = (main_df
            .query('Post == 1')
            .assign(end_date=lambda x: np.where(x['Sued_sample']==1, x['LOSS_END_DATE'], x['LowestRetDate']),
                    start_date=lambda x: x['end_date']-pd.DateOffset(days=9))
            .filter(['MSCAD_ID', 'gvkey', 'start_date', 'end_date'])
            .reset_index(drop=True)
            .dropna()
            )
panel_df = (ak
            .create_ts(panel_df, 'start_date', 'end_date', 'D')
            .merge(compq_df, on=['gvkey', 'date'], how='inner')
			.sort_values(by=['MSCAD_ID', 'gvkey', 'NegSurpriseIbq'])
			.drop_duplicates(subset=['MSCAD_ID', 'gvkey'], keep='last')
            .filter(['MSCAD_ID', 'date', 'gvkey', 'rdq', 'datadate', 'NegSurpriseIbq'])
            .assign(EA_around_BadNews=1)
            )
del compq_df

print(f'Number of observations with an Earnings Announcement in the 10 days leading to, and including, the day of interest: '
      f'{len(panel_df):,}')


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
													Time to turn to IBES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
 Bring in permno: CRSP - Compustat linktable
'''
linktable_df = (pd
                # Import Linktable and keep relevant flags
                .read_parquet('G:/Wrds data/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                        gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                )
sqlcode = '''
            select a.MSCAD_ID, a.gvkey, b.lpermno as permno
            from panel_df as a left join linktable_df as b
            on a.gvkey = b.gvkey and
                date(b.linkdt) <= date(a.date) and
                date(a.date) <= date(b.linkenddt)
          '''
temp_df = ps.sqldf(sqlcode, locals())
panel_df = (panel_df
            .merge(temp_df, on=['MSCAD_ID', 'gvkey'], how='left')
            # Should not have any duplicates, but in case we do, let's keep the last permno
            .sort_values(by=['MSCAD_ID', 'gvkey', 'permno'])
            .drop_duplicates(subset=['MSCAD_ID', 'gvkey'], keep='last')
            )

del temp_df, linktable_df

print(f'Number of observations with an Earnings Announcement in the 10 days leading to, and including, the day of interest after'
      f'bringing in permno: {len(panel_df):,}')


'''
Bring in identifier for analysts' forecasts
'''
crsp_ibes_df = pd.read_stata('G:/WRDS data/ibes_crsp_20190705.dta')

panel_df = (panel_df
            .merge(crsp_ibes_df[['permno', 'ibtic']], on='permno', how='left')
            )
del crsp_ibes_df

print(f'Number of observations with an Earnings Announcement in the 10 days leading to, and including, the day of interest after'
      f'bringing in permno and IBES identifier: {len(panel_df):,}')


'''
Load forecasts and keep relevant ones
'''
ibes_df = (pd
           .read_parquet(f'G:/Wrds Data/zip files/202111/ibes_detu_epsus_20211101.gzip')
           .assign(actdats=lambda x: pd.to_datetime(x['actdats']),
                   fpedats=lambda x: pd.to_datetime(x['fpedats']),
                   anndats=lambda x: pd.to_datetime(x['anndats']),
                   revdats=lambda x: pd.to_datetime(x['revdats']))
           .query('fpi in ["6", "7"] and report_curr == "USD"')
           .rename(columns={'ticker': 'ibtic'})
           )
rel_ibes_df = (panel_df
               .filter(['MSCAD_ID', 'ibtic', 'permno', 'date', 'datadate'])
               .assign(end_date=lambda x: x['date'],
                       start_date=lambda x: x['date'] - pd.DateOffset(days=89))
               .drop(['date'], axis=1)
               .rename(columns={'datadate': 'fpedats'})
               )
rel_ibes_df = (ak
               .create_ts(rel_ibes_df, 'start_date', 'end_date', 'D')
			   .rename(columns={'date': 'revdats'})
               .merge(ibes_df, on=['ibtic', 'revdats', 'fpedats'], how='left')
			   # The vast majority are in D, so let's keep those
               .query('pdf == "D" and measure == "EPS"')
			   # Keep latest
               .sort_values(by=['MSCAD_ID', 'estimator', 'analys', 'revdats'])
               .drop_duplicates(subset=['MSCAD_ID', 'estimator', 'analys'], keep='last')
               .dropna(subset=['value'])
               )

print(f'Number of forecasts for the relevant period issued during the 90 days leading to (and including) earnings announcement: '
      f'{len(rel_ibes_df):,}')

'''
Bring in IBES actual unadjusted
'''
actual_unadj_df = (pd
                   .read_stata('G:/WRDS data/ibes_actu_epsus_20190221.dta')
                   .query('curr_act == "USD" and measure == "EPS" and pdicity == "QTR"')
                   .dropna(subset=['value'])
                   .rename(columns={'value': 'actual',
                                    'ticker': 'ibtic',
                                    'pends': 'fpedats',
                                    'anndats': 'actual_anndats'})
                   .filter(['ibtic', 'actual_anndats', 'fpedats', 'actual'])
                   )
rel_ibes_df = rel_ibes_df.merge(actual_unadj_df, on=['ibtic', 'fpedats'], how='left')

print(f'Number of forecasts for the relevant period issued during the 90 days leading to (and including) earnings announcement after'
      f'merging in actual values: {len(rel_ibes_df):,}')

'''
Bring in CRSP adjustment factors as of the most recent date prior to either the announcement or the forecast date
'''
dsi_df = (pd
          .read_parquet(f'{wrds_loc}/crsp_dsi_20210621.gzip', columns=['date', 'vwretd'])
          .assign(date=lambda x: pd.to_datetime(x['date']))
          )
dsf_df = (pd
          .read_parquet(f'{wrds_loc}/crsp_dsf_20210621.gzip', columns=['date', 'permno', 'cfacshr'])
          .assign(date=lambda x: pd.to_datetime(x['date']))
          )
# Forecast...
temp_df = (pd
           .concat([rel_ibes_df[['MSCAD_ID', 'permno', 'revdats']].assign(date=lambda x: x['revdats'] - pd.DateOffset(days=i))
                    for i in range(0, 10)])
           .drop_duplicates()
		   .merge(dsi_df, on=['date'], how='inner')
           .merge(dsf_df, on=['permno', 'date'], how='inner')
           .sort_values(by=['MSCAD_ID', 'permno', 'revdats', 'date'])
           .drop_duplicates(subset=['MSCAD_ID', 'permno', 'revdats'], keep='last')
           .rename(columns={'cfacshr': 'forecast_cfacshr'})
           .filter(['MSCAD_ID', 'permno', 'revdats', 'forecast_cfacshr'])
           )
rel_ibes_df = rel_ibes_df.merge(temp_df, on=['MSCAD_ID', 'permno', 'revdats'], how='left')

print(f'Number of forecasts for the relevant period issued during the 90 days leading to (and including) earnings announcement after \n'
      f'merging in actual values and adjustment factor for forecast date: {len(rel_ibes_df):,}')


# Actual...
temp_df = (pd
           .concat([rel_ibes_df[['MSCAD_ID', 'permno', 'actual_anndats']].assign(date=lambda x: x['actual_anndats'] - pd.DateOffset(days=i))
                    for i in range(0, 10)])
           .drop_duplicates()
		   .merge(dsi_df, on=['date'], how='inner')
           .merge(dsf_df, on=['permno', 'date'], how='inner')
           .sort_values(by=['MSCAD_ID', 'permno', 'date'])
           .drop_duplicates(subset=['MSCAD_ID', 'permno'], keep='last')
           .rename(columns={'cfacshr': 'actual_cfacshr'})
           .filter(['MSCAD_ID', 'permno', 'actual_anndats', 'actual_cfacshr'])
           )
rel_ibes_df = (rel_ibes_df
               .merge(temp_df, on=['MSCAD_ID', 'permno', 'actual_anndats'], how='left')
               )

print(f'Number of forecasts for the relevant period issued during the 90 days leading to (and including) earnings announcement after \n'
      f'merging in actual values and adjustment factors for forecast and announcement date: {len(rel_ibes_df):,}')


'''
Calculate measures
'''
summ_df = (rel_ibes_df
           .assign(adjusted_value=lambda x: (x['actual_cfacshr'] / x['forecast_cfacshr']) * x['value'])
           .groupby(['MSCAD_ID', 'permno', 'actual'], as_index=False).agg(MedianAdjValue=('adjusted_value', 'median'))
           .assign(NegSurpriseIbes=lambda x: np.where(x['actual'] < x['MedianAdjValue'], 1, 0))
           )

'''
Transfer back to main dataset and export
'''
panel_df = (panel_df
            .merge(summ_df[['MSCAD_ID', 'permno', 'NegSurpriseIbes']], on=['MSCAD_ID', 'permno'], how='left')
            .drop(['date', 'rdq', 'datadate', 'permno', 'ibtic'], axis=1)
            .assign(Post=1)
            )

main_df = (main_df
           .filter(['MSCAD_ID', 'gvkey', 'Post'])
           .merge(panel_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')
           .assign(NegSurpriseIbq=lambda x: x['NegSurpriseIbq'].fillna(0),
                   NegSurpriseIbesIbq=lambda x: x['NegSurpriseIbes'].fillna(x['NegSurpriseIbq']) )
           )

main_df.to_parquet(f'{os.getcwd()}/9e. Earnings Announcement Surprises around Class Period End or Lowest Return Date.gzip',
                   index=False, compression='gzip')
