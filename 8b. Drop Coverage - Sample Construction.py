
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


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
										INITIAL FILES FOR LITIGATION FILINGS AND REPUTATION DATA
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
           .rename(columns={'GVKEY':'gvkey'})
           )
for i in ['FILING_DATE', 'DISPOSITION_DATE', 'LOSS_START_DATE', 'LOSS_END_DATE']:
	sued_df[f'{i}_p6m'] = sued_df[i] + pd.offsets.DateOffset(months=6)

sued_df = (sued_df
           .assign(PreClassStart=lambda x: x['LOSS_START_DATE'] - pd.offsets.DateOffset(months=24),
                   PostDispositionEnd=lambda x: x['DISPOSITION_DATE'] + pd.offsets.DateOffset(months=24) )
           # Fix a case that has settled based on Stanford
           .assign(CASESTATUS=lambda x: np.where(x['MSCAD_ID']==604091, 'Settled', x['CASESTATUS']),
                   SETTLEMENT_AMOUNT=lambda x: np.where(x['MSCAD_ID']==604091, 7750000, x['SETTLEMENT_AMOUNT']) )
           # One case with missing settlement let's fix it
           .assign(SETTLEMENT_AMOUNT=lambda x: np.where(x['MSCAD_ID'] == 50461, 3500000, x['SETTLEMENT_AMOUNT']) )
           )


#%%
'''
Reputation Data
'''

# Our scrapped data
rep_df = (pd
          .read_csv('1. Raw Data/Scrapped Data/5c. Combined_Sample_with_Gvkey_and_initial_correction.csv')
          .query('Year <= 2016 and gvkey != -1')
          )
rep_df = rep_df[~( (rep_df['Source']=='Fortune') & (rep_df['Year'] == 2014))]
rep_df = rep_df[['gvkey', 'Firm', 'FirmLink', 'Score', 'Quality Of Management', 'Year', 'Industry']]


# PDF Files
rep1_df = (pd
           .read_excel('1. Raw Data/Scrapped Data/7k. PDF files - Linktable.xlsx',
                       usecols=['FortuneDate', 'FirmEdited', 'Score', 'gvkey', 'Industry'], parse_dates=['FortuneDate'])
           .assign(Year=lambda x: x['FortuneDate'].dt.year)
           .drop(['FortuneDate'], axis=1)
           .query('gvkey != -1')
           .rename(columns={'FirmEdited': 'Firm'})
           )

rep_df = (rep_df
          .append(rep1_df, sort=False)
          .sort_values(by=['gvkey', 'Year'])
          .reset_index(drop=True)
          .assign(PriorYear=lambda x: x['Year'] - 1,
                  Score=lambda x: pd.to_numeric(x['Score']).round(2),
                  Industry=lambda x: x['Industry'].str.upper(),
                  IndustryYearMedianScore=lambda x: np.where(x['Industry'].notnull(),
                                                             x.groupby(['Year', 'Industry'])['Score'].transform('median').round(2),
                                                             np.nan),
                  IndustryYearFirmPercentileRank=lambda x: (x
                                                            .groupby(['Year', 'Industry'])
                                                            ['Score'].transform('rank', pct=True, method='max', na_option='keep')
                                                            .round(2)),
                  IndustryYearNumFirms=lambda x: x.groupby(['Year', 'Industry'])['gvkey'].transform('nunique'))
          )
del rep1_df

print(len(rep_df))

# Prior Year's Data
rep_df = (rep_df
          .merge(rep_df[['gvkey', 'Year', 'FirmLink', 'Score']].assign(Year=lambda x: x['Year'] + 1),
                 on=['gvkey', 'Year'], how='left', suffixes=['_t', '_tm1'])
          .assign(Score_chg=lambda x: x['Score_t'] - x['Score_tm1'])
          )

print(len(rep_df))


#%%
'''
Bring in Pub Date
'''

pub_mth_df = (pd
              .read_excel('1. Raw Data/Scrapped Data/6. Month of Publication.xlsx')
              .sort_values(by=['Year'])
              .assign(DateOfPub_tm1=lambda x: x['DateOfPub'].shift(1),
                      DateOfPub_tp1=lambda x: x['DateOfPub'].shift(-1))
              .rename(columns={'DateOfPub': 'DateOfPub_t'})
              .filter(['Year', 'DateOfPub_t', 'DateOfPub_tm1', 'DateOfPub_tp1'])
              )


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
												COMBINE REPUTATION DATA WITH LAWSUITS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
Combine Reputation Data with Lawsuits --- If no reputation score, the rows for the lawsuits will still be in but without scores
'''

# First score published at least 6 months after Filing Date
sqlcode = '''
            select a.*, b.*
            from sued_df as a left join pub_mth_df as b
            on b.DateOfPub_tm1 < a.FILING_DATE_p6m and
               a.FILING_DATE_p6m <= b.DateOfPub_t
          '''
post_df = ps.sqldf(sqlcode, locals())
for i in ['DateOfPub_t', 'DateOfPub_tm1', 'DateOfPub_tp1', 'LOSS_START_DATE', 'LOSS_END_DATE', 'FILING_DATE',
          'LOSS_START_DATE_p6m', 'LOSS_END_DATE_p6m', 'FILING_DATE_p6m', 'DISPOSITION_DATE_p6m', 'DISPOSITION_DATE',
          'PostDispositionEnd', 'PreClassStart']:
	post_df[i] = pd.to_datetime(post_df[i])
post_df = (post_df
           .sort_values(by=['MSCAD_ID', 'DateOfPub_t'])
           # Shouldn't have any, but just in case
           .drop_duplicates(subset='MSCAD_ID', keep='first')
           .assign(Post=1)
           # Bring in reputation scores
           .merge(rep_df, on=['gvkey', 'Year'], how='left')
           )

# Report score published prior to Filing Date
sqlcode = '''
            select a.*, b.*
            from sued_df as a left join pub_mth_df as b
            on b.DateOfPub_t < a.FILING_DATE and
               a.FILING_DATE <= b.DateOfPub_tp1
          '''
pre_df = ps.sqldf(sqlcode, locals())
for i in ['DateOfPub_t', 'DateOfPub_tm1', 'DateOfPub_tp1', 'LOSS_START_DATE', 'LOSS_END_DATE', 'FILING_DATE',
          'LOSS_START_DATE_p6m', 'LOSS_END_DATE_p6m', 'FILING_DATE_p6m', 'DISPOSITION_DATE_p6m', 'DISPOSITION_DATE',
          'PostDispositionEnd', 'PreClassStart']:
	pre_df[i] = pd.to_datetime(pre_df[i])
pre_df = (pre_df
          .sort_values(by=['MSCAD_ID', 'DateOfPub_t'])
          # Shouldn't have any, but just in case
          .drop_duplicates(subset='MSCAD_ID', keep='last')
          .assign(Post=0)
          # Bring in reputation scores
          .merge(rep_df, on=['gvkey', 'Year'], how='left')
          )

def settled_ind(status):

	if status == 'Settled':
		return 1
	# Dismissed; Dismissed w/o Prejudice
	elif status.startswith('Dismissed'):
		return 0
	# Award; Transferred to MDL
	else:
		return np.nan

# Create the list
comb_df = (pre_df
           .append(post_df, sort=False)
           .sort_values(by=['MSCAD_ID', 'DateOfPub_t'])
           # Create indicator dummy for settled cases and also an indicator that this is our sued sample
           .assign(Settled=lambda x: x['CASESTATUS'].apply(lambda y: settled_ind(y)),
                   Sued_sample=1)
           .reset_index(drop=True)
           )


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
									IDENTIFY POTENTIAL MATCHES AND COMBINE WITH SUED
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
Potential Matched Sample
'''


# All Lawsuits
litig_df = (pd
            .read_excel('1. Raw Data/1. Sued_sample_20211104.xlsx', usecols=['GVKEY', 'FILING_DATE'])
            .rename(columns={'GVKEY': 'gvkey'})
            )
non_sued_df = (rep_df
               .assign(unq_id=lambda x: range(0, len(x)))
               .merge(litig_df, on=['gvkey'], how='left')
               # If within current and prior publication then mark to exclude the case
               .merge(pub_mth_df, on=['Year'], how='left')
               .assign(to_drop1=lambda x: np.where( (x['DateOfPub_tm1'] < x['FILING_DATE']) &
                                                    (x['FILING_DATE'] <= x['DateOfPub_t']), 1, 0))
               # If filing less than 6 months away from DateOfPub_tm1, then the effect may still be on DateOfPub_t so exclude
               .assign(to_drop2=lambda x: np.where( (x['FILING_DATE'] <= x['DateOfPub_tm1']) & (x['FILING_DATE'] + pd.DateOffset(months=6) > x['DateOfPub_tm1']), 1, 0))
               # Combine the two indicators
               .assign(to_drop=lambda x: x[['to_drop1', 'to_drop2']].max(axis=1))
               .assign(to_drop=lambda x: x.groupby(['unq_id'])['to_drop'].transform('max'))
               .drop_duplicates(subset=['unq_id'])
               .drop(['to_drop1', 'to_drop2', 'FILING_DATE', 'unq_id', 'DateOfPub_tm1', 'DateOfPub_tp1', 'DateOfPub_t'], axis=1)
               )
del litig_df

# Let's create a timeline of reputation for each firm otherwise we will not be able to capture both periods for control firms if one
# of the periods doesn't have a reputation score
non_sued_exp_df = pd.concat([pub_mth_df.merge(non_sued_df[non_sued_df['gvkey']==i], on=['Year'], how='left').assign(gvkey=i)
                             for i in tqdm(non_sued_df['gvkey'].unique())])

# Bring in all non-sued firms in a given year to sued firms WITH REPUTATION DATA AVAILABLE FOR AT LEAST THE PRE YEAR
# (all potential matches)
all_matches_df = (comb_df
                  .copy()
                  # Keep sued firms with reputation data available in the pre-period
                  .assign(PreDataExists=lambda x: np.where( (x['Post']==0) & (x['Score_t'].notnull()), 1, 0),
                          MaxPreDataExists=lambda x: x.groupby(['MSCAD_ID'])['PreDataExists'].transform('max'))
                  .query('MaxPreDataExists == 1')
                  .drop(['PreDataExists', 'MaxPreDataExists'], axis=1)
                  # Keep only main cols and bring in info for potential matches
                  .filter(['MSCAD_ID', 'Post', 'DateOfPub_t', 'LOSS_START_DATE', 'LOSS_END_DATE', 'FILING_DATE'])
                  .merge(non_sued_exp_df, on=['DateOfPub_t'], how='left')
                  # Ensure available data for pre-year for the potential matches
                  .assign(PreDataExists=lambda x: np.where( (x['Post']==0) & (x['Score_t'].notnull()), 1, 0),
                          MaxPreDataExists=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['PreDataExists'].transform('max'))
                  .query('MaxPreDataExists == 1')
                  .drop(['PreDataExists', 'MaxPreDataExists'], axis=1)
                  .assign(Sued_sample=0)
                  # Drop cases identified as affected by a lawsuit above
                  .query('to_drop != 1')
                  .assign(obs=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['Year'].transform('count'))
                  .query('obs == 2')
                  .drop(['obs', 'to_drop'], axis=1)
                  )

del pre_df, post_df, non_sued_df, non_sued_exp_df


#%%
'''
Exclude potential matches if losing industry coverage or breaking into additional industries
'''

to_exclude_df = (pd
                 .read_parquet('2. Processed Data/28a. Industries losing coverage or breaking into more industries.gzip')
                 .rename(columns={'Year': 'PreYear',
                                  'Industry': 'PreIndustry'})
                 )

all_matches_df = (all_matches_df
                  .sort_values(by=['MSCAD_ID', 'gvkey', 'Post'])
                  .assign(PreYear=lambda x: np.where(x['Post']==0, x['Year'], x['Year'].shift(1)),
                          PostYear=lambda x: np.where(x['Post']==1, x['Year'], x['Year'].shift(-1)),
                          PreIndustry=lambda x: np.where(x['Post']==0, x['Industry'], x['Industry'].shift(1)),)
                  .merge(to_exclude_df, on=['PreYear', 'PostYear', 'PreIndustry'], how='left', indicator=True)
                  .query('_merge != "both"')
                  .drop(['_merge'], axis=1)
                  )


#%%
'''
Combine sued and non-sued samples and create vars of interest
'''

main_df = (comb_df
           .append(all_matches_df, sort=False)
           .reset_index(drop=True)
           # Ensure that it is not the firm itself
           .assign(SuedGvkey=lambda x: np.where(x['Sued_sample']==1, x['gvkey'], np.nan),
                   MaxSuedGvkey=lambda x: x.groupby(['MSCAD_ID'])['SuedGvkey'].transform('max'))
           .query('Sued_sample == 1 or (Sued_sample == 0 and gvkey != MaxSuedGvkey)')
           .drop(['SuedGvkey', 'MaxSuedGvkey'], axis=1)
           # Exclude cases that are not settled or dismissed (i.e. Awards; transferred to MDL etc)
           .query('Settled == Settled or Sued_sample == 0')
           .assign(Dismissed=lambda x: np.where( (x['Settled'] == 0) & (x['CASESTATUS'].notnull()), 1, 0),
                   Settled=lambda x: x['Settled'].fillna(0),
                   SettledOver5M=lambda x: np.where( (x['Settled'] == 1) & (x['SETTLEMENT_AMOUNT'] > 5000000), 1, 0),
                   SettledUnder5M=lambda x: np.where( (x['Settled'] == 1) & (x['SETTLEMENT_AMOUNT'] <= 5000000), 1, 0),
                   Post_Settled=lambda x: x['Settled'] * x['Post'],
                   Post_SettledOver5M=lambda x: x['SettledOver5M'] * x['Post'],
                   Post_SettledUnder5M=lambda x: x['SettledUnder5M'] * x['Post'],
                   Post_Dismissed=lambda x: x['Dismissed'] * x['Post'] )
           )

print(len(main_df))
del all_matches_df, comb_df


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
							    BRING IN PERMNO AROUND CLASS PERIOD END - DROP MISSING ONLY FOR NON-SUED
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# CRSP - Compustat linktable
linktable_df = (pd
                # Import Linktable and keep relevant flags
                .read_parquet('G:/Wrds data/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                        gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                )

# Bring in Permno as well
sqlcode = '''
            select a.MSCAD_ID, a.gvkey, b.lpermno as permno
            from main_df as a left join linktable_df as b
            on a.gvkey = b.gvkey and
                date(b.linkdt) <= date(a.LOSS_END_DATE) and
                date(a.LOSS_END_DATE) <= date(b.linkenddt)
            where a.Post = 0
          '''
temp_df = ps.sqldf(sqlcode, locals())
main_df = (main_df
           .merge(temp_df, on=['MSCAD_ID', 'gvkey'], how='left')
           # Drop only if missing permno and is a potential matched firm (i.e., keep sued firms no matter what)
           .assign(to_drop=lambda x: np.where((x['permno'].isnull()) & (x['Sued_sample']==0), 1, 0))
           .query('to_drop == 0')
           .drop(['to_drop'], axis=1)
           # Should not have any duplicates, but in case we do, let's keep the lowest permno
           .sort_values(by=['MSCAD_ID', 'gvkey', 'Post', 'permno'])
           .drop_duplicates(subset=['MSCAD_ID', 'gvkey', 'Post'], keep='last')
           )

del temp_df


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
								BRING IN COMPUSTAT DATA BOTH TO SUED AS WELL AS TO POTENTIAL MATCHES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
Compustat Data
'''
comp_df = (pd
           .read_parquet(r'G:\WRDS data\zip files\202106\comp_funda_20210621.gzip',
                         columns=['gvkey', 'datadate', 'fyear', 'indfmt', 'datafmt', 'popsrc', 'consol', 'curcd', 'prcc_f',
                                  'sich', 'csho', 'at', 'ceq', 'ib', 'lt', 'xrd', 'capx', 'aqc', 'sppe', 'sale'])
           .query('indfmt == "INDL" and datafmt == "STD" and popsrc == "D" and consol == "C" and curcd == "USD"')
           .assign(datadate=lambda x: pd.to_datetime(x['datadate']),
                   gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'),
                   log_at=lambda x: np.log(x['at']),
                   btm=lambda x: x['ceq'] / (x['prcc_f'] * x['csho']),
                   roa=lambda x: x['ib'] / x['at'],
                   lt_at=lambda x: x['lt'] / x['at'],
                   log_mve=lambda x: np.log(x['prcc_f'] * x['csho']),
                   )
           )
comp_df[['xrd', 'capx', 'aqc', 'sppe']] = comp_df[['xrd', 'capx', 'aqc', 'sppe']].fillna(0)

# Bring in cik from initial funda used as those CIKs are more accurate for the older filings
company_df = (pd
              .read_stata(r'G:\WRDS data\comp_company_20190221.dta', columns=['gvkey', 'cik'])
              .assign(gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
              )
comp_df = comp_df.merge(company_df, on=['gvkey'], how='left')
del company_df

# Identify when the next year ends
comp_df = (comp_df
           .merge(comp_df[['gvkey', 'fyear', 'datadate']].assign(fyear=lambda x: x['fyear']-1).rename(columns={'datadate': 'ny_datadate'}),
                  on=['gvkey', 'fyear'], how='left')
           .assign(ny_datadate=lambda x: x['ny_datadate'].fillna(x['datadate'] + pd.offsets.DateOffset(months=12)))
           )

# Identify prior year's total assets, sales, roa, etc
comp_df = (comp_df
           .merge( (comp_df[['gvkey', 'fyear', 'at', 'sale', 'roa', 'datadate']]
                    .assign(fyear=lambda x: x['fyear']+1).rename(columns={'at': 'py_at',
                                                                          'sale': 'py_sale',
                                                                          'roa': 'py_roa',
                                                                          'datadate': 'py_datadate',})),
                   on=['gvkey', 'fyear'], how='left')
           )

# Investment - per Biddle, Hilary, and Verdi (2009) - and other growth measures:
comp_df = (comp_df
           .assign(investment=lambda x: ( (x['xrd'] + x['capx'] + x['aqc'] - x['sppe']) * 100 ) / x['py_at'],
                   sale_gr=lambda x: (x['sale'] - x['py_sale']) / x['py_sale'])
           )

# Get lead and lags vars needed
comp_df = (comp_df
           # Identify prior year's sales growth
           .merge(comp_df[['gvkey', 'fyear', 'sale_gr']].assign(fyear=lambda x: x['fyear']+1).rename(columns={'sale_gr': 'py_sale_gr'}),
                  on=['gvkey', 'fyear'], how='left')
           )
# Merge with our sample
sqlcode =   '''
            select a.MSCAD_ID, a.gvkey, a.Post, b.fyear, b.at, b.log_at, b.btm, b.roa, b.cik, b.datadate, b.sich, b.lt_at, 
                   b.investment, b.sale_gr, b.py_roa, b.py_sale_gr, b.py_datadate, b.log_mve
            from main_df as a left join comp_df as b
            on a.gvkey = b.gvkey and 
               b.datadate <= a.DateOfPub_t and
               a.DateOfPub_t < b.ny_datadate
            '''
temp_df = ps.sqldf(sqlcode, locals())
main_df = (main_df
           .merge(temp_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')
           .assign(datadate=lambda x: pd.to_datetime(x['datadate']))
           # Keep if either sued or if potential matches have data available both pre and post
           .assign(full_data=lambda x: np.where( x[['at', 'btm', 'roa', 'py_sale_gr', 'py_roa']].isnull().sum(axis=1) == 0, 1, 0),
                   sum_full_data=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['full_data'].transform('sum'))
           .query('Sued_sample == 1 or sum_full_data == 2')
           .drop(['full_data', 'sum_full_data'], axis=1)
           )

print(len(main_df))
del temp_df


#%%
'''
Let's compute stock price crash risk as well - we follow Hsu, Wang, and whipple 2021 JAE
'''

# Create yearly daily dataset
cr_df = (main_df
         .filter(['gvkey', 'fyear', 'datadate', 'py_datadate'])
         .drop_duplicates()
         .assign(py_datadate=lambda x: pd.to_datetime(x['py_datadate']),
                 year_start=lambda x: x['py_datadate'] + pd.DateOffset(days=1))
         .drop(['py_datadate'], axis=1)
         .dropna()
         .reset_index(drop=True)
         )
cr_ts_df = ak.create_ts_v2(cr_df, 'year_start', 'datadate', 'D')
# Keep only if on a Friday as we summarize to Fridays
cr_ts_df = cr_ts_df[cr_ts_df['date'].dt.weekday == 4]

# Bring in permnos over the year
rel_gvkey_set = set(cr_df['gvkey'])
linktable_df = (pd
                # Import Linktable and keep relevant flags
                .read_parquet('G:/Wrds data/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                        gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                .rename(columns={'lpermno': 'permno'})
                # Keep relevant obs
                .query('gvkey in @rel_gvkey_set')
                # If ending prior to 1990, then drop --- If starting prior to that then change
                .query('linkenddt >= "1990-01-01"')
                .assign(temp=pd.to_datetime('1990-01-01'),
                        linkdt=lambda x: np.where(x['linkdt'].dt.year < 1990, x['temp'], x['linkdt']))
                .filter(['gvkey', 'permno', 'linkdt', 'linkenddt'])
                )
linktable_ts_df = ak.create_ts_v2(linktable_df, 'linkdt', 'linkenddt', 'D')
cr_ts_df = cr_ts_df.merge(linktable_ts_df, on=['gvkey', 'date'], how='inner')
del rel_gvkey_set, linktable_ts_df, linktable_df

# Import returns data
dsi_df = (pd
          .read_parquet(r'G:/WRDS data/zip files/202106/crsp_dsi_20210621.gzip', columns=['date', 'vwretd'])
          .assign(date=lambda x: pd.to_datetime(x['date']))
          # Move all days to Fridays
          .assign(date=lambda x: np.where(x['date'].dt.weekday == 4, x['date'], x['date'] + pd.offsets.Week(weekday=4)))
          # Summarize by week
          .dropna()
          .assign(vwretd=lambda x: x['vwretd'] + 1)
          .groupby(['date'], as_index=False).agg(weekly_vwretd_t=('vwretd', 'prod'))
          .assign(weekly_vwretd_t=lambda x: x['weekly_vwretd_t'] - 1)
          .sort_values(by=['date'])
          .assign(tr_week_id=lambda x: range(0, len(x)))
          )
# Bring in leads as lags --- I use unq_id in case a whole week was a non-trading week (i.e., 9/11 etc)
dsi_df = (dsi_df
          # Prior week and the week before
          .merge(dsi_df.assign(tr_week_id=lambda x: x['tr_week_id'] + 1).drop(['date'], axis=1), on=['tr_week_id'], how='left',
                 suffixes=['', 'm1'])
          .merge(dsi_df.assign(tr_week_id=lambda x: x['tr_week_id'] + 2).drop(['date'], axis=1), on=['tr_week_id'], how='left',
                 suffixes=['', 'm2'])
          # Following week and the week after
          .merge(dsi_df.assign(tr_week_id=lambda x: x['tr_week_id'] - 1).drop(['date'], axis=1), on=['tr_week_id'], how='left',
                 suffixes=['', 'p1'])
          .merge(dsi_df.assign(tr_week_id=lambda x: x['tr_week_id'] - 2).drop(['date'], axis=1), on=['tr_week_id'], how='left',
                 suffixes=['', 'p2'])
          )

rel_permno_set = set(cr_ts_df['permno'])
dsf_df = (pd
          .read_parquet(r'G:/WRDS data/zip files/202106/crsp_dsf_20210621.gzip', columns=['date', 'ret', 'permno'])
          .assign(date=lambda x: pd.to_datetime(x['date']),
                  permno=lambda x: pd.to_numeric(x['permno'], downcast='integer'))
          .query('date >= "1990-01-01" and permno in @rel_permno_set')
          # Move all days to Fridays
          .assign(date=lambda x: np.where(x['date'].dt.weekday == 4, x['date'], x['date'] + pd.offsets.Week(weekday=4)))
          # Summarize by week
          .dropna()
          .assign(ret=lambda x: x['ret'] + 1)
          .groupby(['permno', 'date'], as_index=False).agg(cum_ret=('ret', 'prod'))
          .assign(cum_ret=lambda x: x['cum_ret'] - 1)
          )

cr_ts_df = (cr_ts_df
            .merge(dsi_df, on=['date'], how='inner')
            .merge(dsf_df, on=['permno', 'date'], how='inner')
            .sort_values(by=['gvkey', 'fyear', 'date'])
            # Ensure at least 20 obs pre gvkey-fyear group
            .dropna(subset=['cum_ret', 'weekly_vwretd_tm2', 'weekly_vwretd_tm1', 'weekly_vwretd_t', 'weekly_vwretd_tp1',
                            'weekly_vwretd_tp2'])
            .assign(gvkey_fyear=lambda x: x.apply(lambda y: f'{y["gvkey"]} - {y["fyear"]}', axis=1),
                    obs=lambda x: x.groupby(['gvkey', 'fyear'])['cum_ret'].transform('count'))
            .query('obs >= 20')
            .drop(['obs'], axis=1)
            .reset_index(drop=True)
            )

del dsf_df, dsi_df, rel_permno_set

# Run regressions by gvkey-fyear
def regress(input_df):
	# Include -1 to exclude the intercept
	model = sm.formula.ols('cum_ret ~ weekly_vwretd_tm2 + weekly_vwretd_tm1 + weekly_vwretd_t + weekly_vwretd_tp1 + weekly_vwretd_tp2',
	                       data=input_df).fit()
	predict = model.predict(input_df)
	return predict


predictions_df = (cr_ts_df
                  .groupby(['gvkey_fyear']).apply(regress)
                  .reset_index()
                  .drop(['gvkey_fyear'], axis=1)
                  .set_index('level_1') # level_1 correpsonds to iloc of the main dataset
                  .rename(columns={0: 'pred_cum_ret'})
                  )
cr_ts_df = (cr_ts_df
            .merge(predictions_df, left_index=True, right_index=True, how='inner')
            .assign(resid_cum_ret=lambda x: x['cum_ret'] - x['pred_cum_ret'],
                    resid=lambda x: np.log(x['resid_cum_ret'] + 1),
                    resid2=lambda x: x['resid'] * x['resid'],
                    resid3=lambda x: x['resid'] * x['resid'] * x['resid'],
                    mean_resid=lambda x: x.groupby(['gvkey', 'fyear'])['resid'].transform('mean'),
                    resid_d=lambda x: np.where(x['resid'] < x['mean_resid'], x['resid'], np.nan),
                    resid_u=lambda x: np.where(x['resid'] > x['mean_resid'], x['resid'], np.nan))
            .groupby(['gvkey', 'fyear'], as_index=False).agg(n=('resid', 'count'),
                                                             resid3_sum=('resid3', 'sum'),
                                                             resid2_sum=('resid2', 'sum'),
                                                             resid_d_std=('resid_d', 'std'),
                                                             resid_u_std=('resid_u', 'std'))
            .assign(nskew_PrCrRisk=lambda x: -1 * ( x['n'] * ( (x['n']-1)**(3/2) ) * x['resid3_sum'] ) /
                                             ( (x['n']-1) * (x['n']-2) * (x['resid2_sum']**(3/2))),
                    duvol_PrCrRisk=lambda x: np.log(x['resid_d_std']/x['resid_u_std']))
            .filter(['gvkey', 'fyear', 'nskew_PrCrRisk', 'duvol_PrCrRisk'])
            )

main_df = (main_df
           .merge(cr_ts_df, on=['gvkey', 'fyear'], how='left')
           # Keep if either sued or if potential matches have data available both pre and post for stock price crash risk
           .assign(full_data=lambda x: np.where( x['nskew_PrCrRisk'].notnull(), 1, 0),
                   sum_full_data=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['full_data'].transform('sum'))
           .query('Sued_sample == 1 or sum_full_data == 2')
           .drop(['full_data', 'sum_full_data'], axis=1)
           )

print(len(main_df))
del cr_ts_df, cr_df, predictions_df


#%%
'''
Let's also calculate daily return volatility - given data for Stock price crash risk, this variable shouldn't cause any sample attrition
'''

# Create yearly daily dataset
ret_vol_df = (main_df
              .filter(['gvkey', 'fyear', 'datadate', 'py_datadate'])
              .drop_duplicates()
              .assign(py_datadate=lambda x: pd.to_datetime(x['py_datadate']),
                      year_start=lambda x: x['py_datadate'] + pd.DateOffset(days=1))
              .drop(['py_datadate'], axis=1)
              .dropna()
              .reset_index(drop=True)
              )
ret_vol_ts_df = ak.create_ts_v2(ret_vol_df, 'year_start', 'datadate', 'D')

# Bring in permnos over the year
rel_gvkey_set = set(ret_vol_df['gvkey'])
linktable_df = (pd
                # Import Linktable and keep relevant flags
                .read_parquet('G:/Wrds data/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                        gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                .rename(columns={'lpermno': 'permno'})
                # Keep relevant obs
                .query('gvkey in @rel_gvkey_set')
                # If ending prior to 1990, then drop --- If starting prior to that then change
                .query('linkenddt >= "1990-01-01"')
                .assign(temp=pd.to_datetime('1990-01-01'),
                        linkdt=lambda x: np.where(x['linkdt'].dt.year < 1990, x['temp'], x['linkdt']))
                .filter(['gvkey', 'permno', 'linkdt', 'linkenddt'])
                )
linktable_ts_df = ak.create_ts_v2(linktable_df, 'linkdt', 'linkenddt', 'D')
ret_vol_ts_df = ret_vol_ts_df.merge(linktable_ts_df, on=['gvkey', 'date'], how='inner')
del rel_gvkey_set, linktable_ts_df, linktable_df

# Import returns data
rel_permno_set = set(ret_vol_ts_df['permno'])
dsf_df = (pd
          .read_parquet(r'G:/WRDS data/zip files/202106/crsp_dsf_20210621.gzip', columns=['date', 'ret', 'permno'])
          .assign(date=lambda x: pd.to_datetime(x['date']),
                  permno=lambda x: pd.to_numeric(x['permno'], downcast='integer'))
          .query('date >= "1990-01-01" and permno in @rel_permno_set')
          )
ret_vol_ts_df = (ret_vol_ts_df
                 .merge(dsf_df, on=['permno', 'date'], how='inner')
                 .groupby(['gvkey', 'fyear'], as_index=False).agg(StdDailyRet=('ret', 'std'))
                 )

main_df = main_df.merge(ret_vol_ts_df, on=['gvkey', 'fyear'], how='left')

del ret_vol_ts_df, ret_vol_df, dsf_df


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
												Identify relevant matched firm

				We will use KSLitRisk, returns during the one month period leading to CLASS PERIOD END and max damages 
				--- Need to calculate the latter two first
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


print(f'Number of obs after when starting to identify control firms: {len(main_df)}')


#%%
'''
Bring in KS variable
'''

ks_df = (pd
         .read_parquet('./2. Processed Data/19. KS variables - 20210621.gzip')
         .drop(['KS'], axis=1)
         .assign(gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
         )
main_df = main_df.merge(ks_df, on=['gvkey', 'fyear'], how='left')
del ks_df


#%%
'''
Calculate returns during the one month period leading to CLASS PERIOD END
'''

# Import returns data
dsi_df = (pd
          .read_parquet(r'G:/WRDS data/zip files/202106/crsp_dsi_20210621.gzip')
          .assign(date=lambda x: pd.to_datetime(x['date']))
          )
dsf_df = (pd
          .read_parquet(r'G:/WRDS data/zip files/202106/crsp_dsf_20210621.gzip', columns=['date', 'ret', 'permno', 'prc', 'shrout'])
          .assign(date=lambda x: pd.to_datetime(x['date']),
                  permno=lambda x: pd.to_numeric(x['permno'], downcast='integer'),
                  MVE=lambda x: x['prc'].abs() * x['shrout'])
          .drop(['prc', 'shrout'], axis=1)
          .query('date >= "1985-01-01"')
          )
linktable_df = (pd
                # Import Linktable and keep relevant flags
                .read_parquet('G:/Wrds data/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                        gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                )

# Create a panel data for our dataset
ret_df = (main_df
          .copy()
          # Keep one obs per MSCAD_ID - gvkey
          .query('Post == 0')
          .reset_index(drop=True)
          # Keep relevant columns
          .filter(['MSCAD_ID', 'gvkey', 'Sued_sample', 'LOSS_END_DATE'])
          .assign(start_date=lambda x: x['LOSS_END_DATE'] - pd.DateOffset(months=1) + pd.DateOffset(days=1))
          .dropna(subset=['start_date', 'LOSS_END_DATE'])
          )
ret_df = ak.create_ts(ret_df, 'start_date', 'LOSS_END_DATE', 'D')

# Bring in relevant permno on each day
sqlcode = '''
            select a.*, b.lpermno as permno
            from ret_df as a left join linktable_df as b
            on a.gvkey = b.gvkey and
                date(b.linkdt) <= date(a.date) and
                date(a.date) <= date(b.linkenddt)
          '''
ret_df = ps.sqldf(sqlcode, locals())

# Returns over relevant period
ret_df = (ret_df
          .dropna(subset=['permno'])
          # Shouldn't have any, but just in case
          .sort_values(by=['MSCAD_ID', 'gvkey', 'date', 'permno'])
          .drop_duplicates(subset=['MSCAD_ID', 'gvkey', 'date'])
          # Bring in returns
          .assign(date=lambda x: pd.to_datetime(x['date']))
          .merge(dsf_df, on=['permno', 'date'], how='inner')
          .merge(dsi_df[['date', 'vwretd']], on=['date'], how='left')
          .assign(ar=lambda x: x['ret'] - x['vwretd'])
          # Day with the lowest return
          .sort_values(by=['MSCAD_ID', 'gvkey', 'ar', 'date'])
          .assign(LowestRetDateInd=lambda x: np.where(x.groupby(['MSCAD_ID', 'gvkey']).cumcount() == 0, 1, 0))
          # Cumulative Abnormal Return
          .assign(car=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['ar'].transform('sum'))
          # Clean
          .query('LowestRetDateInd == 1')
          .rename(columns={'date': 'LowestRetDate'})
          .filter(['MSCAD_ID', 'gvkey', 'LowestRetDate', 'car'])
          .dropna(subset=['car'])
          )

# Merge back to the matching dataframe
main_df = main_df.merge(ret_df, on=['MSCAD_ID', 'gvkey'], how='left')

del ret_df


# %%
'''
Calculate maximum damages
'''

# For sued firms: LOSS_START_DATE to LOSS_END_DATE
# For control firms: one year period leading to LowestRetDate
dmg_df = (main_df
          # Keep one obs per MSCAD_ID - gvkey
          .query('Post == 0')
          .reset_index(drop=True)
          .copy()
          # Identify relevant period - use lowest return date for control firms
          .assign(
	start_date=lambda x: np.where(x['Sued_sample'] == 1, x['LOSS_START_DATE'], x['LowestRetDate'] - pd.DateOffset(years=1)
	                              + pd.DateOffset(days=1)),
	end_date=lambda x: np.where(x['Sued_sample'] == 1, x['LOSS_END_DATE'], x['LowestRetDate']),
	end_date_copy=lambda x: x['end_date'])
          .filter(['MSCAD_ID', 'gvkey', 'Sued_sample', 'start_date', 'end_date', 'end_date_copy'])
          .dropna()
          )
dmg_df = ak.create_ts_v2(dmg_df, 'start_date', 'end_date', 'D')

# Bring in relevant permno on each day
temp_linktable_df = ak.create_ts_v2( (linktable_df
                                      .drop(['linkprim', 'liid', 'linktype', 'lpermco', 'usedflag'], axis=1)
                                      .rename(columns={'lpermno': 'permno'}) ),
                                    'linkdt', 'linkenddt', 'D')
dmg_df = (dmg_df
          # Bring in permno
          .merge(temp_linktable_df, on=['gvkey', 'date'], how='left')
          # Shouldn't have any dups, but just in case
          .sort_values(by=['MSCAD_ID', 'gvkey', 'date', 'permno'])
          .drop_duplicates(subset=['MSCAD_ID', 'gvkey', 'date'])
          # Max MVE during class period
          .assign(date=lambda x: pd.to_datetime(x['date']))
          .merge(dsf_df, on=['permno', 'date'], how='inner')
          .groupby(['MSCAD_ID', 'gvkey', 'Sued_sample', 'end_date_copy'], as_index=False).agg(Max_MVE=('MVE', 'max'))
          .rename(columns={'end_date_copy': 'end_date'})
          .assign(end_date=lambda x: pd.to_datetime(x['end_date']))
          )

# Create a panel with 15 days following class period end or lowest return date for control firms
post_df = (pd
           .concat([dmg_df.assign(date=lambda x: x['end_date'] + pd.DateOffset(days=i)) for i in range(1, 16)])
           .filter(['MSCAD_ID', 'gvkey', 'date'])
           .sort_values(by=['MSCAD_ID', 'gvkey', 'date'])
           .reset_index(drop=True)
           )
# # Bring in relevant permno on each day
post_df = (post_df
           # Bring in permno
           .merge(temp_linktable_df, on=['gvkey', 'date'], how='left')
           # Shouldn't have any dups, but just in case
           .sort_values(by=['MSCAD_ID', 'gvkey', 'date', 'permno'])
           .drop_duplicates(subset=['MSCAD_ID', 'gvkey', 'date'])
           # Keep first post MVE
           .assign(date=lambda x: pd.to_datetime(x['date']))
           .merge(dsf_df, on=['permno', 'date'], how='inner')
           .sort_values(by=['MSCAD_ID', 'gvkey', 'date'])
           .drop_duplicates(subset=['MSCAD_ID', 'gvkey'], keep='first')
           .filter(['MSCAD_ID', 'gvkey', 'MVE'])
           .rename(columns={'MVE': 'PostMVE'})
           )
dmg_df = (dmg_df
          .merge(post_df, on=['MSCAD_ID', 'gvkey'], how='inner')
          .assign(Max_Damages=lambda x: (x['Max_MVE'] - x['PostMVE']) / 1000,
                  log_max_dmgs=lambda x: np.where(x['Max_Damages'] < 0, 0, np.log(x['Max_Damages'])),
                  Post=1)
          .filter(['MSCAD_ID', 'gvkey', 'Post', 'PostMVE', 'Max_Damages', 'log_max_dmgs'])
          )

main_df = main_df.merge(dmg_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')

del dmg_df, temp_linktable_df


#%%
'''
Time to apply our criteria
'''


match_df = (main_df
            # We care about pre-period KS and then contemporaneous CAR around class period end and Log Max Damages
            .query('Post == 0')
            .filter(['MSCAD_ID', 'gvkey', 'Sued_sample', 'KSLitRisk', 'log_at'])
            # Bring in right CAR and Damages info
            .merge(main_df.query('Post == 1')[['MSCAD_ID', 'gvkey', 'car', 'log_max_dmgs']], on=['MSCAD_ID', 'gvkey'], how='left')
            # Keep obs that have available data for all variables of interest --- Ensure both sued and control firms available
            .dropna(subset=['car', 'KSLitRisk', 'log_max_dmgs'])
            .assign(max_sued=lambda x: x.groupby(['MSCAD_ID'])['Sued_sample'].transform('max'),
                    min_sued=lambda x: x.groupby(['MSCAD_ID'])['Sued_sample'].transform('min') )
            .query('max_sued == 1 and min_sued == 0')
            .drop(['max_sued'], axis=1)
            # Create relevant groups
            .sort_values(by=['MSCAD_ID', 'Sued_sample'])

            # --------------------- Groupings --------------------------

            # Returns
            .assign(car_gr=lambda x: x.groupby(['MSCAD_ID'])['car'].transform(lambda y: pd.qcut(y, 20, labels=range(1, 21))),
                    temp_gr=lambda x: np.where(x['Sued_sample'] == 1, x['car_gr'], np.nan),
                    sued_car_gr=lambda x: x.groupby(['MSCAD_ID'])['temp_gr'].transform('max'))
            .query('sued_car_gr == car_gr')

            # KS
            .assign(KSLitRisk_gr=lambda x: x.groupby(['MSCAD_ID'])['KSLitRisk'].transform(lambda y: pd.qcut(y, 2, labels=range(1, 3))),
                    temp_gr=lambda x: np.where(x['Sued_sample'] == 1, x['KSLitRisk_gr'], np.nan),
                    sued_KSLitRisk_gr=lambda x: x.groupby(['MSCAD_ID'])['temp_gr'].transform('max') )
            .query('sued_KSLitRisk_gr == KSLitRisk_gr')

            )

# Closest log damages difference
match_df = (match_df
            .assign(temp_log_max_dmgs=lambda x: np.where(x['Sued_sample']==1, x['log_max_dmgs'], np.nan),
                    sued_log_max_dmgs=lambda x: x.groupby(['MSCAD_ID'])['temp_log_max_dmgs'].transform('max'),
                    log_max_dmgs_diff=lambda x: np.abs(x['log_max_dmgs'] - x['sued_log_max_dmgs']))
            # Closest KS
            .query('Sued_sample == 0')
            .sort_values(by=['MSCAD_ID', 'log_max_dmgs_diff', 'gvkey'])
            .drop_duplicates(subset=['MSCAD_ID'], keep='first')
            .assign(ContrFirm=1)
            .filter(['MSCAD_ID', 'gvkey', 'ContrFirm'])
            )

# Transfer to main dataframe
main_df = main_df.merge(match_df, on=['MSCAD_ID', 'gvkey'], how='left')

print(f'Number of obs after bringing in control firm flag: {len(main_df)}')


#%%
'''
Done with criteria! => Match_df contains the relevant pairs
'''


main_df = (main_df
           .assign(car=lambda x: np.where(x['Post']==0, 0, x['car']),
                   Max_Damages=lambda x: np.where(x['Post']==0, 0, x['Max_Damages']))
           # Keep matches and sued sample only
           .query('Sued_sample == 1 or ContrFirm == 1')
           # Keep if Sued sample has score in the pre-period
           .assign(to_drop=lambda x: np.where( (x['Sued_sample'] == 1) & (x['Post'] == 0) & (x['Score_t'].isnull()), 1, 0))
           .query('to_drop == 0')
           .assign(obs=lambda x: x.groupby(['MSCAD_ID'])['gvkey'].transform('count'))
           .query('obs == 4')
           .drop(['to_drop', 'obs'], axis=1)
           )

# Create a dummy as to whether part of settled or dismissed
main_df = (main_df
           .merge(main_df[(main_df['Post']==1) & (main_df['CASESTATUS'].notnull())][['MSCAD_ID', 'Settled', 'Dismissed']],
                  on=['MSCAD_ID'], how='left', suffixes=['', '_sample'])
           )


#%%
'''
Settlement to PostMVE ratio
'''

main_df = (main_df
           .assign(SettleToMVE=lambda x: x['SETTLEMENT_AMOUNT'] / (x['PostMVE'] * 1000), # settlement in actual $, PostMVE in $K,
                   SettledOver05MVE=lambda x: np.where( (x['SettleToMVE'] >= 0.005) & (x['Settled'] == 1), 1, 0) )
           .assign(SettledOver05MVE=lambda x: np.where( (x['SettleToMVE'].isnull()) & (x['Settled'] == 1),
                                                        np.nan, x['SettledOver05MVE']))
           .assign(SettledOver05MVE=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['SettledOver05MVE'].transform('max'),
                   SettledUnder05MVE=lambda x: np.where( (x['SettledOver05MVE'] == 0) & (x['Settled'] == 1), 1, 0),
                   Post_SettledOver05MVE=lambda x: x['SettledOver05MVE'] * x['Post'],
                   Post_SettledUnder05MVE=lambda x: x['SettledUnder05MVE'] * x['Post'],
                   # 50M or 05MVE
                   SettledOver05MVEor50M=lambda x: np.where( ( (x['SETTLEMENT_AMOUNT'] > 50000000) | (x['SettledOver05MVE'] == 1) ) &
                                                             (x['Settled'] == 1), 1, 0),
                   Post_SettledOver05MVEor50M=lambda x: x['Post'] * x['SettledOver05MVEor50M'],
                   SettledUnder05MVEor50M=lambda x: np.where( (x['SettledOver05MVEor50M'] != 1) & (x['Settled'] == 1), 1, 0),
                   Post_SettledUnder05MVEor50M=lambda x: x['Post'] * x['SettledUnder05MVEor50M'])
           )


#%%
'''
Keep sample of interest
'''

main_df = (main_df
           .query('FILING_DATE <= "2011-09-18"')
           .query('FILING_DATE >= "1996-01-01"')
           .query('Sued_sample == 1 or ContrFirm == 1')
           .dropna(subset=['roa', 'btm', 'log_at', 'lt_at', 'car']) # 'duvol_PrCrRisk', 'StdDailyRet'
           # Keep if they have observations both pre and post for both matched and treated
           .assign(count=lambda x: x.groupby(['MSCAD_ID'])['gvkey'].transform('count'))
           .query('count == 4')
           .drop(['count'], axis=1)
           )


#%%
'''
Bring in Correct CIKs for some firms
'''


# These are handcollected CIKs using file 7a
cik_df = (pd
          .read_excel(r'2. Processed Data/1b. Collect CIK - processed.xlsx', usecols=['MSCAD_ID', 'gvkey', 'HC_cik'])
          .assign(HC_cik=lambda x: x['HC_cik'].astype('str').str.zfill(10))
          )
main_df = (main_df
           .merge(cik_df, on=['MSCAD_ID', 'gvkey'], how='left')
           .assign(cik=lambda x: np.where(x['HC_cik'].notnull(), x['HC_cik'], x['cik']))
           .drop(['HC_cik'], axis=1)
           )
del cik_df

# Additional CIK corrections retrieved from cases with missing turnover data
cik_df = (pd
          .read_excel(r'1. Raw Data\4b. Cases with missing turnover data_20220428 - processed.xlsx',
                      usecols=['MSCAD_ID', 'gvkey', 'CorrectCIK'])
          .drop_duplicates()
          .dropna()
          .assign(CorrectCIK=lambda x: pd.to_numeric(x['CorrectCIK'], downcast='integer').astype('str').str.zfill(10))
          )
main_df = (main_df
           .merge(cik_df, on=['MSCAD_ID', 'gvkey'], how='left')
           .assign(cik=lambda x: np.where(x['CorrectCIK'].notnull(), x['CorrectCIK'], x['cik']))
           .drop(['CorrectCIK'], axis=1)
           )
del cik_df


#%%
'''
Export
'''

# main_df.to_parquet(f'{os.getcwd()}/2. Processed Data/28b. New sample for drop analysis with matched firms.gzip', compression='gzip',
#                    index=False)

'''
Then simply follow process similar to file 4 to get to the final sample
'''