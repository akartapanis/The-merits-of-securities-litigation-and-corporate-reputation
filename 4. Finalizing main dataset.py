
#%%
'''
Import Libraries
'''

import numpy as np
import os
import pandas as pd

pd.set_option('display.max_columns', 999,
              'display.width', 1000,
              'display.max_colwidth', 200)

os.chdir('E:/Dropbox/Projects/Litigation Reputation')


# %%
'''
Load and Pre-process Data
'''

main_df = (pd
           .read_parquet(r'./3. Data/2. Processed Data/1. Main Sample_20220428.gzip')
           .assign(log_settlement=lambda x: np.where((x['Settled'] == 1) & (x['Post'] == 1),
                                                     np.log(x['SETTLEMENT_AMOUNT']),
                                                     np.nan),
                   btm=lambda x: np.where(x['btm'] < 0, 0, x['btm']),
                   log_btm=lambda x: np.log(x['btm'] + 1),
                   log_Score_t=lambda x: np.log(x['Score_t']),
                   sic2=lambda x: x['sich'] // 100,
                   gvkey_mscad_id=lambda x: x.apply(lambda y: f'{y["MSCAD_ID"]} - {y["gvkey"]}', axis=1),
                   # Multiply StdDailyRet for presentation purposes (otherwise the DiD for the descriptives doesn't show the proper
                   # change)
                   StdDailyRet=lambda x: x['StdDailyRet'] * 10)
           # Rep change from pre to post
           .sort_values(by=['MSCAD_ID', 'Sued_sample', 'Post'])
           .assign(Pre_log_Score_t=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['log_Score_t'].shift(1),
                   LogScoreChg=lambda x: x['log_Score_t'] - x['Pre_log_Score_t'],
                   LogScorePercChg=lambda x: (x['log_Score_t'] - x['Pre_log_Score_t']) / x['Pre_log_Score_t'],
                   # Use 10.5% decrease as the cut-off for large drop dummy (at the population level, only 5% of firms experience this)
                   LargeRepDropDummy=lambda x: np.select([ x['Post']==0,
                                                           (x['Post']==1) & (x['LogScorePercChg'] <= -0.105),
                                                           (x['Post']==1) & (x['LogScorePercChg'] > -0.105) ],
                                                         [np.nan, 1, 0], np.nan)
                   )
           )


#%%
'''
Drop if in the the Pre-period may be affected from another lawsuit
'''


# Lawsuits
sued_df = (pd
           .read_excel('3. Data/1. Raw Data/1. Sued_sample_20211104.xlsx', usecols=['MSCAD_ID', 'GVKEY', 'FILING_DATE'])
           .rename(columns={'GVKEY': 'gvkey'})
           )


temp_df = (main_df
           # Keep pre-period and only cols of interest
           .copy()
           .query('Post == 0')
           .filter(['MSCAD_ID', 'gvkey', 'Sued_sample', 'DateOfPub_t', 'DateOfPub_tm1'])
           # Merge in all lawsuits
           .merge(sued_df, on=['gvkey'], how='inner', suffixes=['', '_all'])
           # Exclude if it's the same case
           .query('MSCAD_ID != MSCAD_ID_all')
           # If within current and prior publication then mark to exclude the case
           .assign(to_drop1=lambda x: np.where( (x['DateOfPub_tm1'] < x['FILING_DATE']) & (x['FILING_DATE'] <= x['DateOfPub_t']), 1, 0))
           # If filing less than 6 months away from DateOfPub_tm1, then the effect may still be on DateOfPub_t so exclude
           .assign(to_drop2=lambda x: np.where( (x['FILING_DATE'] <= x['DateOfPub_tm1']) & (x['FILING_DATE'] + pd.DateOffset(months=6) > x['DateOfPub_tm1']), 1, 0))
           # Combine the two indicators
           .assign(to_drop=lambda x: x[['to_drop1', 'to_drop2']].max(axis=1))
           # Filter
           .filter(['MSCAD_ID', 'to_drop'])
           .query('to_drop == 1')
           .drop_duplicates()
           )

main_df = main_df[~main_df['MSCAD_ID'].isin(temp_df['MSCAD_ID'])]

del sued_df


# %%
'''
Identify firms with multiple lawsuits affecting the same pre and post year (keep settled if both)
'''

temp_df = (main_df
           .query('Sued_sample == 1')
           .assign(PrePubDate=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['DateOfPub_t'].transform('min'),
                   PostPubDate=lambda x: x.groupby(['MSCAD_ID', 'gvkey'])['DateOfPub_t'].transform('max'))
           .sort_values(by=['gvkey', 'PrePubDate', 'PostPubDate', 'Settled', 'MSCAD_ID'])
           .drop_duplicates(subset=['gvkey', 'PrePubDate', 'PostPubDate'], keep='last')
           )
main_df = main_df[main_df['MSCAD_ID'].isin(temp_df['MSCAD_ID'].unique())]

del temp_df


# %%
'''
Insider Trading
'''

# Load data
ins_trad_df = (pd
               .read_parquet('./3. Data/2. Processed Data/9. Insider trading.gzip')
               .assign(gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
               .rename(columns={'datadate_nq': 'nq_datadate'})
               )

# Post Period: Keep if lated rdq <= DateOfInterest <= nq_datadate
# Notes: It's ok for rdq == DateOfInterest given that in the insider trading code file we keep observations
#        between pq_rdq <= Trandate < rdq
#        It's also ok for nq_datadate == DateOfInterest as we want the quarter with the latest EA already
#        announced prior to the DateOfInterest.
main_df['DateOfInterest'] = np.where(main_df['CASESTATUS'].isnull(),
                                     main_df['LowestRetDate'],
                                     main_df['LOSS_END_DATE'])
temp_df = pd.merge(main_df[['MSCAD_ID', 'gvkey', 'Post', 'DateOfInterest', 'datadate']],
                   ins_trad_df, on=['gvkey'], how='left', suffixes=('_orig', '_ins_tr'))
temp1_df = temp_df[(temp_df['rdq'] <= temp_df['DateOfInterest']) &
                   (temp_df['DateOfInterest'] <= temp_df['nq_datadate']) &
                   (temp_df['Post'] == 1)]

# Pre-period: Keep the quarter of fiscal year end in the pre period
temp2_df = temp_df[(temp_df['datadate_orig'] == temp_df['datadate_ins_tr']) &
                   (temp_df['Post'] == 0)]

# Append the two
temp_df = temp1_df.append(temp2_df)

# For scaling purpose multiply by 100
temp_df['Sale_TV_sc'] = temp_df['Sale_TV_sc'] * 100

# Merge back
main_df = pd.merge(main_df, temp_df.drop(['DateOfInterest', 'datadate_orig', 'fyear', 'datadate_ins_tr', 'nq_datadate', 'rdq'],
                                         axis=1),
                   on=['MSCAD_ID', 'gvkey', 'Post'], how='left')
main_df[['Abn_Sale_Ind', 'Sale_TV_sc']] = main_df[['Abn_Sale_Ind', 'Sale_TV_sc']].fillna(0)

# Clean data
del temp_df, temp1_df, temp2_df, ins_trad_df


#%%
'''
Negative earnings surprises around CPE
'''

earn_surpr_df = pd.read_parquet('./3. Data/2. Processed Data/9e. Earnings Announcement Surprises around Class Period End or Lowest Return Date.gzip',
                                columns=['MSCAD_ID', 'gvkey', 'Post', 'NegSurpriseIbesIbq'])
main_df = (main_df
           .merge(earn_surpr_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')
           .assign(NegSurpriseIbesIbq=lambda x: np.where(x['Post']==0, 0, x['NegSurpriseIbesIbq']))
           )
del earn_surpr_df


# %%
'''
Number of Articles (using Class Period End or Lowest Return Date)
----- Need to be able to retrieve the files from Factiva
'''

# Class Period End window
pubs_t_df = (pd
             .read_parquet(r'./3. Data/2. Processed Data/5a1. Class_Period_or_Lowest_Ret_t - 20220428 - Count File.gzip',
                           columns=['MSCAD_ID', 'gvkey', 'All'])
             .assign(Pubs_ClassPeriodEnd=lambda x: pd.to_numeric(x['All'].astype('str').str.replace(',', '', regex=False),
                                                                 downcast='integer', ),
                     Post=1)
             .drop(['All'], axis=1)
             )

main_df = (main_df
           .merge(pubs_t_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')
           .assign(Pubs_ClassPeriodEnd=lambda x: x['Pubs_ClassPeriodEnd'].fillna(0),
                   log_Pubs_ClassPeriodEnd=lambda x: np.log(x['Pubs_ClassPeriodEnd'] + 1))
           )

del pubs_t_df

# Bring in tone as well of these articles as well as those of the prior year
tone_df = pd.read_parquet('./3. Data/2. Processed Data/5c. Factiva Class Period sentiment.gzip')
tone_t_df = (tone_df
             .copy()
             .filter(['MSCAD_ID', 'gvkey', 'AvgTxtLMSentV2_t'])
             .rename(columns={'AvgTxtLMSentV2_t': 'AvgTxtPubSentV2'})
             .assign(Post=1)
             )
tone_tm1_df = (tone_df
               .copy()
               .filter(['MSCAD_ID', 'gvkey' 'AvgTxtLMSentV2_tm1'])
               .rename(columns={'AvgTxtLMSentV2_tm1': 'AvgTxtPubSentV2'})
               .assign(Post=0)
               )
tone_df = tone_t_df.append(tone_tm1_df)

main_df = main_df.merge(tone_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')

del tone_df, tone_tm1_df, tone_t_df


#%%
'''
Number of Articles calendar year
----- Need to be able to retrieve the files from Factiva
'''

# Class Period End Year

pubs_t_df = (pd
             .read_parquet(r'./3. Data/2. Processed Data/6a. Calendar_Year_Pubs_t_20220428.gzip',
                           columns=['MSCAD_ID', 'gvkey', 'All'])
             .assign(Pubs_CalYear=lambda x: pd.to_numeric(x['All'].astype('str').str.replace(',', '', regex=False),
                                                          downcast='integer'),
                     Post=1)
             .drop(['All'], axis=1)
             )

pubs_tm1_df = (pd
               .read_parquet(r'./3. Data/2. Processed Data/6b. Calendar_Year_Pubs_tm1_20220428.gzip',
                             columns=['MSCAD_ID', 'gvkey', 'All'])
               .assign(Pubs_CalYear=lambda x: pd.to_numeric(x['All'].astype('str').str.replace(',', '', regex=False),
                                                            downcast='integer'),
                       Post=0)
               .drop(['All'], axis=1)
               )

pubs_df = (pubs_t_df
           .append(pubs_tm1_df, sort=False)
           .reset_index(drop=True)
           )

main_df = (main_df
           .merge(pubs_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')
           .assign(log_Pubs_CalYear=lambda x: np.log(x['Pubs_CalYear'] + 1))
           )

del pubs_t_df, pubs_tm1_df, pubs_df


#%%
'''
Industry (2 digit) 1-month CAR leading to CPE or Lowest Return Date
'''

ind_df = (pd
          .read_parquet('3. Data/2. Processed Data/9d. Industry Returns around CPE or Lowest Ret Date.gzip',
                        columns=['MSCAD_ID', 'gvkey', 'AvgIndustryCAR'])
          .assign(Post=1)
          )
main_df = (main_df
           .merge(ind_df, on=['MSCAD_ID', 'gvkey', 'Post'], how='left')
           .assign(AvgIndustryCAR=lambda x: x['AvgIndustryCAR'].fillna(0))
           )

del ind_df


# %%
'''
Firms in final sample
'''

main_df[['MSCAD_ID', 'gvkey']].drop_duplicates().to_excel(r'3. Data/2. Processed Data/13. Cases in final sample_20220428.xlsx',
                                                          index=False)


#%%
'''
Winsorize
'''
