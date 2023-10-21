
#%%
'''
Notes from Elizabeth:

*Variable Descriptions:
- FDATE - file date, the date the file was created
- CDATE - create date, the creation date of the file 
- DCN - 
- SEQNUM - 
- FORMTYP - the type of form filed, Form 4 is the most c\ommonly used
- ROLECODE - the insider's role within the firm
- TRANCODE - the code for the type of insider transaction (S - open market sales, P - open market purchases)
- SECDATE - date the filing was received by the SEC *Use this as transaction date
- SIGDATE - date the filing was signed by the insider
- MAINTDATE - last day the record was edited

Limit population to:
•	FormType = 4
		Following Rogers et al. this is primarily where all the insider trading data is and Agrawal and Cooper report there’s a lot of errors in the other data
•	RoleCode = CB, D, DO, H, OD, VC, AC, CC, EC, FC, MC, SC
		Directors 
•	Cleanse= R, H, L, I, C, W, Y 
		Dropping S & A per WRDs recommendation & following Rogers et al. 2015
•	Trancode = S, P
		Open market sales and purchases following Rogers et al. ;

'''

#%%
'''
Import Libraries and data
'''

from datetime import datetime
import gc
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 100,
              'display.width', 1000)

t1_df = (pd
         .read_parquet(r'G:/WRDS data/zip files/202106/tfn_table1_20210621.gzip',
                       columns=['fdate', 'cdate', 'formtype', 'personid',
                                'cusip6', 'cusip2', 'cusipx', 'rolecode1', 'rolecode2', 'rolecode3',
                                'rolecode4', 'trancode', 'trandate', 'tprice', 'shares', 'cleanse'])
         .assign(fdate=lambda x: pd.to_datetime(x['fdate']),
                 cdate=lambda x: pd.to_datetime(x['cdate']),
                 trandate=lambda x: pd.to_datetime(x['trandate']))
         )
categorical_cols = ['formtype', 'rolecode1', 'rolecode2', 'rolecode3', 'rolecode4', 'trancode', 'cleanse']
t1_df[categorical_cols] = t1_df[categorical_cols].astype('category')
del categorical_cols


#%%
'''
Limiting population as specified above
'''

t1_df = t1_df[(t1_df['formtype'] == '4') &
              (t1_df['cleanse'].isin(['S', 'A']) == False) &
              (t1_df['trancode'].isin(['S', 'P']))]


#%%
'''
Director and Officer Indicator and limit sample to them
'''

rolecode = ['rolecode1', 'rolecode2', 'rolecode3', 'rolecode4']
director_codes = ['CB', 'D', 'DO', 'H', 'OD', 'VC', 'AC', 'CC', 'EC', 'FC', 'MC', 'SC']
officer_codes = ['AV', 'CEO', 'CFO', 'CI', 'CO', 'CT', 'EVP', 'O', 'OB', 'OP', 'OS', 'OT', 'OX', 'P', 'S', 'SVP', 'VP']

t1_df = (t1_df
         .assign(Director=lambda x: np.where(x[rolecode].isin(director_codes).any(axis=1), 1, 0),
                 Officer=lambda x: np.where(x[rolecode].isin(officer_codes).any(axis=1), 1, 0))
         .query('Director == 1 | Officer == 1')
         )


#%%
'''
Summarize data at the Person-Firm-Transaction Date level
'''


t1_df = (t1_df
         .assign(# Sale vs purchase
                 Sale_Transactions=lambda x: np.where(x['trancode'] == 'S', 1, 0).astype('int8'),
                 Purchase_Transactions=lambda x: np.where(x['trancode'] == 'P', 1, 0).astype('int8'),
                 # Value of transactions --- Negative for sales
                 Net_Value=lambda x: np.where(x['Sale_Transactions']==1,
                                              x['tprice'] * x['shares'] * -1,
                                              x['tprice'] * x['shares']),
                 # Sale and Purchase value seperately, let's have both positive here
                 Sale_Value=lambda x: np.where(x['Sale_Transactions']==1, x['Net_Value'].abs(), np.nan),
                 Purchase_Value=lambda x: np.where(x['Purchase_Transactions']==1, x['Net_Value'], np.nan),
                 # Number of shares sold
                 NumSharesSold=lambda x: np.where(x['Sale_Transactions']==1, x['shares'], np.nan))
                 # Group
                 .groupby(['personid', 'trandate', 'cusip6'], as_index=False)[['Net_Value', 'Sale_Value', 'Purchase_Value', 'Sale_Transactions',
                                                                               'Purchase_Transactions', 'NumSharesSold']].sum()
                 .assign(unq_identifier=lambda x: range(0, len(x)))
                 )


#%%
'''
Let's Link to permno
'''

# Import dsenames and keep relevant columns
dsenames_df = (pd
               .read_parquet(r'G:/WRDS data/zip files/202106/crsp_dsenames_20210621.gzip',
                             columns=['permno', 'namedt', 'nameendt', 'shrcd', 'ncusip'])
               .assign(namedt=lambda x: pd.to_datetime(x['namedt']),
                       nameendt=lambda x: pd.to_datetime(x['nameendt']),
                       cusip6=lambda x: x['ncusip'].str[:6])
               .drop(['ncusip'], axis=1)
               )

# Connect to t1 --- Keep permno if TRANDATE between namedt and nameendt
t1_permno_df = (t1_df
                .filter(['unq_identifier', 'trandate', 'cusip6'])
                .merge(dsenames_df, on='cusip6', how='inner')
                .query('namedt <= trandate and trandate <= nameendt')
                )

del dsenames_df


#%%
'''
Let's get gvkey
'''

# CRSP - Compustat linktable
linktable_df = (pd
                .read_parquet('G:/WRDS data/zip files/202106/crsp_ccmxpf_linktable_20210621.gzip')
                .assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
                        linkenddt=lambda x: pd.to_datetime(x['linkenddt'].fillna('2021-06-21')),
                        gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
                .query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
                .filter(['gvkey', 'lpermno', 'linkdt', 'linkenddt'])
                .rename(columns={'lpermno':'permno'})
                )

# Link to t1_permno_df --- Keep if TRANDATE between linkperiod
t1_perm_gvkey_df = (t1_permno_df
                    .filter(['unq_identifier', 'trandate', 'permno', 'shrcd'])
                    .merge(linktable_df, on='permno', how='inner')
                    .query('linkdt <= trandate and trandate <= linkenddt')
                    .filter(['unq_identifier', 'permno', 'shrcd', 'gvkey'])
                    )

del linktable_df, t1_permno_df

# Take care of dups
# 1) Even if multiple permno, they may be linked to the same gvkey
#   -- Try keeping the lowest shrcd and permno
t1_perm_gvkey_df = (t1_perm_gvkey_df
                    .sort_values(by=['unq_identifier', 'gvkey', 'shrcd', 'permno'])
                    .drop_duplicates(subset=['unq_identifier', 'gvkey'], keep = 'first')
                    )
# 2) If multiple gvkey, keep the one with the lowest permno
t1_perm_gvkey_df = (t1_perm_gvkey_df
                    .sort_values(by=['unq_identifier', 'permno'])
                    .drop_duplicates(subset=['unq_identifier'], keep = 'first')
                    )


#%%
'''
Transfer to main dataset
'''

t1_df = t1_df.merge(t1_perm_gvkey_df, on=['unq_identifier'], how='inner')
del t1_perm_gvkey_df


#%%
'''
Get number of shares outstanding on that day
'''

dsf_df = (pd
          .read_parquet('G:/WRDS data/zip files/202106/crsp_dsf_20210621.gzip', columns=['permno', 'date', 'shrout'])
          .assign(trandate=lambda x: pd.to_datetime(x['date']))
          .drop(['date'], axis=1)
          )

t1_df = (t1_df
         .merge(dsf_df, on=['permno', 'trandate'], how='left')
         .assign(SharesSoldToShrout=lambda x: x['NumSharesSold'] / (x['shrout'] * 1000))
         .drop(['shrout', 'NumSharesSold'], axis=1)
         )

del dsf_df


#%%
'''
Bring in Compustat Quarterly
'''

compq_df = (pd
            .read_parquet('G:/WRDS data/zip files/202106/comp_fundq_20210621.gzip',
                          columns=['indfmt', 'datafmt', 'popsrc', 'consol', 'curcdq', 'prccq',
                                   'cshoq', 'gvkey', 'datadate', 'fyearq', 'fqtr', 'rdq'])
            .query("indfmt == 'INDL' and datafmt == 'STD' and popsrc == 'D' and consol == 'C' and curcdq == 'USD'")
            .assign(mve=lambda x: x['cshoq'] * x['prccq'],
                    fyq=lambda x: (x['fyearq'] - 1) * 4 + x['fqtr'],
                    datadate=lambda x: pd.to_datetime(x['datadate']),
                    rdq=lambda x: pd.to_datetime(x['rdq']),
                    gvkey=lambda x: pd.to_numeric(x['gvkey'], downcast='integer'))
            # Deal with dups
            .sort_values(by=['gvkey', 'fyq', 'datadate'])
            .drop_duplicates(subset=['gvkey', 'fyq'], keep='last')
            .filter(['gvkey', 'fyearq', 'fyq', 'datadate', 'mve', 'rdq'])
            )

compq_df = (compq_df
            # Identify when the quarter starter, prior quarter RDQ, and initial MVE
            .merge(compq_df[['gvkey', 'fyq', 'mve', 'datadate', 'rdq']].assign(fyq=lambda x: x['fyq'] + 1),
                   on=['gvkey', 'fyq'], how='left', suffixes=['', '_pq'])
            # Also need nq_datadate
            .merge(compq_df[['gvkey', 'fyq', 'datadate']].assign(fyq=lambda x: x['fyq'] - 1),
                   on=['gvkey', 'fyq'], how='left', suffixes=['', '_nq'])
            # If missing pq and nq datadate then just assume 90 day difference
            .assign(datadate_pq=lambda x: x['datadate_pq'].fillna(x['datadate'] - pd.DateOffset(months=3)),
                    datadate_nq=lambda x: x['datadate_nq'].fillna(x['datadate'] + pd.DateOffset(months=3)) )
            # Keep relevant columns and remove empty
            .filter(['gvkey', 'datadate', 'fyearq', 'datadate_pq', 'datadate_nq', 'mve_pq', 'rdq_pq', 'rdq', 'fyq'])
            .dropna(subset=['mve_pq', 'rdq_pq', 'rdq'])
            .rename(columns={'fyearq': 'fyear'})
            )


#%%
'''
Identify quarter of trade --- what we really care about is rdq dates (I don't love this because quarters
get a little bit mixed, but following Billings)
'''

t1_df = (t1_df
         .merge(compq_df, on=['gvkey'], how='inner')
         .query('rdq_pq <= trandate and trandate < rdq')
         # If multiple for whatever reason, keep the latest datadate
         .sort_values(by=['unq_identifier', 'datadate'])
         .drop_duplicates(subset=['unq_identifier'], keep='last')
         )


#%%
'''
Summarize at the quarter level
'''

t1_df['Sale_TV_sc'] = t1_df['Sale_Value'] / (t1_df['mve_pq'] * 1000000)
t1_df = t1_df.replace(-0, 0)

# keep columns of interst and summarize at the quarter level
t1_summ_df = (t1_df
              .groupby(['gvkey', 'fyear', 'datadate', 'datadate_nq', 'rdq', 'fyq'], as_index=False)
              [['Sale_TV_sc', 'SharesSoldToShrout']].sum()
              )


#%%
'''
Calculate standard deviation over tm7 to tm1 and create indicator for abnormal sales
'''

temp_df = (pd
           .concat(t1_summ_df[['gvkey', 'fyear', 'datadate', 'fyq']].assign(fyq=lambda x: x['fyq']-i).copy() for i in range(1, 8))
           # Ensure the quarter exists
           .merge(compq_df[['gvkey', 'fyq']], on=['gvkey', 'fyq'], how='inner')
           # Bring in insiders data and calculate standard deviation - set missing values to 0
           .merge(t1_summ_df[['gvkey', 'fyq', 'SharesSoldToShrout']], on=['gvkey', 'fyq'], how='left')
           .fillna(0)
           .sort_values(by=['gvkey', 'fyear', 'datadate', 'fyq'])
           .groupby(['gvkey', 'fyear', 'datadate'], as_index=False)
           .agg(StdTm7ToTm1ForSharesSoldToShrout=('SharesSoldToShrout', 'std'))
           )

t1_summ_df = (t1_summ_df
              .merge(temp_df, on=['gvkey', 'fyear', 'datadate'], how='left')
              .assign(Abn_Sale_Ind=lambda x: np.where( ( (x['SharesSoldToShrout']/x['StdTm7ToTm1ForSharesSoldToShrout']) >= 3) |
                                                       ( (x['SharesSoldToShrout'] > 0) & (x['StdTm7ToTm1ForSharesSoldToShrout'] == 0)),
                                                       1, 0))
              .filter(['gvkey', 'datadate', 'fyear', 'rdq', 'datadate_nq', 'Sale_TV_sc', 'Abn_Sale_Ind'])
              )

del compq_df

#%%
'''
Export
'''

t1_summ_df.to_parquet(r'E:\Dropbox\Projects\Litigation Reputation\3. Data\2. Processed Data\9. Insider trading.gzip',
                      index=False, compression='gzip')

