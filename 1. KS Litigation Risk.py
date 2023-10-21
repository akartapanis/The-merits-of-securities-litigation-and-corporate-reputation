
#%%
'''
Load Libraries
'''

import numpy as np
import pandas as pd
import uuid
import sys

if uuid.getnode() == 190070690681122:
	sys.path.append('/Users/antonis/dropbox/python/custom modules')
	import Antonis_Modules as ak
	wrds_loc = '/Users/Antonis/Dropbox/Projects/WRDS data/zip files/202106'
	main_loc = '/Users/Antonis/Dropbox/Projects/Litigation Reputation'
elif uuid.getnode() in [118933448243835, 113780617493, 963350075783]:
	sys.path.append('E:/dropbox/python/custom modules')
	import Antonis_Modules as ak
	wrds_loc = 'G:/WRDS data/zip files/202106'
	main_loc = 'E:/Dropbox/Projects/Litigation Reputation'


#%%
'''
Compustat
'''

# Load Compustat
comp_df = (pd
			.read_parquet('{}/comp_funda_20210621.gzip'.format(wrds_loc),
						  columns = ['gvkey', 'datadate', 'cik', 'conm', 'indfmt', 'curcd', 'datafmt',
									 'popsrc', 'consol', 'fyear', 'at', 'csho', 'prcc_f', 'sale'])
			# Apply common filters
			.query('indfmt == "INDL" & datafmt =="STD" & popsrc == "D" & consol == "C" & curcd == "USD"')
			# Data post 1990
			.query('fyear >= 1990')
			# Drop dups
			.drop_duplicates(subset=['gvkey', 'fyear'], keep = 'first')
			# Drop temporary columns
			.drop(['indfmt', 'datafmt', 'popsrc', 'consol', 'curcd'], axis = 1)
		   # Convert datadate to datetime
		   .assign(datadate=lambda x: pd.to_datetime(x['datadate']))
			)

# Lag Data
comp_df = (comp_df
			.merge(comp_df[['gvkey','fyear','at','sale','datadate']].assign(fyear = lambda x: x['fyear'] + 1), 
					on = ['gvkey', 'fyear'], how='left', suffixes = ['', '_py'])       
			)
			
#%%
'''
Comp-CRSP Linktable
'''

# LinkTable
linktable_df = (pd
				# Import Linktable and keep relevant flags
				.read_parquet('{}/crsp_ccmxpf_linktable_20210621.gzip'.format(wrds_loc))
				.query("linktype  in ['LU', 'LC', 'LN', 'LS'] and linkprim in ['P', 'C']")
				.assign(linkdt=lambda x: pd.to_datetime(x['linkdt']),
						linkenddt=lambda x: pd.to_datetime(x['linkenddt']))
				# Replace missing end date with file date and keep only relevant columns past 1990
				.assign(linkenddt = lambda x: x['linkenddt'].fillna(pd.to_datetime('2021-06-21')))
				.filter(['gvkey', 'lpermno', 'linkdt', 'linkenddt'])
				.query('linkenddt >= "1990-01-01"')
				# If start date prior to 1990, then make it 1990
				.assign(start_temp = pd.to_datetime('1990-01-01'),
						linkdt = lambda x: np.where(x['linkdt'].dt.year < 1990,
													x['start_temp'],
													x['linkdt']) )
				.drop(['start_temp'], axis=1)
				# Take care of permno
				.rename(columns={'lpermno': 'permno'})
				.assign(permno = lambda x: pd.to_numeric(x['permno'], downcast='integer'))
				)

# Create a timeseries and merge
linktable_ts_df = (ak
                   .create_ts_v2(linktable_df, 'linkdt', 'linkenddt', 'M')
                   .rename(columns={'date': 'datadate'})
                   )

comp_df = (comp_df
			.merge(linktable_ts_df, on=['gvkey', 'datadate'], how='inner')
			.sort_values(by=['gvkey', 'fyear', 'permno'])
			.drop_duplicates(subset=['gvkey', 'fyear'], keep='first')
		)
		
del linktable_df, linktable_ts_df


#%%

# Bring in historical data from CRSP    
msenames_df = (pd
				.read_parquet('{}/crsp_msenames_20210621.gzip'.format(wrds_loc),
							columns = ['permno','namedt','nameendt','siccd','exchcd','ncusip','ticker'])
			    .assign(namedt=lambda x: pd.to_datetime(x['namedt']),
						nameendt=lambda x: pd.to_datetime(x['nameendt']))
				.query('nameendt >= "1990-01-01"')
				# If start date prior to 1990, then make it 1990
				.assign(start_temp = pd.to_datetime('1990-01-01'),
						namedt = lambda x: np.where(x['namedt'].dt.year < 1990,
													x['start_temp'],
													x['namedt']) )
				.drop(['start_temp'], axis = 1)
				)

msenames_ts_df = (ak
                  .create_ts_v2(msenames_df, 'namedt', 'nameendt', 'M')
                  .rename(columns={'date': 'datadate'})
                  )
			
comp_df = pd.merge(comp_df, msenames_ts_df, on=['permno', 'datadate'], how='left')

del msenames_df, msenames_ts_df


#%%
'''
Create vars
'''

comp_df = (comp_df
			.assign(# KS Industries
					FPS = lambda x: np.where( ( (2833<=x["siccd"]) & (x["siccd"]<=2836) ) |
											  ( (8731<=x["siccd"]) & (x["siccd"]<=8734) ) |
											  ( (3570<=x["siccd"]) & (x["siccd"]<=3577) ) |
											  ( (7370<=x["siccd"]) & (x["siccd"]<=7374) ) |
											  ( (3600<=x["siccd"]) & (x["siccd"]<=3674) ) |
											  ( (5200<=x["siccd"]) & (x["siccd"]<=5961) ) ,
											  1,0),
					# BS Industries
                    Biotech=lambda x: np.where( (2833 <= x['siccd']) & (x['siccd'] <= 2836), 1, 0),
                    CompHrdw=lambda x: np.where( (3570 <= x['siccd']) & (x['siccd'] <= 3577), 1, 0),
                    Electronics=lambda x: np.where( (3600 <= x['siccd']) & (x['siccd'] <= 3674), 1, 0),
                    Retail=lambda x: np.where( (5200 <= x['siccd']) & (x['siccd'] <= 5961), 1, 0),
                    CompSoft=lambda x: np.where( (7371 <= x['siccd']) & (x['siccd'] <= 7379), 1, 0),
					sales_gr = lambda x: (x['sale'] - x['sale_py']) / x['at_py'],
					ln_at = lambda x: np.log(x['at']) )
			.drop(['siccd', 'sale', 'sale_py', 'at_py', 'at'], axis = 1)
			)


#%%
'''
Let's deal with returns
'''

# Load Data
msf_df = (pd
		  .read_parquet('{}/crsp_msf_20210621.gzip'.format(wrds_loc),
						columns=['permno', 'ret', 'date', 'vol','cfacshr','shrout', 'prc'])
		  .assign(date=lambda x: pd.to_datetime(x['date']))
		  .assign(date=lambda x: x['date'] + pd.offsets.MonthEnd(0)) # Date refers to last trading day, not month date
          .assign(dollar_turnover=lambda x: x['prc'].abs() * x['vol'])
		  )
msi_df = (pd
		  .read_parquet('{}/crsp_msi_20210621.gzip'.format(wrds_loc), columns=['date', 'vwretd'])
		  .assign(date=lambda x: pd.to_datetime(x['date']))
		  .assign(date=lambda x: x['date'] + pd.offsets.MonthEnd(0))
		  )

# Temp compustat timeseries
temp_df = (comp_df
			.filter(['gvkey', 'fyear', 'permno', 'datadate_py', 'datadate'])
			# Offset day by one so it start in the next year
			.assign(datadate_py = lambda x: x['datadate_py'] + pd.offsets.DateOffset(1))
			.dropna()
		  )
temp_ts_df = ak.create_ts_v2(temp_df, 'datadate_py', 'datadate', 'M')

# Bring in returns
temp_ts_df = (temp_ts_df
				.merge(msf_df, on=['permno', 'date'], how='inner')
				.merge(msi_df, on=['date'], how='left')
			)

shr_temp_df = (temp_ts_df
			   .filter(['gvkey', 'fyear', 'date', 'cfacshr', 'shrout'])
               .sort_values(by=['gvkey','fyear', 'date'])
               .dropna()
               .drop_duplicates(subset=['gvkey', 'fyear'], keep='first')
               .rename(columns={'cfacshr': 'startingcfacshr',
                                'shrout': 'startingshrout'})
               )

# Calculate variables of interest - bhar, car, std_ret, turnover
temp_ts_df = (temp_ts_df
				.assign(ret_adj = lambda x: x['ret'] - x['vwretd'],
						log_ret = lambda x: np.log(1 + x['ret']),
						log_vwretd = lambda x: np.log(1 + x['vwretd']))
				# Beginning of period values for adjusting shares
				.merge(shr_temp_df, on=['gvkey', 'fyear'], how='left')
				# Adjust Volume
				.assign(vol = lambda x: np.where(x['cfacshr'] != x['startingcfacshr'], 
												x['vol']*(x['cfacshr']/x['startingcfacshr']),
												x['vol']))
				# Calculate turnover - actual in tens vs shrout in thousands
				.assign(turnover = lambda x: x['vol'] / (10 * x['startingshrout']),
                        dollar_turnover_sc = lambda x: x['dollar_turnover'] / (x['startingshrout'] * 1000) )
				# Summarize
				.groupby(['gvkey','fyear'], as_index=False)
				.agg({'ret_adj': 'sum', 'log_ret': 'sum', 'log_vwretd': 'sum', 'ret': ['std','skew'],
					  'turnover': 'sum', 'dollar_turnover_sc': 'sum'})
			)

temp_ts_df.columns = temp_ts_df.columns.map(''.join)
temp_ts_df = (temp_ts_df
				.rename(columns={'ret_adjsum':'car','log_retsum':'sum_log_ret', 
								 'log_vwretdsum':'sum_log_vwretd', 
								 'retstd':'std_ret', 'retskew':'ret_skewness',
								 'turnoversum': 'turnover',
                                 'dollar_turnover_scsum': 'dollar_turnover_sc'})
				.assign(bhar = lambda x: np.exp(x['sum_log_ret']) - np.exp(x['sum_log_vwretd']))
				.filter(['gvkey', 'fyear', 'car', 'std_ret', 'bhar',  'ret_skewness', 'turnover', 'dollar_turnover_sc'])
			)

comp_df = pd.merge(comp_df, temp_ts_df, on=['gvkey','fyear'], how='left')
del temp_ts_df, msf_df, msi_df, temp_df, shr_temp_df


#%%
'''
Get in lag data and calculate probabilities
'''

comp_df = (comp_df
           .merge( (comp_df
                    .filter(['gvkey', 'fyear', 'ln_at', 'sales_gr', 'car', 'ret_skewness', 'std_ret', 'turnover',
                             'bhar', 'dollar_turnover_sc', 'Biotech', 'CompHrdw', 'Electronics', 'Retail', 'CompSoft'])
                    .assign(fyear = lambda x: x['fyear'] + 1) ),
                   on=['gvkey', 'fyear'], how='inner', suffixes=['', '_py'])
           .assign(KS = lambda x: -7.883 + 0.566*x['FPS'] + 0.518*x['ln_at_py'] + 0.982*x['sales_gr_py']
                                  + 0.379*x['car_py'] - 0.108*x['ret_skewness_py'] + 25.635*x['std_ret_py']
                                  + 0.00007*x['turnover_py']/1000,
                   KSLitRisk = lambda x: ( np.exp(x['KS'])/(1+np.exp(x['KS'])) ) )
           .filter(['gvkey', 'fyear', 'KS', 'KSLitRisk'])
           )

#%%
'''
Export
'''

comp_df.to_parquet(f'{main_loc}/3. Data/2. Processed Data/19. KS variables - 20210621.gzip', compression='gzip', index=False)


