

#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
														Load relevant libraries
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 100,
			  'display.width', 1000)

data_loc = 'E:/Dropbox/Projects/Litigation Reputation/3. Data/1. Raw Data'

#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
															Reputation Data
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# Our scrapped data
rep_df = (pd
          .read_csv(f'{data_loc}/Scrapped Data/5c. Combined_Sample_with_Gvkey_and_initial_correction.csv')
          .query('Year <= 2016 and gvkey != -1')
          )
rep_df = rep_df[~( (rep_df['Source']=='Fortune') & (rep_df['Year'] == 2014))]
rep_df = rep_df[['gvkey', 'Firm', 'FirmLink', 'Score', 'Industry', 'Year']]


# PDF Files
rep1_df = (pd
           .read_excel(f'{data_loc}/Scrapped Data/7k. PDF files - Linktable.xlsx',
                       usecols=['FortuneDate', 'FirmEdited', 'Score', 'gvkey', 'Industry'],
					   parse_dates=['FortuneDate'])
           .assign(Year=lambda x: x['FortuneDate'].dt.year)
           .drop(['FortuneDate'], axis=1)
           .query('gvkey != -1')
           .rename(columns={'FirmEdited': 'Firm'})
           )

rep_df = (pd
          .concat([rep_df, rep1_df])
          .sort_values(by=['gvkey', 'Year'])
          .reset_index(drop=True)
		  .dropna(subset=['Score'])
          .assign(Score=lambda x: pd.to_numeric(x['Score']).round(2),
		  		  Industry=lambda x: x['Industry'].str.upper() )
          )
del rep1_df

print(len(rep_df))


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
												Adjust Industry Names so we can get Mins
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def adjust_name(ind_txt):

	ind_txt = ind_txt.strip()

	if ind_txt == 'ADVERTISING, MARKETING':
		return 'ADVERTISING AND MARKETING'
	elif ind_txt in ['AUTOMOTIVE RETAILING, SVCS.', 'AUTOMOTIVE, RETAILING', 'AUTOMOTIVE RETAILING, SERVICES',
	                 'AUTOMOTIVE RETAILING, SVCS.']:
		return 'AUTOMOTIVE RETAILING SVCS.'
	elif ind_txt == 'PHARMACEUTICAL':
		return 'PHARMACEUTICALS'
	elif ind_txt in ['MAIL, PKG. & FREIGHT DELIVERY', 'MAIL, PKG. & FREIGHT DELIVERY', 'PACKAGE & FREIGHT DELIVERY',
	                 'MAIL, PKG., FREIGHT DELIVERY', 'DELIVERY', 'DELIVERY AND LOGISTICS']:
		return 'MAIL, PACKAGE & FREIGHT DELIVERY'
	elif ind_txt in ['SOAPS AND COSMETICS', 'HOUSEHOLD & PERSONAL PRODUCTS', 'SOAPS, COSMETICS']:
		return 'HOUSEHOLD AND PERSONAL PRODUCTS'
	elif ind_txt == 'FOOD & DRUG STORES':
		return 'FOOD AND DRUG STORES'
	elif ind_txt in ['WHOLESALERS: ELECTRN., OFFICE', 'WHOLESALERS: ELECTRONICS AND OFFICE EQUIPMENT',
	                 'WHOLESALERS: ELECTRON., OFFICE', 'WHOLESALERS: ELECTRONICS']:
		return 'WHOLESALERS: OFFICE EQUIPMENT AND ELECTRONICS'
	elif ind_txt in ['HOME EQUIPMENT, FURNISHINGS', 'HOME EQUIP., FURNISHINGS', 'HOME EQUIP., FURNISHINGS', 'HOUSEHOLD PRODUCTS']:
		return 'HOME EQUIPMENT AND FURNISHINGS'
	elif ind_txt == 'BUILDING MATERIALS, GLASS':
		return 'BUILDING, MATERIALS, GLASS'
	elif ind_txt == 'COMPUTER & DATA SERVICES':
		return 'COMPUTER AND DATA SERVICES'
	elif ind_txt in ['COMPUTER, OFFICE EQUIPMENT', 'COMPUTERS, OFFICE EQUIP.', 'COMPUTERS']:
		return 'COMPUTERS, OFFICE EQUIPMENT'
	elif ind_txt in ['HEALTH CARE: INSURANCE', 'HEALTH CARE: INSURANCE, MANAGED CARE']:
		return 'HEALTH CARE: INSURANCE AND MANAGED CARE'
	elif ind_txt in ['ENGINEERING, CONSTRUCTION', 'ENGINEERING & CONSTRUCTION']:
		return 'ENGINEERING AND CONSTRUCTION'
	elif ind_txt == 'FOREST & PAPER PRODUCTS':
		return 'FOREST AND PAPER PRODUCTS'
	elif ind_txt in ['GENERAL MERCHANDISERS', 'GENERAL MERCHANDISE']:
		return 'GENERAL MERCHANDISER'
	elif ind_txt in ['MOTOR VEHICLE PARTS', 'MOTOR VEHICLES & PARTS', 'MOTOR VEHICLE & PARTS']:
		return 'MOTOR VEHICLES AND PARTS'
	elif ind_txt in ['RUBBER & PLASTIC PRODUCTS', 'RUBBER AND PLASTICS PRODS.']:
		return 'RUBBER AND PLASTIC PRODUCTS'
	elif ind_txt in ['SCIENTIFIC, PHOTO & CONTROL EQUIPMENT', 'SCI., PHOTO, CONTROL EQUIP.']:
		return 'SCIENTIFIC, PHOTOGRAPHIC & CONTROL EQUIPMENT'
	elif ind_txt in ['INDUSTRIAL & FARM EQUIPMENT', 'INDUSTRIAL AND FARM EQUIP.']:
		return 'INDUSTRIAL AND FARM EQUIPMENT'
	elif ind_txt in ['ELECTRIC & GAS UTILITIES', 'ELECTRIC AND GAS']:
		return 'ELECTRIC AND GAS UTILITIES'
	elif ind_txt == 'AEROSPACE':
		return 'AEROSPACE AND DEFENSE'
	elif ind_txt in ['INSURANCE: PROPERTY, CASUALTY', 'INSURANCE: PROP., CASUALTY', 'INSURANCE: PROP, CASUALTY',
	                 'INSURANCE: PROPERTY & CASUALTY']:
		return 'INSURANCE: PROPERTY AND CASUALTY'
	elif ind_txt == 'SUPERREGIONAL BANKS (U.S.)':
		return 'SUPERREGIONAL BANKS'
	elif ind_txt in ['MORTGAGE FINANCE']:
		return 'MORTGAGE SERVICES'
	elif ind_txt in ['ELECTRONICS, NETWORKS', 'NETWORK COMMUNICATIONS']:
		return 'NETWORK AND OTHER COMMUNICATIONS EQUIPMENT'
	elif ind_txt in ['SPECIALTY RETAILERS: DIVERSIFIED', 'SPECIALIST RETAILERS']:
		return 'SPECIALTY RETAILERS'
	elif ind_txt == 'WHOLESALERS: FOOD, GROCERY':
		return 'WHOLESALERS: FOOD AND GROCERY'
	elif ind_txt == 'DIVERSIFIED OUTSOURCING SERVICES':
		return 'DIVERSIFIED OUTSOURCING'
	elif ind_txt in ['MEDICAL PRODUCTS, EQUIP.', 'MEDICAL PRODUCTS', 'MEDICAL PRODUCTS & EQUIPMENT', 'MEDICAL PRODUCTS, EQUIPMENT',
	                 'MEDICAL EQUIPMENT', 'MEDICAL PRODUCTS AND EQUIPMENT']:
		return 'MEDICAL AND OTHER PRECISION EQUIPMENT'
	elif ind_txt in ['INSURANCE: LIFE, HEALTH', 'INSURANCE: LIFE & HEALTH']:
		return 'INSURANCE: LIFE AND HEALTH'
	elif ind_txt == 'OIL AND GAS EQUIPMENT, SVCS.':
		return 'OIL AND GAS EQUIPMENT, SERVICES'
	elif ind_txt in ['TRANSPORTATION, LOGISTICS', 'TRANSPORTATION']:
		return 'TRANSPORTATION AND LOGISTICS'
	elif ind_txt == 'CONSUMER CREDIT CARD AND SERVICES':
		return 'CONSUMER CREDIT CARD AND RELATED SERVICES'
	elif ind_txt == 'BROKERAGE':
		return 'SECURITIES'
	elif ind_txt == 'INFOTECH SERVICES':
		return 'INFORMATION TECHNOLOGY SERVICES'
	elif ind_txt in ['MEGA-BANKS', 'MEGABANKS AND CREDIT CARD COMPANIES', 'MEGABANKS, CREDIT CARD COS.']:
		return 'MEGABANKS'
	elif ind_txt == 'PUBLISHING: NEWSPAPERS, MAGAZINES':
		return 'PUBLISHING: NEWSPAPERS AND MAGAZINES'
	elif ind_txt == 'TRUCKING':
		return 'TRUCKING, TRANSPORTATION, LOGISTICS'
	elif ind_txt in ['MINING, CRUDE-OIL PRODUCTION', 'MINING, CRUDE OIL']:
		return 'MINING AND CRUDE OIL'
	elif ind_txt == 'ELECTRONICS, ELECTRICAL EQUIP.':
		return 'ELECTRONICS, ELECTRICAL EQUIPMENT'

	return ind_txt


rep_df['IndustryAdj'] = rep_df['Industry'].apply(lambda x: adjust_name(str(x)))


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
											Identify cases where the whole industry was dropped
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def identify_industry_drop(lead):

	temp_df = (rep_df
			   .assign(PostYear=lambda x: x['Year'] + lead)
			   # Bring in the firm's next year obs to figure out how many firms from this industry continue to have coverage
			   .merge( (rep_df
			            .assign(Year=lambda x: x['Year'] - lead)
						.filter(['gvkey', 'Year', 'Industry', 'IndustryAdj']) ),
						on=['gvkey', 'Year'], how='left', suffixes=['', '_ny_match'])
			   # Fill missing next year industry values with missing
	           .assign(Industry_ny_match=lambda x: x['Industry_ny_match'].fillna('missing'),
	                   IndustryAdj_ny_match=lambda x: x['IndustryAdj_ny_match'].fillna('missing').replace('nan', 'missing', regex=False))
			   )

	temp1_df = (temp_df
				# Summarize at the Industry-Year level
	            .groupby(['Year', 'PostYear', 'Industry', 'IndustryAdj', 'Industry_ny_match', 'IndustryAdj_ny_match'], as_index=False)
	            .agg(NumFirmsUnderThisClass=('Score', 'count'))
	            .assign(PercentUnderThisClass=lambda x: x['NumFirmsUnderThisClass'] / x.groupby(['Year', 'Industry'])['NumFirmsUnderThisClass'].transform('sum'))
	            )

	return temp1_df


df = (pd
      .concat([identify_industry_drop(1), identify_industry_drop(2)])
      .query('PostYear <= 2012')
      )


industry_drop_df = df.query('PercentUnderThisClass == 1 and Industry_ny_match == "missing"')
complete_df = df.query('PercentUnderThisClass == 1 and Industry_ny_match != "missing"')

# Identify cases where some of the firms lose coverage but the industry continues to exists
some_missing_df = (df
                   .query('PercentUnderThisClass !=1')
                   # If the reason is because some were dropped, but the industry exists and there are only 2 categories (i.e., missing
                   # and the industry itself), then drop as we are ok
                   .assign(missing=lambda x: np.where(x['Industry_ny_match']=='missing', 1, 0),
                           max_missing=lambda x: x.groupby(['Year', 'Industry', 'PostYear'])['missing'].transform('max'),
                           same_industry=lambda x: np.where(x['IndustryAdj'] == x['IndustryAdj_ny_match'], 1, 0),
                           max_same_industry=lambda x: x.groupby(['Year', 'Industry', 'PostYear'])['same_industry'].transform('max'),
                           total_industries=lambda x: x.groupby(['Year', 'Industry', 'PostYear'])['Industry_ny_match'].transform('count') )
                   .query('max_missing == 1 and max_same_industry == 1 and total_industries == 2')
                   )


df = (df
      .merge(some_missing_df[['Year', 'PostYear', 'Industry']], on=['Year', 'PostYear', 'Industry'], how='left', indicator=True)
      .query('_merge != "both"')
      .drop(['_merge'], axis=1)

      .merge(industry_drop_df[['Year', 'PostYear', 'Industry']], on=['Year', 'PostYear', 'Industry'], how='left', indicator=True)
      .query('_merge != "both"')
      .drop(['_merge'], axis=1)

      .merge(complete_df[['Year', 'PostYear', 'Industry']], on=['Year', 'PostYear', 'Industry'], how='left', indicator=True)
      .query('_merge != "both"')
      .drop(['_merge'], axis=1)
      )

# Identify cases where none of the firms is in the same industry next year
industry_chg_df = (df
                   .assign(same_industry=lambda x: np.where(x['IndustryAdj'] == x['IndustryAdj_ny_match'], 1, 0),
                           max_same_industry=lambda x: x.groupby(['Year', 'Industry', 'PostYear'])['same_industry'].transform('max') )
                   .query('max_same_industry == 0')
                   )


to_exclude_df = (pd
                 .concat([industry_drop_df, industry_chg_df])
                 .filter(['Year', 'PostYear', 'Industry'])
                 .drop_duplicates()
                 )

to_exclude_df.to_parquet(f'{data_loc}/../2. Processed Data/28a. Industries losing coverage or breaking into more industries.gzip',
                         index=False, compression='gzip')
