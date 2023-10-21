
#%%
'''
Import Libraries
'''

from collections import Counter
import os
import numpy as np
import pandas as pd
import re
from scipy.stats import ttest_ind
import sys

pd.set_option('display.max_columns', 999,
              'display.width', 1000)

sys.path.append('E:/Dropbox/Python/Custom Modules')
import Antonis_Modules as ak

data_loc = 'E:/Dropbox/Projects/Litigation Reputation/3. Data'
stanf_loc = 'E:/Dropbox/Projects/Other Data/Stanford Securities'


# %%
'''
Load and Pre-process Data and keep only those that are in our final sample
'''

keep_df = pd.read_excel(f'{data_loc}/2. Processed Data/13. Cases in final sample_20220428.xlsx')
rel_mscad_id_set = set(keep_df['MSCAD_ID'])

main_df = (pd
           .read_parquet(f'{data_loc}/2. Processed Data/1. Main Sample_20220428.gzip')
		   # Keep relevant observations and columns
           .query('Sued_sample == 1 and Post == 0 and MSCAD_ID in @rel_mscad_id_set')
           .filter(['MSCAD_ID', 'gvkey', 'LOSS_START_DATE', 'LOSS_END_DATE', 'FILING_DATE', 'Stanford_ID', 'Settled'])
           )

del keep_df, rel_mscad_id_set


#%%
'''
Let's load all filings from Stanford
'''

rel_stanf_id = set(main_df['Stanford_ID'])
filings_df = (pd
              .read_excel(f'{stanf_loc}/Initial Files/20211223/file_links_20211223.xlsx')
              # Extract Stanford_ID from Link and covert to numeric
              .assign(Stanford_ID=lambda x: x['Link'].str.extract('=(\d+)$'))
              .assign(Stanford_ID=lambda x: pd.to_numeric(x['Stanford_ID'], downcast='integer'))
              # Keep only FIC filings and only cases that are in our sample
              .query('Type == "fic" and Stanford_ID in @rel_stanf_id')
              # Convert Date to numeric
              .assign(Date=lambda x: pd.to_datetime(x['Date']))
              )

del rel_stanf_id


#%%
'''
Let's bring in all filings and keep those that have dates within +/-3 days from FILING_DATE per Advisen, then we will check for the
word compalint
'''


def prepare_filename(row):

	if row['ShortName'].endswith('.pdf'):
		if os.path.isfile(f'{stanf_loc}/txt folder using tika/{row["Stanford_ID"]}_{row["Type"]}_{row["ShortName"][:-4]}.txt'):
			return f'{stanf_loc}/txt folder using tika/{row["Stanford_ID"]}_{row["Type"]}_{row["ShortName"][:-4]}.txt'
		elif os.path.isfile(f'{stanf_loc}/txt folder/{row["Stanford_ID"]}_{row["Type"]}_{row["ShortName"][:-4]}.txt'):
			return f'{stanf_loc}/txt folder/{row["Stanford_ID"]}_{row["Type"]}_{row["ShortName"][:-4]}.txt'
	elif row['ShortName'].endswith(('.html', '.htm')):
		return f'{stanf_loc}/txt folder/{row["Stanford_ID"]}_{row["Type"]}_{row["ShortName"].split(".")[0]}.txt'


temp_df = (main_df
           .merge(filings_df, on=['Stanford_ID'], how='left')
           # Keep if the word complaint appears and does not have amended
           .assign(complaint=lambda x: x['File_Name'].apply(lambda y: 1 if
									re.search(r'complaint|\bjury\b.*?\b(?:trial|demand)', str(y), flags=re.I) else 0),
                   amended=lambda x: x['File_Name'].apply(lambda y: 1 if re.search('amended', str(y), flags=re.I) else 0))
           .query('complaint == 1 and amended == 0')
           # If multiple keep the earliest one
           .sort_values(by=['MSCAD_ID', 'Date'])
           .drop_duplicates(subset=['MSCAD_ID'], keep='first')
           # Clean Data
           .drop(['complaint', 'amended', 'Sorting', 'File_Name'], axis=1)
           # Also get downloaded filename ready
           .assign(ShortName=lambda x: x['File_Link'].apply(lambda y: str(y).split('/')[-1],),
                   file_name=lambda x: x.apply(lambda y: prepare_filename(y), axis=1))
           .drop(['Date', 'File_Link', 'Link', 'Type'], axis=1)
           )

# These complaints had issues with conversions, so exclude
temp_df = temp_df[~temp_df['MSCAD_ID'].isin([50165, 50461, 51106, 50645, 51657, 600061, 600940])]

del filings_df


#%%
'''
Read the data in
'''

def read_complaint(file_name):

	try:
		with open(file_name, 'r', encoding='utf-8') as f:
			txt = f.read()
	except:
		txt = 'ERROR'

	return txt


temp_df = (temp_df
		   # Read the complaint
           .assign(orig_complaint_txt=lambda x: x['file_name'].apply(lambda y: read_complaint(y)))
		   # Remove complaints that we cannot read
           .query('orig_complaint_txt != "ERROR"')
           )


#%%
'''
Keep only the relevant part of the complaint
'''


def text_preprocessing(txt):

	# Replace some weird spaces
	txt = re.sub('\xa0', ' ', txt)

	# Remove and leading numbers
	txt = re.sub('\n\d+', '\n', txt)
	txt = re.sub('\n *?[IVX]+\.', '\n', txt)

	# Remove any leading spaces
	txt = re.sub('\n[ |\t]+', '\n', txt)

	# Remove any spaces at the end of the line and then just add one
	txt = re.sub('[ |\t]+\n', '\n', txt)
	txt = re.sub('\n', ' \n', txt)

	# Remove Case lines on top of a page
	txt = re.sub('\nCase:? \d.*?\d \n', '\n\n', txt, flags=re.I|re.M)

	# Let's remove page numbers
	txt = re.sub('\n(?:- ?\d+ ?-|\d{1,3}|Page \d+ of[^\n]+) \n', '\n', txt)
	txt = re.sub('\n \n\s+', '\n \n', txt)

	# Fix some issues
	txt = re.sub("COUNT!|COUNT'", 'COUNT I', txt)
	txt = re.sub('RELIE F', 'RELIEF', txt)
	txt = re.sub('SCIENTER AND SCHEME ALLEGATION S', 'SCIENTER AND SCHEME ALLEGATIONS', txt)
	txt = re.sub('SUBSTANTIVE ALLEGATIONS COMMON TO ALL COUNT S', 'SUBSTANTIVE ALLEGATIONS COMMON TO ALL COUNTS', txt)
	txt = re.sub('SUBSTANTIVE ALLEGATION S', 'SUBSTANTIVE ALLEGATIONS', txt)
	txt = re.sub('FACITAL ALLWATIONS -	 •', 'FACTUAL ALLEGATIONS', txt)
	txt = re.sub('FIRST CLAIM FOR RELIE F', 'FIRST CLAIM FOR RELIEF', txt)
	txt = re.sub('\nCOUNTI \n', '\nCOUNT I \n', txt)
	txt = re.sub("I\)EFEIVDANTS' FALSE AND MISLEADING STATEMENTS", "DEFENDANTS' FALSE AND MISLEADING STATEMENTS", txt)
	txt = re.sub('CILASS ACTION ALLEGATIONS', 'CLASS ACTION ALLEGATIONS', txt)
	txt = re.sub('COUNT 1 i', 'COUNT 1', txt)

	return txt


def get_rel_txt(row):

	txt = row['orig_complaint_txt']

	# Preprocess txt
	txt = text_preprocessing(txt)

	# Deal with the weird ones first
	if row['Stanford_ID'] == 100517:
		return txt[txt.find('Count I.  Exchange Act Section 14(a)'):txt.find('Count II - Securities Act Section 11(a)')]
	elif row['Stanford_ID'] == 104511:
		return txt[txt.find('\nTHE PARTIES \n'):txt.find('\nCLASS ACTION ALLEGATIONS \n')]
	elif row['Stanford_ID'] == 100788:
		return txt[txt.find('\nSUBSTANTIVE ALLEGATIONS \n'):txt.find('\nFIRST CLAIM \n')]


	# Identify starting point and grab all remaining text
	# Our primary identifier is "Substantive Allegations" and "False and Misleading Statements" -- only if we don't find those do we
	# go to the next list of keywords
	main_start_ptr = re.compile('\n(?:Substantive Allegations|False and Misleading Statements) \n', flags=re.I)
	sec_start_ptr = re.compile('\n(?:Additional Substantive Allegations|The Facts|Basis of Allegations|Scienter And Scheme Allegations|'
	                           'Background|Defendants. False and misleading|Substantive allegations common to all counts|'
	                           'false and misleading|Underlying Facts|Factual Allegations|Background to the action|'
	                           'Background to Defendants. scheme|Background to the class period|Facts|General Factual Allegations|'
	                           'Materially false and misleading|OVERVIEW OF HOME DEPOT.S FRAUDULENT RTV SCHEME|Defendants. Scheme|'
	                           'Fraudulent Scheme and Course of Business|General Allegations|Class Period Statements|'
	                           'Defendants. fraudulent course of conduct|Factual Background|'
	                           'Defendants. False and Misleading Statements|The scheme to defraud|FAME STARTS|'
	                           'VI, OVERVIEW OF LILLY.S FRAUDULENT SCHEME|Statement of Facts|Substantive Allegations.|'
	                           'Class Period Events and Statements|Misleading Statements and Omissions|'
	                           'JP Morgan.s misrepresentations and omissions) \n',
	                           flags=re.I)
	if main_start_ptr.search(txt) or sec_start_ptr.search(txt):
		if main_start_ptr.search(txt):
			start_pos = main_start_ptr.search(txt).start()
		else:
			start_pos = sec_start_ptr.search(txt).start()
		txt = txt[start_pos:]

		# Identify point to stop --- the earliest point among the following phrases
		end_ptr = re.compile('\n(?:Class Action Allegations|Counts{0,1}|Count I|Count 1|Applicability of presumption of reliance:{0,1}|'
		                     'First Claim|First Claim for Relief|No Statutory Safe Harbor|No Safe Harbor|Claims For Relief|'
		                     'Inapplicability of Statutory Safe Harbor|Plaintifs. Class Action Allegations|First Count|'
		                     'First Cause of Action|AS AND FOR A FIRST CAUSE OF ACTION  FOR VIOLATION OF SECTION 11 OF THE|'
		                     'Statutory Safe Harbor|Count One) \n', flags=re.I)
		if end_ptr.search(txt):
			end_position = end_ptr.search(txt).start()
			txt = txt[:end_position]

			return txt.strip()


temp_df = (temp_df
           .assign(complaint_txt=lambda x: x.apply(lambda y: get_rel_txt(y), axis=1))
           .dropna(subset=['complaint_txt'])
           )


#%%
'''
Search for Reputation Kwds
'''


kwd_list = ['trustworthiness', 'reputation', 'character', 'honor']
kwd_search = fr'\b(?:{"|".join(kwd_list)})\b'
kwd_search = re.compile(kwd_search, flags=re.I|re.M)
def reputation_kwd_search(f):

	return np.log(1 + len(kwd_search.findall(f)) )

temp_df = (temp_df
		   .assign(log_ReputationKwd_count=lambda x: x['complaint_txt'].apply(lambda y: reputation_kwd_search(y)),
                   ReputationKwd=lambda x: np.where(x['log_ReputationKwd_count']>0, 1, 0) )
           )

del kwd_list, kwd_search


#%%
'''
Get Length and Cazier kwd searches
'''

def get_length(txt):

	# Remove numbers
	txt = re.sub('\$?\d(?:\.|,)\d', '', txt)
	txt = re.sub('[\$\d]+', '', txt)

	# Remove any non-alpha characters
	txt = re.sub(r'[^a-zA-Z]', '', txt)

	# Remove all spaces
	txt = re.sub('\s+', '', txt)

	return len(txt)


def cazier_kwds(type, txt):

	txt = re.sub('\s+', ' ', txt)

	if type == 'systemic_allegations':
		kwd_list = ['conflict of interest', 'conflicts of interest', 'knew', 'knowingly', 'knowledge', 'scam', 'scandal', 'scheme',
		            'self\-interest']
	elif type == 'internal_control':
		kwd_list = ['disclosure control', 'internal control', 'financial control', 'material weakness',
		            'generally accepted accounting principles', 'GAAP']

	srch_ptr = re.compile(fr'\b(?:{"|".join(kwd_list)})\b', flags=re.I)

	return len(srch_ptr.findall(txt))


temp_df = (temp_df
           .assign(complaint_length=lambda x: x['complaint_txt'].apply(lambda y: get_length(y)),
                   log_complaint_length=lambda x: np.log(x['complaint_length']),
                   systemic_allegations_kwd=lambda x: x['complaint_txt'].apply(lambda y: cazier_kwds('systemic_allegations', y)),
                   log_systemic_allegations_kwd=lambda x: np.log(x['systemic_allegations_kwd'] + 1),
                   internal_control_kwd=lambda x: x['complaint_txt'].apply(lambda y: cazier_kwds('internal_control', y)),
                   log_internal_control_kwd=lambda x: np.log(x['internal_control_kwd'] + 1))
           )

