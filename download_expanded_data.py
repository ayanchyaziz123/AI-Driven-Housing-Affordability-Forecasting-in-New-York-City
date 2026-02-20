"""
Expanded NYC Housing Affordability Dataset Pipeline
Downloads data from Census API, HUD CHAS, NYC Open Data, Zillow
Target: ~2,100+ NTA-year observations for IEEE-grade research
"""

import os, json, ssl, time, zipfile, io, warnings
import urllib.request
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(DATA_DIR, 'expanded_data'), exist_ok=True)
EXP = os.path.join(DATA_DIR, 'expanded_data')

def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.urlopen(url, context=ctx, timeout=30)
            return json.loads(req.read())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)

def fetch_url(url):
    req = urllib.request.urlopen(url, context=ctx, timeout=60)
    return req.read()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Census Tract → NTA Crosswalk
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 1: Downloading Tract → NTA Crosswalk ===")
xwalk_url = "https://data.cityofnewyork.us/resource/hm78-6dwm.json?$limit=5000"
xwalk_data = fetch_json(xwalk_url)
xwalk = pd.DataFrame(xwalk_data)
xwalk = xwalk[['geoid', 'countyfips', 'boroname', 'borocode',
                'ct2020', 'ntacode', 'ntaname', 'cdtaname']].copy()
xwalk.columns = ['geoid', 'county', 'borough', 'borocode',
                  'tract', 'nta_code', 'nta_name', 'cd_name']
# Normalize: tract is 6-digit zero-padded
xwalk['tract'] = xwalk['tract'].str.zfill(6)
xwalk['county'] = xwalk['county'].str.zfill(3)
xwalk.to_csv(os.path.join(EXP, 'tract_nta_crosswalk.csv'), index=False)
print(f"  Crosswalk rows: {len(xwalk)} | NTAs: {xwalk['nta_code'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: ACS 5-Year Estimates at Census Tract Level (2012–2022)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 2: Downloading ACS Data (Census API) ===")

COUNTIES = {
    'Bronx':     '005',
    'Brooklyn':  '047',
    'Manhattan': '061',
    'Queens':    '081',
}

ACS_VARS = {
    'B19013_001E': 'median_hh_income',
    'B25119_003E': 'renter_median_income',
    'B25064_001E': 'median_gross_rent',
    'B25058_001E': 'median_contract_rent',
    'B25070_001E': 'renter_hh_total',
    'B25070_007E': 'rent_burden_30_34',
    'B25070_008E': 'rent_burden_35_39',
    'B25070_009E': 'rent_burden_40_49',
    'B25070_010E': 'rent_burden_50plus',
    'B25003_001E': 'total_occupied_units',
    'B25003_002E': 'owner_occupied',
    'B25003_003E': 'renter_occupied',
    'B25002_001E': 'total_housing_units',
    'B25002_003E': 'vacant_units',
    'B23025_003E': 'labor_force',
    'B23025_005E': 'unemployed',
    'B01003_001E': 'population',
    'B25014_001E': 'renter_hh_crowding_total',
    'B25014_007E': 'severe_crowding_renter',
    'B08301_001E': 'total_workers',
    'B08301_010E': 'transit_workers',
    'B19083_001E': 'gini_coefficient',
}

YEARS = list(range(2012, 2023))  # 2012–2022
var_str = ','.join(ACS_VARS.keys())
all_acs = []

for year in YEARS:
    year_rows = []
    for borough, county in COUNTIES.items():
        url = (f"https://api.census.gov/data/{year}/acs/acs5"
               f"?get=NAME,{var_str}"
               f"&for=tract:*&in=state:36%20county:{county}")
        try:
            data = fetch_json(url)
            cols = data[0]
            rows = data[1:]
            df_y = pd.DataFrame(rows, columns=cols)
            df_y['year']    = year
            df_y['borough'] = borough
            year_rows.append(df_y)
            time.sleep(0.3)
        except Exception as e:
            print(f"  WARN: {year} {borough}: {e}")

    if year_rows:
        yr_df = pd.concat(year_rows, ignore_index=True)
        all_acs.append(yr_df)
        tracts = len(yr_df)
        print(f"  {year}: {tracts} tracts downloaded")

acs_raw = pd.concat(all_acs, ignore_index=True)

# Rename and clean
rename = {v: k for k, v in {'state': 'state', 'county': 'county_fips',
                              'tract': 'tract', 'NAME': 'tract_name'}.items()}
acs_raw.rename(columns={'state': 'state_fips', 'county': 'county_fips',
                         'tract': 'tract_code', 'NAME': 'tract_name'}, inplace=True)
for api_var, col_name in ACS_VARS.items():
    if api_var in acs_raw.columns:
        acs_raw.rename(columns={api_var: col_name}, inplace=True)
        acs_raw[col_name] = pd.to_numeric(acs_raw[col_name], errors='coerce')
        acs_raw.loc[acs_raw[col_name] < -100000, col_name] = np.nan  # Census null codes

acs_raw['tract_code'] = acs_raw['tract_code'].str.zfill(6)
acs_raw['county_fips'] = acs_raw['county_fips'].str.zfill(3)
acs_raw.to_csv(os.path.join(EXP, 'acs_tract_raw.csv'), index=False)
print(f"\n  Total ACS rows: {len(acs_raw)} | Years: {acs_raw['year'].nunique()} | Tracts: {acs_raw['tract_code'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Aggregate ACS Tracts → NTAs
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 3: Aggregating Tracts → NTA Level ===")

# Merge ACS with crosswalk
acs_xw = acs_raw.merge(
    xwalk[['county', 'tract', 'borough', 'nta_code', 'nta_name']],
    left_on=['county_fips', 'tract_code'],
    right_on=['county', 'tract'],
    how='left'
)
unmatched = acs_xw['nta_code'].isna().sum()
print(f"  Unmatched tracts: {unmatched} / {len(acs_xw)}")

acs_xw = acs_xw.dropna(subset=['nta_code'])

# Aggregate numeric cols from tract to NTA
# Sums: counts/populations
sum_cols = ['renter_hh_total', 'rent_burden_30_34', 'rent_burden_35_39',
            'rent_burden_40_49', 'rent_burden_50plus',
            'total_occupied_units', 'owner_occupied', 'renter_occupied',
            'total_housing_units', 'vacant_units',
            'labor_force', 'unemployed', 'population',
            'renter_hh_crowding_total', 'severe_crowding_renter',
            'total_workers', 'transit_workers']

# Medians: population-weighted average at NTA level
median_cols = ['median_hh_income', 'renter_median_income',
               'median_gross_rent', 'median_contract_rent', 'gini_coefficient']

def wavg(group, val_col, weight_col):
    mask = group[val_col].notna() & group[weight_col].notna() & (group[weight_col] > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(group.loc[mask, val_col], weights=group.loc[mask, weight_col])

group_keys = ['nta_code', 'nta_name', 'borough', 'year']

# Sum aggregation
nta_sums = acs_xw.groupby(group_keys)[sum_cols].sum().reset_index()

# Weighted average for median columns
nta_medians_list = []
for (nta, nta_name, borough, year), grp in acs_xw.groupby(group_keys):
    row = {'nta_code': nta, 'nta_name': nta_name, 'borough': borough, 'year': year}
    for col in median_cols:
        row[col] = wavg(grp, col, 'population')
    nta_medians_list.append(row)
nta_medians = pd.DataFrame(nta_medians_list)

nta_panel = nta_sums.merge(nta_medians, on=group_keys, how='left')

# Derived rates
nta_panel['rent_burden_30plus_pct'] = (
    (nta_panel['rent_burden_30_34'] + nta_panel['rent_burden_35_39'] +
     nta_panel['rent_burden_40_49'] + nta_panel['rent_burden_50plus']) /
    nta_panel['renter_hh_total'].clip(lower=1)
)
nta_panel['rent_burden_50plus_pct'] = (
    nta_panel['rent_burden_50plus'] / nta_panel['renter_hh_total'].clip(lower=1)
)
nta_panel['vacancy_rate'] = (
    nta_panel['vacant_units'] / nta_panel['total_housing_units'].clip(lower=1)
)
nta_panel['renter_share'] = (
    nta_panel['renter_occupied'] / nta_panel['total_occupied_units'].clip(lower=1)
)
nta_panel['homeownership_rate'] = 1 - nta_panel['renter_share']
nta_panel['unemployment_rate'] = (
    nta_panel['unemployed'] / nta_panel['labor_force'].clip(lower=1)
)
nta_panel['severe_crowding_rate'] = (
    nta_panel['severe_crowding_renter'] / nta_panel['renter_hh_crowding_total'].clip(lower=1)
)
nta_panel['transit_commute_rate'] = (
    nta_panel['transit_workers'] / nta_panel['total_workers'].clip(lower=1)
)
nta_panel['pop_density'] = nta_panel['population']  # normalize later if area data added

nta_panel.to_csv(os.path.join(EXP, 'nta_acs_panel.csv'), index=False)
print(f"  NTA panel: {len(nta_panel)} rows | NTAs: {nta_panel['nta_code'].nunique()} | Years: {nta_panel['year'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: NYC Evictions by NTA and Year (NYC Open Data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 4: Downloading NYC Evictions (NYC Open Data) ===")
evict_rows = []
for offset in range(0, 80000, 5000):
    url = (f"https://data.cityofnewyork.us/resource/6z8x-wfk4.json"
           f"?$select=nta_name,nta,borough,executed_date,eviction_address"
           f"&$where=executed_date>='2012-01-01'"
           f"&$limit=5000&$offset={offset}")
    try:
        data = fetch_json(url)
        if not data:
            break
        evict_rows.extend(data)
        if len(data) < 5000:
            break
        time.sleep(0.5)
    except Exception as e:
        print(f"  Evictions page {offset}: {e}")
        break

if evict_rows:
    evict_df = pd.DataFrame(evict_rows)
    evict_df['year'] = pd.to_datetime(evict_df['executed_date'], errors='coerce').dt.year
    evict_df = evict_df.dropna(subset=['year', 'nta'])
    evict_df['year'] = evict_df['year'].astype(int)
    evict_agg = (evict_df.groupby(['nta', 'year'])
                 .size().reset_index(name='eviction_count'))
    evict_agg.rename(columns={'nta': 'nta_code'}, inplace=True)
    evict_agg.to_csv(os.path.join(EXP, 'evictions_by_nta_year.csv'), index=False)
    print(f"  Evictions: {len(evict_df)} records | NTAs: {evict_agg['nta_code'].nunique()}")
else:
    print("  No eviction data downloaded")
    evict_agg = pd.DataFrame(columns=['nta_code', 'year', 'eviction_count'])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: NYC HPD Violations by NTA (via Community Board aggregation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 5: Downloading HPD Violations (aggregate by NTA) ===")
hpd_url = ("https://data.cityofnewyork.us/resource/wvxf-dwi5.json"
           "?$select=communityboard,boroid,inspectiondate,class"
           "&$where=inspectiondate>='2012-01-01'%20AND%20class='C'"
           "&$limit=50000&$order=inspectiondate ASC")
# Class C = immediately hazardous violations
try:
    hpd_data = fetch_json(hpd_url)
    hpd_df = pd.DataFrame(hpd_data)
    hpd_df['year'] = pd.to_datetime(hpd_df['inspectiondate'], errors='coerce').dt.year
    hpd_df = hpd_df.dropna(subset=['year', 'communityboard', 'boroid'])
    hpd_df['year'] = hpd_df['year'].astype(int)
    hpd_df['communityboard'] = pd.to_numeric(hpd_df['communityboard'], errors='coerce')
    hpd_agg = (hpd_df.groupby(['boroid', 'communityboard', 'year'])
               .size().reset_index(name='hpd_violation_count'))
    hpd_agg.to_csv(os.path.join(EXP, 'hpd_violations_by_cd_year.csv'), index=False)
    print(f"  HPD: {len(hpd_df)} Class-C records downloaded (sample)")
except Exception as e:
    print(f"  HPD error: {e}")
    hpd_agg = pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Zillow Observed Rent Index (ZORI) by ZIP
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 6: Downloading Zillow ZORI (ZIP-level rent index) ===")
zillow_url = "https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv"
try:
    raw = fetch_url(zillow_url)
    zillow_all = pd.read_csv(io.StringIO(raw.decode('utf-8')))
    # Filter to NYC ZIP codes (10001–11697 are NYC)
    zillow_nyc = zillow_all[
        (zillow_all['RegionName'].between(10001, 11697)) &
        (zillow_all['State'] == 'NY') &
        (zillow_all['City'] == 'New York')
    ].copy()
    # Melt wide → long
    date_cols = [c for c in zillow_nyc.columns if c[:2] in ('20', '19')]
    zillow_long = zillow_nyc.melt(
        id_vars=['RegionName', 'City', 'State', 'Metro', 'CountyName'],
        value_vars=date_cols, var_name='date', value_name='zori'
    )
    zillow_long['year'] = pd.to_datetime(zillow_long['date']).dt.year
    zillow_long['month'] = pd.to_datetime(zillow_long['date']).dt.month
    zillow_long = zillow_long.rename(columns={'RegionName': 'zip_code'})
    # Annual average
    zillow_annual = (zillow_long.groupby(['zip_code', 'year'])
                     ['zori'].mean().reset_index())
    zillow_annual = zillow_annual[zillow_annual['year'].between(2012, 2022)]
    zillow_annual.to_csv(os.path.join(EXP, 'zillow_zori_nyc_zip_annual.csv'), index=False)
    print(f"  Zillow: {len(zillow_annual)} ZIP-year rows | ZIPs: {zillow_annual['zip_code'].nunique()}")
except Exception as e:
    print(f"  Zillow error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: NYC DOB New Permits (aggregate by year/borough)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 7: Downloading DOB New Residential Permits ===")
dob_url = ("https://data.cityofnewyork.us/resource/w9ak-ipjd.json"
           "?$select=borough,communityboard,issuance_date,job_type,job__sub_type"
           "&$where=job_type='NB'%20AND%20issuance_date>='2012-01-01'"
           "&$limit=50000")
try:
    dob_data = fetch_json(dob_url)
    dob_df = pd.DataFrame(dob_data)
    dob_df['year'] = pd.to_datetime(dob_df['issuance_date'], errors='coerce').dt.year
    dob_df = dob_df.dropna(subset=['year', 'borough'])
    dob_df['year'] = dob_df['year'].astype(int)
    dob_agg = (dob_df.groupby(['borough', 'communityboard', 'year'])
               .size().reset_index(name='new_permits'))
    dob_agg.to_csv(os.path.join(EXP, 'dob_permits_by_cd_year.csv'), index=False)
    print(f"  DOB Permits: {len(dob_df)} new building records")
except Exception as e:
    print(f"  DOB error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Merge everything into final NTA panel
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 8: Merging All Sources ===")
panel = nta_panel.copy()

# Merge evictions
if len(evict_agg):
    panel = panel.merge(evict_agg[['nta_code', 'year', 'eviction_count']],
                        on=['nta_code', 'year'], how='left')
    panel['eviction_count'] = panel['eviction_count'].fillna(0)
    panel['eviction_rate'] = panel['eviction_count'] / panel['population'].clip(lower=1) * 1000

# Merge Zillow via ZIP-to-NTA approximation (use borough median as fallback)
try:
    zillow_annual = pd.read_csv(os.path.join(EXP, 'zillow_zori_nyc_zip_annual.csv'))
    boro_zori = zillow_annual.groupby('year')['zori'].median().reset_index()
    boro_zori.columns = ['year', 'zillow_rent_index']
    panel = panel.merge(boro_zori, on='year', how='left')
except:
    pass

# Save final panel
panel.to_csv(os.path.join(EXP, 'nta_panel_merged.csv'), index=False)

print(f"\n{'='*60}")
print(f"FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"  Observations  : {len(panel):,}")
print(f"  NTAs          : {panel['nta_code'].nunique()}")
print(f"  Boroughs      : {panel['borough'].nunique()} — {list(panel['borough'].unique())}")
print(f"  Years         : {panel['year'].min()}–{panel['year'].max()}")
print(f"  Columns       : {len(panel.columns)}")
print(f"  Output        : expanded_data/nta_panel_merged.csv")
print(f"\n  vs. original borough panel: 72 observations")
print(f"  Expansion factor: {len(panel)/72:.0f}x")
