import pandas as pd  # type: ignore
import matplotlib as plt # type: ignore

df = pd.read_csv('/Users/amardani/IMA-5-Factor-Model/F-F_Research_Data_5_Factors_2x3_daily.csv', header = 3, skipfooter = 2, engine = 'python')
#column_names = df.columns
#print(column_names)

# The date column is 'Unnamed: 0', fixing that here
df = df.rename(columns={'Unnamed: 0': 'Date_old'})
#print(df.head(20))

df['Date'] = pd.to_datetime(df['Date_old'], format='%Y%m%d')

smb_returns = [1]
hml_returns = [1]
rmw_returns = [1]
cma_returns = [1]

for i in range(len(df)):
    prev_smb = smb_returns[-1]
    new_smb = prev_smb * (1 + df['SMB'][i])
    smb_returns.append(new_smb)

    prev_hml = hml_returns[-1]
    new_hml = prev_hml * (1 + df['HML'][i])
    hml_returns.append(new_hml)

    prev_rmw = rmw_returns[-1]
    new_rmw = prev_rmw * (1 + df['RMW'][i])
    rmw_returns.append(new_rmw)

    prev_cma = cma_returns[-1]
    new_cma = prev_cma * (1 + df['CMA'][i])
    cma_returns.append(new_cma)

dates = df['Date']
dates.insert(0, '1963-06-30')

data = {
    'Date' : dates,
    'SMB' : smb_returns,
    'HML' : hml_returns,
    'RMW' : rmw_returns,
    'CMA' : cma_returns
}
returns = pd.DataFrame(data=data)
print(returns.head())
