import pandas as pd

df = pd.read_csv('tam.txt', delimiter='\t', header=None)
df.columns = ['English', 'Tamil', 'License']

print(df.head())

df.drop(columns=['License'], inplace=True)

print(df.head())

# convert english column to lower case
df['English'] = df['English'].str.lower()

# save as tab seperate file without column names
df.to_csv('tam_cleaned.txt', index=False, sep='\t')
