import pandas as pd

df = pd.read_csv('data/chessData.csv')

df_smaller = df.sample(n=10000)

df_smaller.to_csv('data/chessData_10000sub.csv', index=False)

'''
df_smaller_white = df_smaller[df_smaller['FEN'].apply(lambda x: x.split(' ')[1]) =='w']

with open('data/smallChessData3White.csv', 'w') as f:
    df_smaller_white.to_csv(f, index=False)
'''