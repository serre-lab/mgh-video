import glob
import pandas as pd
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
cfg = config.get('configsection', 'config')
joints = config.get(cfg, 'joints').split(', ')

fileLists = glob.glob('/remote_cifs/lester/pose_bootstrap_shortlist/**/*.csv')

count = 0
for file in fileLists:
    df = pd.read_csv(file)
    print(f'[{count}]Processing: ' + file)
    # df['quality'] = 0
    #
    # for index, row in df.iterrows():
    #     for joint in joints:
    #         str_val = df.loc[df['frame_n'] == index, joint].values[0]
    #         if index != 0 and '0-0-0' == str_val:
    #             df.loc[df['frame_n'] == index, joint] = df.loc[df['frame_n'] == index-1, joint].values[0]
    #             print(f'[{count}]' + df.loc[df['frame_n'] == index, joint])

    # df.to_csv(file, index=False)


    df.columns.values[0] = "index"
    df.to_csv(file, index=False)
    print('Done!')
    count += 1


