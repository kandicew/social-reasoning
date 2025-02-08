MEETING_DATA = '/PATH/data/meeting.csv'

import pandas as pd

def get_conv(data_path=MEETING_DATA):
    df = pd.read_csv(data_path)
    conv_lst = df['conversation'].to_list()
    label = df['power'].to_list()
    return conv_lst, label

