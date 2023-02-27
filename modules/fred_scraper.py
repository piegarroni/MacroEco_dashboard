import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import date
from functools import reduce
import numpy as np
import time 
import os
from sklearn import preprocessing


class fred_retriever():
    """
    Entity that scrapes fred and returns merged dataframe
    """
    def __init__(self):
        
        self.measures = {'nasdaqcom' :'NASDAQCOM', 'nyse' : 'BOGZ1FL073164003Q', 'willshire' : 'WILL5000INDFC', 
                        'gdp':'GDP', 'gnp':'GNP', 'unem_rate' : 'UNRATE', 'interest_rate':'DFF', 
                        'leverage':'NFCILEVERAGE', 'M2': 'M2SL', 'cpi' :'FPCPITOTLZGUSA', 
                        'expenditures' :'PCECTPI', 'oil':'DCOILWTICO', 'core_inflation': 'CORESTICKM159SFRBATL', 
                        'gdp_defl':'GDPDEF', 'ppi':'PPIACO', 'nnp': 'A027RC1Q027SBEA', '5y Breakeven Inflation': 'T5YIE'}
        
        

    def scrape_fred(self, measure):
        """
        Method to retrieve macro measures from https://fred.stlouisfed.org/series
        """
        # get today's date
        today = str(date.today())

        time.sleep(0.7)

        # get download url given measure and today's date
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=748&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id={measure}&scale=left&cosd=1945-10-01&coed={today}&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={today}&revision_date={today}&nd=1945-10-01'

        headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
        }

        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")

        # convert soup to string
        data = str(soup)

        # convert string to list
        columns = [i.split(",") for i in data.split('\n')][0]
        rows = [i.split(",") for i in data.split('\n')][1:]

        # convert list to dataframe and drop last row
        df = pd.DataFrame(rows, columns=columns)
        df.drop(df.tail(1).index,inplace=True) 

        df['DATE'] = pd.to_datetime(df['DATE'], utc=True)
        return df


    def preprocess_data(self, df):
        """
        Method to clean data, interpolate nans, convert to numeric and feature engineer columns (to change)
        """
      #  df.set_index('DATE', inplace=True)
        
        df= df.loc[df.index > '01-01-1971']
        
        df = df.replace(".", np.nan)

        df = df.apply(pd.to_numeric)

        df = df.interpolate()
        df = df.interpolate(limit_direction='backward')

        def convert_to_change(df):
            """
            Method to convert dataframe variable into change variables
            """
            df1= df.copy()
            for column in df1.columns:
                try:
                    df1[column] = df1[column].replace(".", np.nan)
                    df1[column] = pd.to_numeric(df1[column])
                    df1[column] = df1[column] - df1[column].shift(1)
                except ValueError:
                    pass
            return df1

        df_change = convert_to_change(df)

        df['GDP_change'] = df_change['GDP']
        df['GNP_change'] = df_change['GNP']
        df['A027RC1Q027SBEA_change'] = df_change['A027RC1Q027SBEA']
        df['GDPDEF_change'] = df_change['GDPDEF']

        df['GDP_change'].replace(0, np.nan, inplace=True)
        df['GDP_change'].interpolate(method = 'pad', inplace=True)
        df['GDP_change'].replace(np.nan, 0, inplace=True)

        df['GNP_change'].replace(0, np.nan, inplace=True)
        df['GNP_change'].interpolate(method = 'pad', inplace=True)
        df['GNP_change'].replace(np.nan, 0, inplace=True)

        df['A027RC1Q027SBEA_change'].replace(0, np.nan, inplace=True)
        df['A027RC1Q027SBEA_change'].interpolate(method = 'pad', inplace=True)
        df['A027RC1Q027SBEA_change'].replace(np.nan, 0, inplace=True)

        df['GDPDEF_change'].replace(0, np.nan, inplace=True)
        df['GDPDEF_change'].interpolate(method = 'pad', inplace=True)
        df['GDPDEF_change'].replace(np.nan, 0, inplace=True)

        df['-UNRATE'] = -df['UNRATE'] 
        
        df['BOGZ1FL073164003Q'] = pd.to_numeric(df['BOGZ1FL073164003Q'])
        
        def scaler(df):
            x = df.values
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled, columns = df.columns).set_index(df.index)
            return df
        
        df = scaler(df)
        

        df['ECOGROWTH'] = 0.7*df['-UNRATE'] + 0.7*df['GDP_change'] + 0.6*df['GNP_change'] + 0.5*df['A027RC1Q027SBEA_change'] + 0.4*df['WILL5000INDFC']
        df['INFLATION'] = df['PCECTPI']  + df['FPCPITOTLZGUSA']  + df['NFCILEVERAGE'] + df['CORESTICKM159SFRBATL'] + df['GDPDEF_change'] + df['PPIACO'] + df['M2SL']

        return df
    
    
    def merge(self):
        """
        Method to merge all the datasets
        """
        # retrieve all the measures 
        dataframes=[]
        for i in range(0, len(self.measures)):
            name = list(self.measures.keys())[i]
            value = list(self.measures.values())[i]
            locals()[name] = self.scrape_fred(f"{value}") 
            dataframes.append(locals()[name])


        data = reduce(lambda  left,right: pd.merge_asof(left.sort_values('DATE'), 
                        right.sort_values('DATE'),
                        left_on='DATE',
                        right_on='DATE',
                        direction='backward'), dataframes).set_index('DATE')
        
        data = self.preprocess_data(data)
        return data


my_instance = fred_retriever()
data = my_instance.merge()
data.to_csv(os.getcwd() + '/data.csv')

