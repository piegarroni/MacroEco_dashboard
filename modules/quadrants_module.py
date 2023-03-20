import pandas as pd
import numpy as np
import time
from datetime import date
import requests
from bs4 import BeautifulSoup

def retrieve_fred_csv(measure: str):
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




def preprocess_quadrants(gdp, cpi):
    '''
    Function to preprocess CPI and GDP to visualize quadrants
    '''
    gdp = retrieve_fred_csv("GDP") #quarter 
    gdp = gdp.set_index('DATE')



    #saving_rate = retrieve_fred_csv("PSAVERT") # monthly
    cpi = retrieve_fred_csv("CPIAUCSL") # monthly
    cpi = cpi.set_index('DATE')

    gdp['GDP'] = pd.to_numeric(gdp['GDP'])
    # Calculate inflation rate
    gdp['GDP_rate'] = (gdp['GDP'] - gdp['GDP'].shift(1)) / gdp['GDP'].shift(1)

    # Convert inflation rate to percentage
    gdp['GDP_rate']  = gdp['GDP_rate']  * 100

    # Subset only after year x
    gdp_subset = gdp.loc[gdp.index> '01-01-1989']

    # Smoothen it
    gdp_subset['GDP_rate'] = gdp_subset['GDP_rate'].rolling(4).mean() - 0.745

    gdp_subset['GDP_rate_log'] = np.log10(gdp_subset['GDP_rate'])
    gdp_subset['GDP_rate_log'] =gdp_subset['GDP_rate_log'].fillna(-400)

    # INFLATION -----------------------------------------------------
    cpi['CPIAUCSL'] = pd.to_numeric(cpi['CPIAUCSL'])

    # Calculate inflation rate
    cpi['CPIAUCSL_rate'] = (cpi['CPIAUCSL'] - cpi['CPIAUCSL'].shift(1)) / cpi['CPIAUCSL'].shift(1)

    # Convert inflation rate to percentage
    cpi['CPIAUCSL_rate'] = cpi['CPIAUCSL_rate'] * 1000

    #log inflation rate (cpi)
    cpi_subset = cpi.loc[cpi.index> '01-01-2000']

    # Smoothen it
    cpi_subset['CPIAUCSL_rate'] = cpi_subset['CPIAUCSL_rate'].rolling(24).mean() -1.9

    # convert to log scale 
    cpi_subset['CPIAUCSL_rate_log'] = np.log10(-cpi_subset['CPIAUCSL_rate'])
    cpi_subset['CPIAUCSL_rate_log'] = cpi_subset['CPIAUCSL_rate_log'].fillna(-400)



    df = pd.merge_asof(cpi_subset, 
                    gdp_subset,
                    left_on=cpi_subset.index,
                    right_on=gdp_subset.index,
                    direction='backward').set_index('key_0')
    #print(df.head(10))
    return df

# ### Algorithm (Macroeconomic quadrant)-------------------------------------------------


def quadrants_algorithm(df):


    quadrant = []
    for i in range(len(df)):
        if df['GDP_rate_log'][i] != -400 and df['CPIAUCSL_rate_log'][i] == -400:
            quadrant.append(1)
            #print('invest in ("Quadrant 1: Inflation up, GDP up")') #quadrant 1 consists of the following assets= ["Emerging equities", "International real estate", "Gold", "Commodities", "emerging bond spreads", "Inflation protected bonds"]  
        elif df['GDP_rate_log'][i] == -400 and df['CPIAUCSL_rate_log'][i] == -400:
            quadrant.append(2)

            #print('invest in ("Quadrant 2: Inflation up, GDP down")') # quadrant 2 consists of the following assets= ["Gold", "Commodities", "Emerging bond spreads", "Inflation protected bonds", "cash"]
        elif df['GDP_rate_log'][i] == -400 and df['CPIAUCSL_rate_log'][i] != -400:
            quadrant.append(3)
            #print('invest in ("Quadrant 3: Inflation down, GDP up")') # quadrant 3 consists of the following assets= ["Developed corporate bond spreads", "intermediate treasuries", "Developed real estate", "Developed equities"]  
        elif df['GDP_rate_log'][i] != -400 and df['CPIAUCSL_rate_log'][i] != -400: # add if to remember state
            quadrant.append(4)
            #print('invest in ("Quadrant 4: Inflation down, GDP down")') #quadrant 4 consists of the following assets = ["Gold", "Long term treasuries", "cash"]
        else:
            quadrant.append(np.nan)


    df['quadrant'] = quadrant

    quadrant_cleaned = [i for i in df['quadrant']]
    for i in range(1, len(quadrant_cleaned)):
        if quadrant_cleaned[i] ==4 and quadrant_cleaned[i-1] == 1 or quadrant_cleaned[i] ==4 and quadrant_cleaned[i-1] == 1.5:
            quadrant_cleaned[i]=1.5
        
    df['quadrant_cleaned'] = quadrant_cleaned
    print(quadrant_cleaned)

    return df['quadrant_cleaned'] 


