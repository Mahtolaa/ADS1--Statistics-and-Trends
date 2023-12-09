# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:10:39 2023

@author: Mariam Maliki
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Setting Plot Style for the subsequent plots
plt.style.use('seaborn-whitegrid')


def read_data(a, b):
    """
    Reads and imports files from comma separated values, to a python DataFrame

    Arguments:
    a: string, The name of the csv file which is to be read
    b: integer, indicates the number of rows on the csv file to be skipped

    Returns:
    data: A pandas dataframe with all values from the excel file
    transposed_data: The transposed pandas dataframe
    """
    data = pd.read_csv(a, skiprows=b)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    transposed_data = data.set_index(
        data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    transposed_data = transposed_data.set_index('Year').dropna(axis=1)
    transposed_data = transposed_data.drop(['Country Name'])
    return data, transposed_data


# import data from the csv file
a = 'worldbank_data.csv'
b = 4

data, transposed_data = read_data(a, b)

# Dataframe creation to get data for the climate change indicators of interest


def indicator_set(indicators):
    """
    Reads and selects precise indicators related to climate
    change from world bank dataframe, to a python DataFrame

    Arguments:
    indicators: list of climate change related indicators

    Returns:
    ind: A pandas dataframe with specific indicators selected
    """
    ind = data[data['Indicator Name'].isin(indicators)]
    return ind


climate_change_indicators = [
    'CO2 emissions (metric tons per capita)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Forest area (% of land area)',
    'Access to electricity (% of population), Agriculture, forestry, and \
    fishing, value added (% of GDP)', 'Population, total',  
]

ind = indicator_set(climate_change_indicators)

# Slicing the dataframe to get data for the countries of interest


def country_set(countries):
    """
    Reads and selects country of interest from world bank dataframe,
    to a python DataFrame

    Arguments:
    countries: A list of countries selected from the dataframe
    Returns:
    specific_count: A pandas dataframe with specific countries selected
    """
    specific_count = ind[ind['Country Name'].isin(countries)]
    specific_count = specific_count.dropna(axis=1)
    specific_count = specific_count.reset_index(drop=True)
    return specific_count


# Selecting the countries specifically
countries = ['United Kingdom', 'China', 'India', 'Kenya', 'Russian Federation',
             'Bangladesh', 'Canada', 'Sweden', 'Nigeria', 'Maldives']

specific_count = country_set(countries)
