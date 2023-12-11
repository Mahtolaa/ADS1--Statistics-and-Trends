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
    'Access to electricity (% of population)', 
    'Agriculture, forestry, and fishing, value added (% of GDP)', 
    'Population, total'  
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
             'Canada', 'Sweden', 'Nigeria', 'Maldives']
specific_count = country_set(countries)


def grp_countries_ind(indicator):
    """
    Selects and groups countries based on the specific indicators,
    to a python DataFrame

    Arguments:
    indicator: Choosing the indicator

    Returns:
    grp_ind_con: A pandas dataframe with specific countries selected
    """
    grp_ind_con = specific_count[specific_count["Indicator Name"] == indicator]
    grp_ind_con = grp_ind_con.set_index('Country Name', drop=True)
    grp_ind_con = grp_ind_con.transpose().drop('Indicator Name')
    grp_ind_con[countries] = grp_ind_con[countries].apply(pd.to_numeric, errors='coerce', axis=1)
    
    return grp_ind_con

# Giving each indicator a dataframe
co2_em = grp_countries_ind("CO2 emissions (metric tons per capita)")
ren_energy = grp_countries_ind("Renewable energy consumption (% of total final energy consumption)")
for_area = grp_countries_ind("Forest area (% of land area)")
access_to_ele = grp_countries_ind("Access to electricity (% of population)")
pop = grp_countries_ind("Population, total")
agric = grp_countries_ind("Agriculture, forestry, and fishing, value added (% of GDP)")


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    value = np.sum(((dist-aver) / std)**3) / len(dist-1)

    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-1) - 3.0

    return value


# Now check for the skewness and kurtosis of each indicator selected
print(skew(ren_energy))
print(kurtosis(for_area))

# Statistical properties of indicators

# Total population
print(pop.describe())
# CO2 Emission from liquid fuel consumption
print(co2_em.describe())
# Renewable Energy Consumption
print(ren_energy.describe())
# Access to Electricity
print(access_to_ele.describe())
# Forest Area(% of land Area)
print(for_area.describe())
# Agriculture
print(agric.describe())

# Function


def plot_total_population(pop):
    plt.figure(figsize=(15, 10))
    for country in pop.columns:
        plt.plot(pop.index, pop[country], label=country)
    plt.xlabel('Year')
    plt.ylabel('Population, total')
    plt.title('Total Population growth Over years for Selected Countries')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.rcParams["figure.dpi"] = 300
    plt.grid(False)
    plt.show()
 
    
def plot_forest_area(for_area):
    plt.figure(figsize=(15, 10))
    for country in co2_em.columns:
        plt.plot(for_area.index, for_area[country], label=country)
    plt.xlabel('Year')
    plt.ylabel('Forest area (% of land area)')
    plt.title('Forest area Over years for Selected Countries')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.rcParams["figure.dpi"] = 300
    plt.grid(False)
    plt.show()
    

def plot_co2_emissions_bar():
    """
    Plots a bar plot for co2 emissions in the selected countries.
    """


    # Plotting
    plt.figure(figsize=(10, 8))
    co2_em.T.iloc[:, 1::4].plot(kind='bar')
    plt.title("CO2 emissions (metric tons per capita)")
    plt.xlabel("Country")
    plt.ylabel("CO2 Emissions")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.rcParams["figure.dpi"] = 300
    plt.grid(False)
    plt.show()
    
    
def plot_Agriculture_bar():
    """
    Plots a bar plot for co2 emissions in the selected countries.
    """


    # Plotting
    plt.figure(figsize=(10, 8))
    agric.T.iloc[:, 1::4].plot(kind='bar')
    plt.title("Agriculture, forestry, and fishing, value added (% of GDP")
    plt.xlabel("Country")
    plt.ylabel("% of GDP")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.rcParams["figure.dpi"] = 300
    plt.grid(False)
    plt.show()


def plot_heatmap_correlation(data, country,
                             indicators, years, fig_size=(8, 5)):
    """
    Plots a correlation heatmap for selected indicators of a specific country.

    :param data: DataFrame containing the data.
    :param country: String, name of the country to filter the data.
    :param indicators: List of strings, names of the indicators to be included.
    :param years: List of strings, years to be considered for analysis.
    :param dpi: Integer, the resolution of the figure in dots-per-inch.
    :param fig_size: Tuple of integers, the size of the figure (width, height).
    """
    # Filter data for the specified country and indicators
    filtered_df = data[data['Country Name'] == country]
    filtered_df = filtered_df[filtered_df['Indicator Name'].isin(indicators)]
    filtered_df = filtered_df.set_index('Indicator Name')

    # Drop unnecessary columns
    filtered_df = filtered_df.drop(columns=['Country Name']
                                   + [str(year) for year in range(1990, 2020)
                                      if str(year) not in years])

    # Transpose the DataFrame for correlation calculation
    transposed_df = filtered_df.transpose()

    # Calculate correlation
    correlation = transposed_df.corr().round(2)

    # Plotting the heatmap
    plt.imshow(correlation, cmap='Accent_r', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation)), correlation.columns)
    plt.gcf().set_size_inches(*fig_size)
    plt.rcParams["figure.dpi"] = 300

    # Adding labels to the heatmap
    for y in range(correlation.shape[0]):
        for x in range(correlation.shape[1]):
            plt.text(x, y, '{:.2f}'.format(correlation.values[y, x]),
                     ha='center', va='center', color='black')

    plt.title(f'Correlation Map of Indicators for {country}')
    plt.savefig('Heatmap_Correlation.png')
    plt.show()
 
    
indicators = [
    'CO2 emissions (metric tons per capita)',
    'Population, total',
    'Electricity production from oil sources (% of total)',
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Forest area (% of land area)'

]
years = ['2000', '2001', '2002', '2003', '2004',
         '2005', '2006', '2007', '2008', '2009',
         '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2019', '2020']

# Plot-Function Executions
plot_total_population(pop)
plot_forest_area(for_area)
plot_co2_emissions_bar()
plot_Agriculture_bar()
plot_heatmap_correlation(data, 'India', indicators, years)
plot_heatmap_correlation(data, 'Sweden', indicators, years)
