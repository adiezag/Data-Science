from hashlib import new
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import pgeocode

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, tree
from dtreeviz.trees import dtreeviz
import graphviz

"""
Name:       Antonio Diez
Email:      antonio.diezaguilar90@myhunter.cuny.edu
Resources:  Lecture slides
            https://scikit-learn.org/
Title:      My project
URL:        https://github.com/adiezag/Data-Science
"""





# Note:
# The original file is called 'NYPD_Shooting_Incident_Data__Historic_.csv'
# The following code allowed me to get the zip code for every shooting event.
# However, it takes too much time to compute these values. 
# That's why I exported and saved my dataset as 'NYC_.csv'

# *******************************************************
# filename = 'NYPD_Shooting_Incident_Data__Historic_.csv'
# dataset = pd.read_csv(filename)
# # Dropping columns
# dataset.drop('INCIDENT_KEY', axis = 1, inplace=True)
# dataset.drop('Lon_Lat', axis = 1, inplace=True)


# geolocator = Nominatim(user_agent="test/1")
# def zip_code(df):
#     zpcd = geolocator.reverse("{}, {}".format(df['Latitude'], df['Longitude']))
#     z = zpcd.raw['address']['postcode']
#     return z

# dataset['zipcode'] = dataset.apply(zip_code, axis=1)
# dataset.to_csv('NYC_.csv')
# *******************************************************

# Now, I'll be working  with 'NYC_.csv'


# Open file, drop columns with nan values
# Return dataframe
def make_df_dropna(filename):
    df = pd.read_csv(filename)
    df = df.dropna(axis='columns')

    return df

# Filter zip codes
# Convert ['zipcode'] column into int data type. The purpose is to make it numerical data
def filter_zipcodes(df):
    df = df.iloc[:, 1:] # Delete first column. This column was generated while computing the zip codes.
    df = df.loc[df.zipcode.map(len) == 5]
    #df['zipcode'] = df['zipcode'].astype(int)
    return df

# Add a column to define when the event took place, before or after noon
# Not used
def time_convention(df):
    df['TIME_CONVENTION'] = np.where(df['OCCUR_TIME'] < '11:59:59', 'AM', 'PM')
    return df

# Filter data to keep categorical data only and ['zipcode'] column
def filter(df, columns = ['OCCUR_DATE', 'OCCUR_TIME', 'BORO', 'PRECINCT', 'X_COORD_CD', 'Y_COORD_CD', 'Longitude', 'Latitude']):
    df = df.drop(labels=columns, axis = 1)
    df = df[(df['VIC_AGE_GROUP'] != 'UNKNOWN') & (df['VIC_SEX'] != 'U')]
    return df


def filter_data(df, columns = ['PRECINCT', 'X_COORD_CD', 'Y_COORD_CD', 'Longitude', 'Latitude']):
    df = df.drop(labels=columns, axis = 1)
    df = df[(df['VIC_AGE_GROUP'] != 'UNKNOWN') & (df['VIC_SEX'] != 'U')]
    return df

def date_to_year(df):
    df['YEAR'] = pd.DatetimeIndex(df['OCCUR_DATE']).year
    return df

# ****** Generate a stacked bar plot ******
def stacked_bar_plot(df):
    df = df[['YEAR', 'VIC_RACE']]
    fraction_race = pd.crosstab(index=df['YEAR'], columns = df['VIC_RACE'], normalize = "index")
    proportion_race = pd.crosstab(index = df['YEAR'], columns = df['VIC_RACE'])
    plt.style.use("seaborn")
    proportion_race.plot(kind = 'bar', stacked = True, colormap = 'bone', figsize = (10,6), width = 0.8)
    plt.legend(loc = 1, prop={'size' : 8})
    plt.xlabel("YEAR")
    plt.ylabel("NUMBER OF SHOOTING EVENTS")
    plt.title('NUMBER OF SHOOTING EVENTS BETWEEN 2006 - 2020', size = 20)
    for m, n in enumerate([*proportion_race.index.values]):
        for (a, b, c, d) in zip(proportion_race.loc[n], proportion_race.loc[n], proportion_race.loc[n].cumsum(), fraction_race.loc[n]):
            if (d*100 >= 0.8):
                plt.text(x = m-0.28, y = (c-a)+(a/2), s = f'{b} ({np.round(d*100, 1)}%)', color = 'darkgray', fontsize = 6, fontweight = "bold")

    plt.show()


# Filtrate and return the dataframe for creating a choropleth map

def df_chroropleth(df):
    d_z_r =df[['zipcode','VIC_RACE']]
    proportion_z_r = pd.crosstab(index = d_z_r['zipcode'], columns = d_z_r['VIC_RACE'])
    new_df = proportion_z_r[['AMERICAN INDIAN/ALASKAN NATIVE', 'ASIAN / PACIFIC ISLANDER','BLACK','BLACK HISPANIC', 'UNKNOWN','WHITE','WHITE HISPANIC']].reset_index()
    return new_df

# Get geographic coordinates from zip code
def lat_lon(df):
    LAT = []
    LNG = []
    Z_CODE = df['zipcode'].tolist()
    for i in range(len(Z_CODE)):
        geo_coord = pgeocode.Nominatim('us')
        x = geo_coord.query_postal_code(Z_CODE[i])
        LAT.append(x.latitude)
        LNG.append(x.longitude)
    df['LAT'] = LAT
    df['LNG'] = LNG
    df = df.dropna()
    return df

# Generate a choropleth map
def generate_choropleth_map(df, n_df, geodata="nyc_.geojson"):
    data_zip_code = df['zipcode'].value_counts().rename_axis('z_code').reset_index(name = 'counts')

    map = folium.Map(location=[40.693943, -73.985880], default_zoom_start = 15)
    map.choropleth(geo_data = geodata, data = data_zip_code, columns = ['z_code','counts'], key_on = 'feature.properties.postalCode', fill_color = 'Reds', fill_opacity = 0.75, line_opacity = 0.3, legend_name = 'Number of victims')

    lat = n_df['LAT'].tolist()
    lng = n_df['LNG'].tolist()
    am_in_race = n_df['AMERICAN INDIAN/ALASKAN NATIVE'].tolist()
    as_pa_race = n_df['ASIAN / PACIFIC ISLANDER'].tolist()
    b_race = n_df['BLACK'].tolist()
    b_h_race = n_df['BLACK HISPANIC'].tolist()
    un_race = n_df['UNKNOWN'].tolist()
    wh_race = n_df['WHITE'].tolist()
    wh_h_race = n_df['WHITE HISPANIC'].tolist()
    z_code = n_df['zipcode'].tolist()
    
    m_cluster = MarkerCluster().add_to(map)

    for i in range(len(lat)):
        location = [lat[i], lng[i]]
        tooltip = "Zip code: {}<br> Click for more".format(z_code[i])

        folium.Marker(location, popup="""
                    <i>AMERICAN INDIAN/ALASKAN NATIVE: </i> <br>  <b>{}</b> <br>
                    <i>ASIAN / PACIFIC ISLANDER: </i> <br>  <b>{}</b> <br>
                    <i>BLACK: </i> <br>  <b>{}</b> <br>
                    <i>BLACK HISPANIC: </i> <br>  <b>{}</b> <br>
                    <i>UNKNOWN: </i> <br>  <b>{}</b> <br>
                    <i>WHITE: </i> <br>  <b>{}</b> <br>
                    <i>WHITE HISPANIC: </i> <br>  <b>{}</b> <br>""".format(
                        am_in_race[i],
                        as_pa_race[i],
                        b_race[i],
                        b_h_race[i],
                        un_race[i],
                        wh_race[i],
                        wh_h_race[i]),
                        tooltip=tooltip).add_to(m_cluster)
                 
    map.save('my_map.html')


# ****** Predictive modeling  ******
# The following functions will help to predict the model
# Convert categorical values into indicator variables
def encode_df(df, cols):
    df = pd.get_dummies(df, columns = cols)
    return df

def split_into_train_test_subsets(df_x, df_y, test_size = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = test_size)
    return X_train, X_test, y_train, y_test

def decision_tree_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    return score, model

# ****** Visualization *******
# Generate a GraphViz representation of the decision tree
def tree_visualization(model, y, out_file = 'my_tree.dot'):
    column_names = ['STATISTICAL_MURDER_FLAG_False', 'STATISTICAL_MURDER_FLAG_True','VIC_AGE_GROUP_18-24','VIC_AGE_GROUP_25-44',
                    'VIC_AGE_GROUP_45-64','VIC_AGE_GROUP_65+', 'VIC_AGE_GROUP_<18', 'VIC_SEX_F', 'VIC_SEX_M']
    
    graph = tree.export_graphviz(model, out_file = out_file, feature_names = column_names,
            class_names = sorted(y.unique()), label = 'all', rounded = True, filled = True)
    return graph
   
def dtreeviz_visualization(dataframe):
    column_names = ['STATISTICAL_MURDER_FLAG_False', 'STATISTICAL_MURDER_FLAG_True','VIC_AGE_GROUP_18-24','VIC_AGE_GROUP_25-44',
                    'VIC_AGE_GROUP_45-64','VIC_AGE_GROUP_65+', 'VIC_AGE_GROUP_<18', 'VIC_SEX_F', 'VIC_SEX_M']
    
    dataframe = dataframe.drop(columns = 'zipcode')
    encoder = preprocessing.LabelEncoder()
    encoder.fit(dataframe['VIC_RACE'])
    dataframe['RACE'] = encoder.transform(dataframe['VIC_RACE'])
    model_ = tree.DecisionTreeClassifier()
    model_.fit(dataframe.iloc[:,1:10], dataframe['RACE'])
    d_tree_viz = dtreeviz(model_, x_data = dataframe.iloc[:,1:10], y_data = dataframe['RACE'],
                            target_name = 'VIC_RACE', feature_names = column_names, 
                            class_names = list(encoder.classes_), show_node_labels = True)
    d_tree_viz.save('my_tree.svg')
    return d_tree_viz




# *********** Testing the code ***********
# Get and clean data

# ****************** PART A ******************
# Generate a stacked bar plot

filename = 'NYC_.csv'
data = make_df_dropna(filename)
data = filter_zipcodes(data)
data = filter_data(data)
data = date_to_year(data)


stacked_bar_plot(data) # Returns view of the plot

# ****************** PART B ******************

# Generate a Choropleth map

new_df = df_chroropleth(data)
new_df = lat_lon(new_df)
generate_choropleth_map(data, new_df) # Generates and saves an html file


# ****************** PART C ******************

# Predicting the model

file = 'NYC_.csv'
d_frame = make_df_dropna(file)
d_frame = filter_zipcodes(d_frame)
dataframe = filter(d_frame)

# Convert categorical values into dummy/indicator variables
cols = ['STATISTICAL_MURDER_FLAG','VIC_AGE_GROUP','VIC_SEX']
dataframe = encode_df(dataframe, cols) 

# *** Split the data ***

X = dataframe.drop(columns = ['zipcode', 'VIC_RACE'])
y = dataframe['VIC_RACE']

# Split the data into train and test subsets
X_train, X_test, y_train, y_test = split_into_train_test_subsets(X, y, test_size=0.20)

# Model, predict and get the accuracy score
score, model = decision_tree_model(X_train, X_test, y_train, y_test)

print('The accuracy score of the Decision Tree model is: ', score)

# *** Visualization ****
# --- Visualization A ---
# Representation of the decision tree written into out_file.
graph = tree_visualization(model, y)
print('Tree visualization stored as my_tree.dot')

# --- Visualization B ---
# Another representation of the decision tree using dtreeviz
graph_d_tree_viz = dtreeviz_visualization(dataframe)
graph_d_tree_viz.view()
print('Tree visualization stored as my_tree.svg')