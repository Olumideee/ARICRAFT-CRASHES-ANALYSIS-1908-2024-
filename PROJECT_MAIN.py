import streamlit as st  #issue 70
import pandas as pd 
import altair as alt 
import numpy as np 
import matplotlib.pyplot as plt
def load_dataset():

    #cleaning of the country column
    df = pd.read_csv("aircrahesFullDataUpdated_2024.csv")
    df["Country/Region"] = df["Country/Region"].fillna("N/A")
    df["Operator"].fillna("N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("10", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("'-", "N/A")
    df["Country/Region"] = df["Country/Region"].replace(["Brazil\tAmazonaves", "Brazil\r\nFlorianopolis", "Brazil\tLoide"], "Brazil")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Chile\tAerolineas", "Brazil")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("USSRAerflot", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Western", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Bias", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("18", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("570", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("800", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("near", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("USSRAeroflot", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("unknown0", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("San", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Spain\r\n\t\r\nMoron", "Spain")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("United", "U.S.A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Upper", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("HIPan", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Indonesia\r\n\t\r\nSarmi", "Indonesia")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Los", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Margarita", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("qld", "Queensland")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Qld", "Queensland")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("1unknown", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Azerbaijan\r\n\t\r\nBakou", "Azerbaijan")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("CADuncan", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("USSRBalkan", "Balkan")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("FL", "Florida")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("CAMilitary", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Inner", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Democtratic", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Djibouti\r\n\tDjibouti", "Djibouti")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Tajikistan\tMilitary", "Tajikistan")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Norway\tCHC", "Norway")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("NYUS", "NewYork")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("South-West", "unknown")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Azerbaijan\r\n\t\r\nBakou", "Azerbaijan")
    df['Country/Region'] = df['Country/Region'].replace(['NSW', 'New', 'SK', 'BC', 'PQ', 'San', 'Da', 'WA', 'El', 'PE', 'nsw', 'OLD', 'QC', 'Mt', 'N', 'FL', "de", 'ON', 'SK'], 'N/A')
    df["Country/Region"] = df["Country/Region"].replace(["Brazil\tAmazonaves", "Brazil\r\nFlorianopolis", "Brazil\tLoide"], "Brazil")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Chile\tAerolineas", "Brazil")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("USSRAerflot", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Western", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("Bias", "N/A")
    df['Country/Region'] = df['Country/Region'].replace("US", 'U.S.A')
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("USSRMilitary", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("SC", "N/A")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("?", "")
    df.loc[:, "Country/Region"] = df["Country/Region"].str.replace("U.S,ASRMilitary", "U.S.A")
    # cleaning of the aircraft manufacturers column 
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("?", "")
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("??", "N/A")
    df['Aircraft Manufacturer'] = df['Aircraft Manufacturer'].replace(["HS", "C", "H", "A", "F", "VP", "FALSE", "CH", "Mi"], 'N/A')
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("Doublas", "Douglas")
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("Short", "Short Brothers")
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("139", "N/A")
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("42", "N/A")
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("Unknown /", "N/A")
    df['Aircraft Manufacturer'] = df['Aircraft Manufacturer'].replace("NAMC", "Nihon Aircraft Manufacturing Corporation")
    df['Aircraft Manufacturer'] = df['Aircraft Manufacturer'].replace("Mil", 'Military')
    df.loc[:, "Aircraft Manufacturer"] = df["Aircraft Manufacturer"].str.replace("Swallow\r\nSwallow", "Swallow")
    # cleaning of the aircraft column
    df.loc[:, "Aircraft"] = df["Aircraft"].str.replace("??", "N/A")
    df.loc[:, "Aircraft"] = df["Aircraft"].str.replace("?", "")
    df.loc[:, "Aircraft"] = df["Aircraft"].str.replace("139", "N/A")
    df.loc[:, "Aircraft"] = df["Aircraft"].str.replace("Li 2/ Li", "Lisunov Li-2")
    df.loc[:, "Aircraft"] = df["Aircraft"].str.replace("DC 3", "Douglas DC-3")
    df.loc[:, "Aircraft"] = df["Aircraft"].str.replace("Swallow Swallow", "Swallow")
    # cleaning of the locations column 
    df.loc[:, "Location"] = df["Location"].str.replace("?", "")
    df.loc[:, "Location"] = df["Location"].str.replace("Aeroflot", "N/A")
    # cleaning of the operator column 
    df.loc[:, "Operator"] = df["Operator"].str.replace("-", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("YPF", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Aeroput", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("York?", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("?", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Sabena", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Travel", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("AREA", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Egypt", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("REAL", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Air by", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Zanex", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Ir", "Iran Air")
    df.loc[:, "Operator"] = df["Operator"].str.replace("ATI", "Air Transport International")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Ts", "Air Transnat")
    df.loc[:, "Operator"] = df["Operator"].str.replace("Surrey", "N/A")
    df.loc[:, "Operator"] = df["Operator"].str.replace("TAN", "Transportes Areos Nacionales")
    df['Operator'] = df['Operator'].replace(["Air"], 'N/A')
    

# creating a bin for number of survivors and total fatalities column
    df['Survivors'] = df['Aboard'] - df['Fatalities (air)']
    def categorize_fatality(row):
        if row['Fatalities (air)'] == row['Aboard']:
            return 'Total Fatality'
        elif row['Fatalities (air)'] == 0:
            return 'Total Survival'
        else:
            return 'Non-Total Fatality'


    df['Fatality (Grouping)'] = df.apply(categorize_fatality, axis=1)

# combining the Day, Month and Year into 1 column
    df['Month_Num'] = pd.to_datetime(df['Month'], format='%B', errors='coerce').dt.month
    df['Full_Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month_Num'], day=df['Day']), errors='coerce')
    df['Full_Date'] = df['Full_Date'].dt.date
    df = df.drop(columns=['Year', 'Month', 'Day', 'Month_Num'])
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Full_Date')))
    df = df[cols]
    df.rename(columns={'Full_Date': 'Date'}, inplace=True)

# creating the Seasons column 
    def get_season(date):
        month_name = date.strftime('%B')
        if month_name in ["December", "January", "February"]:  # also fixed typo in "February"
            return 'Winter'
        elif month_name in ["March", "April", "May"]:
            return 'Spring'
        elif month_name in ["June", "July", "August"]:
            return 'Summer'
        elif month_name in ["September", "October", "November"]:
            return 'Fall'

    df['Season'] = df['Date'].apply(get_season)

# creating the decades column 
    df['Year'] = pd.to_datetime(df['Date']).dt.year

    min_year = df['Year'].min()
    max_year = df['Year'].max()

# Create bin edges: every 10 years from min to max
    bins = list(range((min_year // 10) * 10, (max_year // 10 + 1) * 10 + 1, 10))

    labels = [f"{b}s" for b in bins[:-1]]

    df['Decade'] = pd.cut(df['Year'], bins=bins, labels=labels, right=False)

    df.drop(columns=['Year'], inplace=True)


    return df 

# loading the data set 
df = load_dataset()
st.title("ARICRAFT CRASHES ANALYSIS(1908-2024)")


# creating filters 
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
filters={
        "Country/Region":df["Country/Region"].unique(),
        "Quarter":df["Quarter"].unique(),
        "Aircraft Manufacturer":df["Aircraft Manufacturer"].unique(),
        "Operator":df["Operator"].unique(),
        "Month":df["Month"].unique(),
        "Year":df["Year"].unique(),
        "Decade":df["Decade"].unique(),
        "Season":df["Decade"].unique()
        }


        

#storing user selection 
selected_filters={}

#generating multiselect widget dynmatically 
for key,options in filters.items():
    selected_filters[key] = st.sidebar.multiselect(f"Filter by {key}", options)

filtered_df= df.copy()

for key,selected_values in selected_filters.items():
    if selected_values:
        filtered_df= filtered_df[filtered_df[key].isin(selected_values)]
st.dataframe(filtered_df, height=300, width=1300)


no_of_reported_cases = len(filtered_df)
total_deaths = filtered_df["Fatalities (air)"].sum()
number_of_countries_affected = filtered_df["Country/Region"].nunique()
number_of_people_aboard = filtered_df["Aboard"].sum()
number_of_people_survived = filtered_df["Survivors"].sum()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Number of Reported Cases", no_of_reported_cases)

with col2:
    st.metric("Total Fatalities", total_deaths)

with col3:
    st.metric("Countries Affected", number_of_countries_affected)

with col4:
    st.metric("Number of People Aboard", number_of_people_aboard)
with col5:
    st.metric("Number of People Survived", number_of_people_survived)

#tables 
#countries with the highest reported cases 
st.subheader("Top 5 countries with the Highest Air crash recording")

top_countries = (df['Country/Region'].value_counts().head(6).reset_index())
top_countries.columns = ['Country', 'Reported Cases']
range = top_countries.iloc[1:7]
st.table(range)

# aircraft manufacturer with the most aircrashes
st.subheader("Aircraft Manufacturers with the most Recorded cases")
top_aircraft_manufacturer = (df["Aircraft Manufacturer"].value_counts().head(5).reset_index())
top_aircraft_manufacturer.columns = ["Aircraft Manufacturer", "number of crashes from manufacturer"]
st.table(top_aircraft_manufacturer)

# operator with the most air crashes 
st.subheader("Aircraft Operators with the most crashes")
top_operator = (df.Operator.value_counts().head(5).reset_index())
top_operator.columns = ["Operator", "Number of Crashes from operator"]
st.table(top_operator)

# Aircraft with th most aircrashes 
st.subheader("Aircraft with the most air crashes")
top_aircrashes = (df.Aircraft.value_counts().head(5).reset_index())
top_aircrashes.columns = ["Aircraft", "Number of crashes from Aircraft"]
st.table(top_aircrashes)

# plots and graphs 

# plot for the reported number of cases per decade 
decade_counts = df['Decade'].value_counts().sort_index().reset_index()
decade_counts.columns = ['Decade', 'Count']
chart = alt.Chart(decade_counts).mark_line(point=True).encode(
    x=alt.X('Decade:N', title='Decade'),  # N = Nominal (category)
    y=alt.Y('Count:Q', title='Count'),    # Q = Quantitative (numeric)
    color=alt.value('red')              # make the line black
).properties(
    title='Numbers of Reported Aircrashes by Decade',
    width=600,
    height=400
)
st.altair_chart(chart)

# chart for Aircrashes by seasons
st.subheader("Aircrashes By Seasons")
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X('Season:N', sort='ascending', title='Season'),
    y=alt.Y('count():Q', title='Count'),
    color=alt.value('orange')  # Optional: make bars orange
).properties(height=600)
st.altair_chart(chart)


# chart for Aircrashes by Months 
df['Month'] = pd.to_datetime(df['Date']).dt.month
st.subheader("Aircrahses By Months")
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X('Month:N', sort='ascending', title='Month'),
    y=alt.Y('count():Q', title='Count'),
    color=alt.value('orange')  # Optional: make bars orange
).properties(height=600)
st.altair_chart(chart)
df.drop(columns=['Month'], inplace=True)
# pie chart for fatalities and shit 
# Pie Chart
bin_counts = df['Fatality (Grouping)'].value_counts()

st.subheader("Groupings of Fatalities (Total, Partial Fatalities, and Total Survival)")
plt.figure(figsize=(6,6))
plt.pie(
    bin_counts.values, 
    labels=bin_counts.index,  
    colors=['red', 'lightgreen', 'blue'],
    autopct='%1.1f%%', 
    wedgeprops={'edgecolor': 'black'}
)
plt.title('AIR CRASHES FATALITY RATE')
st.pyplot(plt)


# Heat map for year and country 
df['Year'] = pd.to_datetime(df['Date']).dt.year
heatmap = df.groupby(['Year', 'Country/Region']).size().reset_index(name='Crash_Count')
chart = alt.Chart(heatmap).mark_rect().encode(
    x=alt.X('Year:O', title='Decade'),        
    y=alt.Y('Country/Region:N', title='Country'),
    color=alt.Color('Crash_Count:Q', scale=alt.Scale(scheme='reds'), title='Crash Count'),
    tooltip=['Year', 'Country/Region', 'Crash_Count']
).properties(
    width=600,
    height=400,
    title='Heatmap of Aircrashes by Year and Country'
)

st.altair_chart(chart)


