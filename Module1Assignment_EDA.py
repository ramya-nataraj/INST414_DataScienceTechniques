import pandas as pd
import matplotlib.pyplot as plt

tourismData = "Tourism_Hospitality_Industry_Analysis.csv"

df = pd.read_csv(tourismData)

# look at first five rows
# print(df.head())
        
# Frequency of visits by country
visitFreq = df.groupby("Country")["Number_of_Tourists"].sum()
# print(visitFreq)

# barplot
visitFreq.plot(kind= "bar")
plt.title("Frequency of Visits by Country")
plt.xlabel("Country")
plt.ylabel("Nmber of Tourists")
plt.show()
        
# Analze relationship between hotel occupency and availability
country_hotels = df.groupby("Country").agg({
    "Number_of_Hotels": "sum",
    "Hotel_Occupancy_Rate": "mean"})
print(country_hotels)

# Scatter plot
plt.scatter(country_hotels["Number_of_Hotels"], 
            country_hotels["Hotel_Occupancy_Rate"])
plt.xlabel("Number of Hotels")
plt.ylabel("Average Hotel Occupancy Rate (%)")
plt.title("Hotel Occupancy vs Number of Hotels by Country")

# ChatGPT Assistance: add country labels
for i, country in enumerate(country_hotels.index):
    plt.text(country_hotels["Number_of_Hotels"][i], 
             country_hotels["Hotel_Occupancy_Rate"][i], 
             country, fontsize=8)
plt.show() 

# Relationship between tourism revenue and contribution to GDP
tourismRevenue = df.groupby("Country")[["Tourism_Revenue_USD",
"Contribution_to_GDP_Percent"]].sum()
print(tourismRevenue)

# barplot
plt.scatter(df["Tourism_Revenue_USD"], df["Contribution_to_GDP_Percent"])
plt.title("Tourism Revenue vs. Contribution to GDP")
plt.xlabel("Tourism Revenue (USD)")
plt.ylabel("Contribution to GDP (%)")
plt.show()

# correlation between tourism revenue and countribution to GDP
correlation = df["Tourism_Revenue_USD"].corr(df["Contribution_to_GDP_Percent"])
print(f"""Correlation between Tourism Revenue and Contribution to GDP: 
      {correlation}""")

# Tourist spending and contribution to GDP
plt.scatter(df["Tourist_Spending_USD"], df["Contribution_to_GDP_Percent"])
plt.title("Tourist Spending vs. Contribution to GDP")
plt.xlabel("Tourist Spending (USD)")
plt.ylabel("Contribution to GDP (%)")
plt.show()
        
# Distribution of tourist spending by purpose of visit
spendingByPurpose = df.groupby("Purpose_of_Visit")["Tourist_Spending_USD"].sum()
print(spendingByPurpose)

# barplot
spendingByPurpose.plot(kind= "bar")
plt.title("Tourist Spending by Purpose of Visit")
plt.xlabel("Purpose of Visit")
plt.ylabel("Tourist Spending (USD)")
plt.show()