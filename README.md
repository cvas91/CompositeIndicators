# A Cluster Analysis Of Composite Indicators In Technology Readiness Across Countries

Author: [Camilo Vargas](https://www.github.com/cvas91)

[Click Here to Access Python Notebook](https://github.com/cvas91/Composite_Indicators/blob/main/cluster-analysis-of-composite-indicators-pca-fa.ipynb)

**Abstract:** 
- Technology readiness measures the capacity of a country to adapt and implement new technological developments based on its current capacity and resources. 
- Several organizations have developed composite indicators to rank countries’ technological performance. However, these methods imply a linear, equal comparison across all countries, which may not be a reliable tool for policymakers worldwide with different necessities and economic conditions. 
- Therefore, this project diverts from this standard simple version by implementing data mining techniques like Principal Components Analysis (PCA) and Factor Analysis (FA) by clustering countries with similar technical and economic conditions. 
- The statistical findings from the PCA method suggest that countries share similar components regardless of their inner conditions. However, FA suggests that technology factors vary across clusters; this means that countries could be diverse as it is expected to observe distinct composite indicators among clusters. 
- For further analysis and discussion, the full printed version of this project is available upon request.

### Motivation
- The last technological revolution has left many lingering concerns among policymakers and scholars about a country's technological capacity. 
- More specifically, if the effects of globalization could evolve in developing countries, or whether elite developed nations will continue to be the front-runners of technological advances and innovation. 
- This project will discuss the significant need to quantify and measure technology readiness at the country level through the aggregation of different technical dimensions.

### Hypothesis
By clustering a set of countries due to their technological and economic similarities, the null hypothesis will test if the composite indicator of each cluster is the same for all K groups of countries; in other words, 𝐻0: 𝐶𝐼(1) = 𝐶𝐼(2) = ⋯ = 𝐶𝐼(𝐾) ; 𝐻1: 𝑂𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒, where 𝐶𝐼(𝐾) is the composite indicator of cluster K. This means that if each composite indicator is the same, all countries would focus on the same factors, implying linearity and equally weighted comparisons across all nations. However, if the composite indicators are different, then each group of countries should focus on a personalized set of factors to enhance their technological deficit.

### Data selection
Data for this project will be cross-sectional for a reference timeline as of 2021 or the latest year available for the total population of 217 countries with 127 variables drawn from various international sources and databases, including those of the United Nations Educational, Scientific and Cultural Organization (UNESCO); the World Bank; the International Telecommunication Union (ITU); the World Economic Forum (WEF); the International Monetary Fund (IMF); the Organisation for Economic Co-operation and Development (OECD); the International Labour Organization (ILO) and other international organizations.

```python
# Load the required files for this project.

# Verify the current working directory, files must be posted in the same directory.
print(os.getcwd())

# Basic information about countries:
Demographics = pd.read_excel(r"/kaggle/input/0-demographics-wdi/0_Demographics_WDI.xlsx",na_values='..') #Replace the '..' as null values

# Exogenous Indexes from external organizations:
NRI_Index = pd.read_excel('/kaggle/input/1-nri-index/1_NRI_Index.xlsx')
EGOV_Index = pd.read_csv('/kaggle/input/1-egov-data-2022/1_EGOV_DATA_2022.csv') # This dataset does not have ISO3 codes
GII_Index = pd.read_excel('/kaggle/input/1-wipo-global-innovation-index/1_wipo_Global_Innovation_Index.xlsx')

# Indicators consolidated from several sources:
Indicators = pd.read_excel('/kaggle/input/2-indicators/2_Indicators.xlsx') # Dataset with 127 individual indicators

#The list of all countries recognized by the UN is accessible through the UN Stats to retrieve each country's 'ISO 3 alpha' code, which will be the key for relating to further data frames.
#url = 'https://unstats.un.org/unsd/methodology/m49/'
#ISOalpha3 = pd.read_html(requests.get(url).content)[0]
ISOalpha3 = pd.read_excel('/kaggle/input/0-isoalpha3/0_ISOalpha3.xlsx') 
```

### Cleaning Datasets
Beforehand the transformations and computation of the datasets, it is necessary to perform essential cleaning functions to ensure the data can be treated and manipulated easily.

1. Firstly, standardize the list of all countries recognized by the UN by retrieving their respective ISO 3 alphanumeric codes, which will be the key for associating each country to further data frames. Observations without an assigned code or demographic information could be dropped from the database.
2. Second, identify the indicators with missing values, which could be dropped if the missing data is considerably large, i.e. more than 20 percent; otherwise, they could be considered for upcoming imputation if the missing values are less than 20 percent of the sample.
3. Once the datasets and exogenous indexes are cleaned and reduced, they can be merged into a single dataset for further transformation and analysis.

```python
# Returns info and the non-null values for each loaded file.
Loaded = [ISOalpha3,Demographics,NRI_Index,EGOV_Index,GII_Index,Indicators]
LoadedName = ['ISOalpha3','Demographics','NRI_Index','EGOV_Index','GII_Index','Indicators']

for i, j in enumerate(Loaded, 0):
    print('\033[1m',f'Info:',LoadedName[i],'\033[0m')
    print(f'{j.info()}')
    
# In the Demographics data frame, insert a numeric code for each Region (continent).
# Create a Dictionary with the list of Regions and associate a number RegionNum.
RegionNames = Demographics['Region'].sort_values().unique().tolist()
RegionNum = dict(zip(RegionNames, range(100, 10000, 100)))
RegionNum

# Add a new column with the RegionNum for each country.
if 'RegionNum' in Demographics:
    Demographics
else: Demographics.insert(loc=2, column='RegionNum', value=(Demographics['Region'].map(RegionNum)))
Demographics

# Join the EGOV_Index data frame to their respective ISO-alpha3 code.
EGOV_Index3 = EGOV_Index.join(ISO_Alt.set_index('Country or Area')[['ISO-alpha3 code','M49 code']], on='Country Name')
EGOV_Index3

# Compute the percentage of missing values in each indicator:
percent_missing = Indicators.isnull().sum() * 100 / len(Indicators)
IndMissing = pd.DataFrame({'% Missing': percent_missing})

# Filter and sort only the indicators with more than 20% of missing values: 
IndDrop = IndMissing[IndMissing['% Missing'] >=20].sort_values('% Missing', ascending=False )
IndDrop

# Drop indicators with more than 20% of missing values:
if IndDrop.index.isin(Indicators.columns).all():
    Indicators.drop(IndDrop.index, axis=1, inplace=True) 
else: print(Indicators.shape)
```    

### Join Datasets
Joining all countries with their respective score from different exogenous Indexes. The remaining Indicators with less than 20% missing values will proceed to be imputed.

```python
# Countries from UN joined to the data frames: Demographics, NRI.score from NRI_Index, E-Government Index from EGOV_Index
Merge2 = ISOalpha3.join(
    Demographics.set_index('Country Code')[['RegionNum','Region','GDP_PC','Population']], on='ISO-alpha3 code').join(
    NRI_Index.set_index('ISO3Code')[['NRI.score']], on='ISO-alpha3 code').join(
    EGOV_Index3.set_index('ISO-alpha3 code')[['E-Government Index']], on='ISO-alpha3 code').join(
    GII_Index.set_index('ISO3')[['SCORE']], on='ISO-alpha3 code')
Merge2.rename(columns={'SCORE' : 'GII_Score'}, inplace = True)
Merge2

# In the recently joined data frame are small country estates without demographic data, so these can be dropped.
Merge2[Merge2['Population'].isnull().values] 

Merge2.dropna(subset=['Population'], inplace=True) # Drop data with null values in Demographic features.
Merge2

# Join the data frame with demographic and exogenous indexes to the data frame with individual indicators.
Merge3 = Merge2.join(
    Indicators.set_index('ISO3Code'), on='ISO-alpha3 code')
Merge3

Merge3[Merge3[['Country']].isnull().values] # Countries that did not join with the Indicators df.
Merge3.dropna(subset=['Country'], inplace=True) # Drop countries with null values in all Indicators.
Merge3
```

### Treatment and Imputation of Missing Data
Imputation of missing data and coverage at the country level will be included for those countries that might have missed information across the chosen indicators. Even though this could not be the most transparent and reliable approach to avoid the loss of vital information, given that the total population of countries is not large enough, it could be empirically valid to impute specific values for this project implementation.

The method chosen for imputation is known as the hot-deck imputation where each missing value is replaced with an average value from similar countries’ so-called donors based on data of other variables within the dataset. The **KNNImputer** algorithm is implemented to find the closest K neighbours (where K = 3) and impute a mean value for the missing data.

The remaining Countries with missing values will proceed to be imputed. Impute missing values for countries without a previous GDP_PC, Indexes or Indicators according to the country's neighbours.

```python
# Create a new df with the missing numerical values to be imputed:
ImpDem = pd.DataFrame(
    Merge3[Merge3.columns.drop(['Country or Area','ISO-alpha3 code','Region','Country'])]) # All columns except the ones with string values
ImpDem[ImpDem.isnull().values.any(axis=1)] # Returns only the missing values

# Imputing with KNNImputer K-Neighbor method:
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
ImpDemGdp = pd.DataFrame(imputer.fit_transform(ImpDem))
ImpDemGdp.columns =ImpDem.columns # Labels for the new imputed columns
round(ImpDemGdp,1)
```

#### Testing the imputation
The previously missing data can be compared with the imputed values to test the imputation procedure. To do so, generate boxplot graphs for any imputed features to compare the dispersion of data pre and post-imputation.

```python
# Create a subset of the imputed data frame with the exogenous indexes to be tested: 
ImpDemTest = ImpDemGdp[['M49 code','RegionNum','NRI.score','E-Government Index','GII_Score']].rename(columns=lambda x: x+'_Imp')
ImpDemTest

# Compare missing values to new imputed values. 
Dem2 = pd.merge(
    Merge3[['Country or Area','M49 code','ISO-alpha3 code','Region','NRI.score','E-Government Index','GII_Score']],
    ImpDemTest[['M49 code_Imp','NRI.score_Imp','E-Government Index_Imp','GII_Score_Imp']],
    left_on='M49 code', right_on='M49 code_Imp', how='left')
round(Dem2,1)

# Plotbox graph to compare the dispersion of E-Government Index pre and post-imputation.
Dem_EGov = Dem2[['Region','GII_Score','GII_Score_Imp']]
boxplot_EGov = Dem_EGov.boxplot(by='Region', figsize = (10,5), rot = 90)
```

![Figure 1: Imputation of missing values grouped by region.](https://github.com/cvas91/Composite_Indicators/blob/main/Figures/Screenshot%202023-05-14%20140728.png)

### Treatment of outliers and normalization
To evaluate the presence of outliers in the datasets can be measured through the skewness and kurtosis of each indicator. In the case of having no more than four outliers, they can be initially replaced by the smallest and largest values with the observations closest to them. This is done to limit the effect of abnormal extreme values, or outliers, on the dispersion of each indicator.

```python
Indicators1 = Master.iloc[:,10:] # Create a data frame with only the individual indicators.
# 
# The descriptive statistics will show the indicators with outlier values: 
Stats = Indicators1.describe().transpose()
Stats

StatsMax = Stats[Stats['max'] >100] # Verify features with values greater than 100
StatsMax

StatsMin = Stats[Stats['min'] <0] # Verify features with values less than 0
StatsMin

skew = Indicators1.skew()
Skewness = pd.DataFrame({'Skewness': abs(skew)})

# Filter only the indicators with more than an absolute value of 2.25 of Skewness: 
SkewAbs = Skewness[Skewness['Skewness'] >=2.25]
SkewAbs

kurtos = Indicators1.kurtosis()
Kurtosis = pd.DataFrame({'Kurtosis': abs(kurtos)})

# Filter only the indicators with more than an absolute value of 3.5 of kurtosis: 
KurtosAbs = Kurtosis[Kurtosis['Kurtosis'] >=3.5]
KurtosAbs

# Histogram plots of the indicators with kurtosis
Indicators1.hist(KurtosAbs.index,figsize=(30, 20))
plt.show()
```

![Figure 1.1: Histogram plots of the indicators with kurtosis](https://github.com/cvas91/Composite_Indicators/blob/main/Figures/Screenshot%202023-05-14%20141411.png)

### Clustering
Before starting the clustering process on the consolidated dataset, it is necessary to identify the optimal number of clusters in which the countries can be grouped according to their GDP per capita level and the mean of the exogenous indexes. To do so by applying the **K-Means method**, this algorithm creates a plot for the number of clusters on the x-axis and the total sum of squares errors (SSE) on the y-axis and then identifies where an “elbow” or bend appears indicating the optimal number of clusters on the x-axis for the k-means clustering algorithm.

```python
X = Master[['Mean Exo Indexes', 'GDP_PC_log']].values
np.random.seed(42)
inertia = []
distortions = []

for i in range(1, 10): # Iterating the process
    model = KMeans(n_clusters=i, n_init='auto')  # Instantiate the model
    model.fit(X)  # Fit The Model
    inertia.append(model.inertia_)  # Extract the error of the model
    distortions.append(sum(np.min(cdist(X, model.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) # Create a panel with 2 subplots (1x2)
plt.plot(range(1, 10), distortions, 'bx-') #Visualize the model
axs[0].plot(range(1, 10), inertia, 'bx-')
axs[0].set_title("Elbow Method Using Inertia")
axs[0].grid()
axs[1].plot(range(1, 10), distortions, 'bx-')
axs[1].set_title("Elbow Method Using Distortion")
axs[1].grid()
plt.suptitle("Optimal KMeans Clusters").set_y(1)
plt.show()

# Clustering with the KMeans method.

from sklearn.cluster import KMeans
np.random.seed(42) #Instantiate the model
y_pred = KMeans(n_clusters=3 ).fit_predict(X)

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_pred , cmap='Accent')
plt.title('Clusters')
plt.xlabel('Mean Exogenous Indexes')
plt.ylabel('GDP_PC_log')
plt.grid()
plt.show()
```

![Figure 3: Countries among 3 clusters.](https://github.com/cvas91/Composite_Indicators/blob/main/Figures/Screenshot%202023-05-14%20142603.png)

An alternative clustering method is through a Clustering Map, in which all individual indicators and countries will be clustered according to their inner variance.

As shown in the **Cluster Map**, the hierarchic dendrograms cluster countries through the vertical axis with similar income levels, i.e. dark gray for high-income countries, green for middle-income, and blue for low-income. Although some noise with irregular countries (Kuwait, Qatar, Saudi Arabia, UAE) is reported within each cluster.
In the same way, hierarchical dendrograms cluster indicators in the horizontal axis, where highly correlated indicators are grouped close to each other to form aggregate indicators.

```python
Indicators2 = Master.iloc[:,14:106] # Create a data frame with only the lastly imputed indicators.
Indicators2.set_index(Master['Country or Area'], inplace=True)

# Define colours for previously clustered countries
color_dict = dict(zip(np.unique(Master['ClusterNames']),np.array(['blue','green','grey'])))
target_df = Master[['Country or Area','ClusterNames']]
target_df['Clus'] = target_df['ClusterNames'].map(color_dict) 
target_colors = target_df[['Clus']]
target_colors.set_index(target_df['Country or Area'], inplace=True)

cgCountries = sns.clustermap(Indicators2, cmap ="YlGnBu", figsize=(30, 50), linewidths = 0.1, row_colors = target_colors);
plt.setp(cgCountries.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)
```

![Figure A1: Cluster Map across all countries and indicators.]()


