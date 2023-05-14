# A Cluster Analysis Of Composite Indicators In Technology Readiness Across Countries

Author: [Camilo Vargas](https://www.github.com/cvas91)

[Click Here to Access Python Code](https://github.com/cvas91/Composite_Indicators/blob/main/cluster-analysis-of-composite-indicators-pca-fa.ipynb)

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
