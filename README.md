# Vaccine-Adoption-to-Prevent-Covid-Outbreaks
<br/>
The Delta COVID variant is on the rise in the U.S. The biggest weapon we have in our prevention aresenal is the vaccine. Other mitigation strategies are mask-wearing, hand-washing, and avoid crowds, especially indoors. We have a broad national vaccine rollout strategy, but regional influences may be a dominating factor in a strong rollout and creating herd immunity. The problem I am trying to solve is WHERE outbreaks are likely to occur, and what features of a geographic region make those outbreaks more likely.

By looking at county demographic features such as education, income, age, gender, race, religion, political affiliation, and other social factors and behaviors, health professionals can develop strategies for overcoming vaccine hesitancy with the goal of developing local herd immunity. Preventing COVID outbreaks is the best way to stimulate local economies, and vaccination is key to that effort.

The data sets used for this research effort are publically available through the following organizations:
- US Census Data by County: Age / Gender / Household / Public Transportation / 
- US Political Voting Data by County: MIT Election Lab: Voted GOP / Voted Democrat 
- Religious Data by County: Association of Religion Data Archives
- Covid / Vaccine Data by County: CDC
Common key between databases: FIPS County Code

### Data description
My target variable is the rate of change (Target_PctChg) in covid rates from June 11th to July 11th of this year. This is a supervised learning model.
My feature variables are:

      ,County      ,State      ,AgeUnder15      ,Age15to24      ,Age25to34      ,Age35to44      ,Age45to64      ,Age65to84      ,asAgeOver85      ,Married_Family
      ,Avg_HH_Size      ,Avg_Fam_Size      ,High_School      ,BS      ,Veterans      ,Born_in_State      ,Own_Comp      ,Broadband_Access      ,Mean_HHI
      ,Median_HHI      ,Unempl_Rate      ,maskLow      ,maskHigh      ,DensPerSqMile      ,DominantReligion      ,Religiosity      ,Region
      ,vaccineHest      ,CVAC_level_of_concern_for_vaccination_rollout      ,Social_Vulnerability_Index_SVI      ,per_dem      ,per_gop
    
    Categorical variables County / State were removed from the feature set
    Categorical variables DominantReligion / Region were one-hot-encoded
    All data was scaled

<b>Tools Used in this Analysis</b><br/>
SQL<br/>
Spreadsheets<br/>
Python<br/>
Jupyter Notebooks<br/>
Google CoLab<br/>

<b>Models Used in this Analysis</b><br/>
Regession

<b>Libraries Used</b><br/>
Numpy<br/>
Pandas<br/>
statsmodels<br/>
matplotlib<br/>
sklearn<br/>
collections

<b>models used</b><br/>
LinearRegression (with KFold)<br/>
DecisionTreeRegressor (with KFold / max_depth tuning)<br/>
RandomForestRegressor (with RandomizedSearchCV)<br/>
PCA 

I used PCA and r2_score to identify the features that contributed most of the variance in the model. Not surprisingly, vaccine hesitancy, CVAC_level_of_concern, region=South, and religion=Evangelical are the most important features informing the model. Unfortunately, when I reduced features, the MSE went up, so each feature is contributing to the model.

Links to Data / Graphs / Src<br/>
https://github.com/rathbird/Vaccine-Hesitancy-Driving-Covid-Outbreaks/tree/main/data<br/>
https://github.com/rathbird/Vaccine-Hesitancy-Driving-Covid-Outbreaks/tree/main/img<br/>
https://github.com/rathbird/Vaccine-Hesitancy-Driving-Covid-Outbreaks/tree/main/src<br/>



