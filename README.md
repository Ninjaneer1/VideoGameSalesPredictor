# VideoGameSalesPredictor
Machine learning model that attempts predicting sales based on descriptors and ratings.

In this notebook, I am aiming to create a machine learning model that predicts sales of video games within a reasonable accuracy
using some descriptors along with critic and user data. 

The data used in this notebook was found on Kaggle, at: 
  https://www.kaggle.com/kendallgillies/video-game-sales-and-ratings
  
There is only critic and user data for about half of the available video game data, but this still leaves a lot to work with.

The models did best when attemping to predict the global sales. There are a lot outliers in the sales data and the 
distribution are very non-normal The sales data is log-transformed to help gather it into a managable range, and
from there, the correlations with the critic and user data can be found. Multicolinearlity is addressed using VIF.
Scaling is performed to the numeric data prior to fiting the models. The categorical data that I found to be the
most useful was reduced to 50 unique values by re-bagging, or renaming the least frequent elements to  "other".

Tools Used:
 - From scikit-learn I used XGBoost Regressor, Lasso and ElasticNet Regressors, and a KNN Regressor. 
   - Preprocessing and organization is handled with StandardScaler, and ColumnTransformer and Pipeline.
   - GridSeachCV is used for iteration and parameter tuning.  
 - Seaborn is the main library used for visuals. 
 - Pandas and Numpy handle my data frames and computation. 

Future Goals:
 - I believe that experimenting futher with SVM might be fruitfull. 
 - Using other methods for dimension reduction and primary component anylsis could be implemented. 
 - Seeking other data sources to merge with this set.
 - Find other ways to prepare data, futher tune models, implement other models, or gather other insights.
 


