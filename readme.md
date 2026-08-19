# MovieIQ — Predictive Analytics on Film Success

MovieIQ is an interactive Streamlit dashboard that analyzes movie data and predicts whether a movie is likely to be financially successful.

The project combines Exploratory Data Analysis (EDA), Statistical Testing, and Machine Learning to understand the factors associated with movie success and provide a simple prediction tool.

### Project Overview

A movie is considered successful when its revenue is greater than its budget.

Success = 1 → Revenue > Budget
Success = 0 → Revenue ≤ Budget

A Random Forest Classifier is used to predict movie success based on selected movie features.

### Dashboard Features

1. Interactive Filters

The dashboard provides filters in the sidebar that allow users to explore the movie dataset.

Users can filter movies by:

Genre
Minimum Vote Average

The dashboard visuals and statistics update based on the selected filters.

2. Dataset Overview

The dashboard provides a quick summary of the currently filtered movies.

It displays:

Total number of movies
Number of successful movies
Success rate
Average vote average

This gives users a quick overview of the selected movies before exploring the detailed analysis.

3. Exploratory Data Analysis

MovieIQ includes the key visualizations created during the EDA stage.

Budget vs Revenue

A scatter plot showing the relationship between a movie's budget and its revenue.

Genre Success Rate

A comparison of success rates across different movie genres.

Popularity vs Success

A box plot comparing the popularity of successful and unsuccessful movies.

Runtime vs Success

A box plot comparing the runtime of successful and unsuccessful movies.

Vote Average vs Success

A box plot comparing the vote average of successful and unsuccessful movies.

Correlation Heatmap

A heatmap showing the correlations between the main numerical features in the dataset.

4. Statistical Testing

The dashboard displays the results of the statistical tests performed during the project.

T-Test

An independent two-sample T-Test is used to examine whether the average popularity of successful movies is significantly different from unsuccessful movies.

The dashboard displays:

T-Statistic
P-Value
Statistical conclusion
Chi-Square Test

A Chi-Square test is used to examine whether movie genre and movie success are associated with each other.

The dashboard displays:

Chi-Square Statistic
Degrees of Freedom
P-Value
Statistical conclusion

A significance level of 0.05 is used for the statistical tests.

5. Movie Success Prediction

MovieIQ includes an interactive prediction section where users can enter movie details.

The inputs include:

Budget
Popularity
Runtime
Vote Average
Genre

The Random Forest model then predicts whether the movie is:

Successful

or:

Not Successful

The dashboard also displays the estimated probability of success.

6. Random Forest Classifier

MovieIQ uses a Random Forest Classifier for movie success prediction.

The model uses:

Budget
Popularity
Runtime
Vote Average
Genre information

The target variable is:

Success

Revenue is not used as a model feature because revenue is used to create the success variable:

Success = 1 if Revenue > Budget

Using revenue as an input feature would result in data leakage.

The dashboard displays the following model performance metrics:

Accuracy
Precision
Recall
Project Workflow
Data Preparation
        ↓
Exploratory Data Analysis
        ↓
Statistical Testing
        ↓
Random Forest Classification
        ↓
Streamlit Dashboard
Dataset

The dataset contains movie-related information including:

Movie title
Budget
Revenue
Popularity
Runtime
Vote Average
Genres

The data is prepared and analyzed using Python before being used in the dashboard and machine learning model.

Technologies Used
Technology	Purpose
Python	Data analysis and machine learning
Pandas	Data manipulation
NumPy	Numerical operations
Matplotlib	Data visualization
Seaborn	Statistical visualizations
SciPy	Statistical testing
Scikit-learn	Machine learning
Streamlit	Interactive dashboard



**Model Evaluation**

The Random Forest model is evaluated using three metrics:

Accuracy

Measures the overall percentage of correct predictions.

Precision

Measures how many movies predicted as successful were actually successful.

Recall

Measures how many of the actual successful movies were correctly identified.

Using multiple evaluation metrics provides a better understanding of model performance than relying only on accuracy.

Limitations

Movie success can depend on many factors that are not available in the dataset.

Some factors that could influence movie success include:

Marketing and promotion
Cast and crew
Director
Production quality
Release strategy
Audience preferences
Competition from other movies

Since these factors are not included in the dataset, the model's prediction should be treated as an analytical estimate, not a guarantee of financial success.

**Future Improvements**

The project could be improved by:

Adding more movie-related features
Including cast and director information
Including marketing data
Testing additional machine learning models
Improving model hyperparameter tuning
Adding more interactive dashboard features
Deploying the dashboard online

Author

Avin Dcosta

BCA Graduate | Data Analytics

Skills Used

Python · Pandas · NumPy · SQL · Power BI · Machine Learning · Statistics · Streamlit

**Project Goal**

The goal of MovieIQ is to bring data analysis, statistical testing, and machine learning together in one interactive application.

Instead of viewing the analysis, statistical results, and machine learning model separately, MovieIQ combines them into a single dashboard that allows users to explore movie performance and predict potential movie success.