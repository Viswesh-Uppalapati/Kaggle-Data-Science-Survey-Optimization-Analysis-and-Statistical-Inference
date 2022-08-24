
#  Survey Optimization using Kaggle Users Survey Data

Author Contributions:

**Vineet Tallavajhala:** Introduction, 1, 2, and 4

**Viswesh Uppalapati:** Question 3, Advanced Analysis, and Conclusion

## Introduction

With the recent boom in data, there has been a vast growth of data enthusiasts throughout
the world. To understand the demographics of the growing data enthusiast community, Kaggle (a
community of data scientists and machine learning practitioners) creates annual surveys among
its users that track a plethora of data points. Some of these questions include the individual’s
favorite programming language, demographic information, work experience, work title, favorite
types of visualization tools, and many more similar questions that revolve around preferences in
technology. Within data science, the process of surveying is often critical in obtaining
information that is then used to conduct data science processes. Therefore, it is imperative that
surveys are both efficient and accurate to ensure that data science projects are produced to the
highest quality. Consequently, in this analysis we aimed to understand factors that are most
correlated with the duration an individual takes on a survey in order to optimize surveys for
future data science projects.

The main goal of our analysis was to present a narrative as to what might be driving
changes in duration time in hopes of optimizing future surveys. After creating our analysis, we
aimed to provide a proposal in order to maximize survey efficiency. To effectively craft our
proposal, we centralized our analysis of the Kaggle data science survey around the following
questions:

1. [ **Graphical Methods: Visualizing the Demographics of the Survey]:** Create
    visualizations to illustrate the demographics of individuals from the survey. What do
    these graphs indicate about a typical data scientist? How could the composition of these
    demographics introduce bias in future modeling?
2. [ **Comparing Duration Time Between different Subgroups of Kaggle Users]:** Visually
    or numerically compare the duration time between different subgroups of data scientists.
    Consider factors such as country, gender, salary, education, and occupation title. What do
    these comparisons between subgroups tell about duration time? Does there seem to be
    patterns between individuals taking more time than others?
3. [ **Point and Interval estimates, Bootstrap, and, KS-test** ]:Provide a point and interval
    estimate for the average amount of duration spent in completing the survey. Is the
    distribution of completion time normal? Adjust the data for normality and bootstrap if
    needed.
4. [ **Regression and R^2** ]: Fit regression lines to each feature within an encoded data set,
    attempting to predict the duration time of a survey. After creating different regression
    lines, find the R^2 to quantify the performance of each feature on predicting the duration
    time of a survey. Which features were most important in predicting the duration time of a
    survey? How do these individual models perform and why might this performance
    become worse with more information? Furthermore, why might duration time be difficult
    to predict?
5. [ **Advanced Analysis, Permutation Test]** : Do men and women perform differently in the
    completion of the survey? Specifically, is either group faster or slower than the other
    while completing the survey or are the distributions of completion times between the two
    groups relatively the same? What about when we consider the level of education, repeat
    the prior experiment with men and women that have a higher level of education.

## Data Processing

Prior to conducting our analysis, data cleaning and processing was necessary in order to
answer the questions presented above. The Kaggle dataset provided gave the duration time
column as a string, so we converted these values to integers in order to make intuitive
aggregations. Moreover, there were many null values within the dataset. It was decided that for
most survey questions these null values could be considered as an individual not selecting a
certain option. Therefore, these null values were converted to zeros in order to conduct analysis
in future steps. It was decided that outliers for duration time would not be excluded as these
values provide critical information into how long an individual may take on a survey. In other
words, if a survey time was abnormally short (less than 3 minutes) it could indicate that the data
scientist is clicking through questions. On the other hand if the survey time was abnormally
large, it could indicate that the person neglected the survey and came back to it after completing
other tasks. Both of these scenarios provide vital information for future modeling, so these data
points were maintained in the data set. Lastly, since much of the data is categorical (everything
besides the duration column), one hot encoding was required to effectively model the data set. In
later stages of the analysis, we one hot encoded each feature to achieve this modeling.

## Analysis and Inference

**1. Graphical Methods: Visualizing the Demographics of the Survey**

Prior to delving into analyzing duration times, we first investigated the demographics of
our survey to better understand the underlying data. The Kaggle survey data asks thirty five
different questions; the first six of which are demographic data regarding the individuals age,
gender, country, education, occupation title, programming experience, and also asks their income
level. To better understand the characteristics of individuals in the survey we decided to create
bar charts of each of these demographic questions provided below:

![pic 6](https://user-images.githubusercontent.com/50200188/186502551-7c4915dc-6b49-464a-a724-2f9b5e56715e.PNG)
![pic 7](https://user-images.githubusercontent.com/50200188/186502586-cef4c90c-46a3-4448-8d51-403579a5dd3c.PNG)

From these distributions a few observations can be made. For one, the data set primarily
consists of Master’s students/grads who make little to no money and are between 18-29 years
old. This makes sense since the data set comes from Kaggle which primarily consists of
individuals who are trying to bolster their portfolio in search of a job (which is what the
demographic above describes). With a vast majority of the data set coming from this
demographic profile, it is important to note that this could produce future bias in our model since
a typical survey is not conducted by what is typical for a data scientist. Even then, however, the
data set is insightful in terms of providing solutions to optimize data sets that revolve around
technologists since the demographic profile that represents a data scientist is similar to most that
work in technology.

**2. Subgroup Analysis: Comparing Duration Time Between Subgroups of Kaggle Users**

Having looked at the demographics of the survey data, the next step was to analyze how
these demographics were related to duration rates. To effectively understand the duration rates,
we decided to look at two demographic variables at a time, and compare the mean duration rate
for these two variables. For example, the mean duration rate for all female data scientists was
calculated. Similarly, this was conducted for each demographic using pivot tables. To better
understand the underlying results of these pivot tables, multiple bar charts were created which is
presented below:

![pic 8](https://user-images.githubusercontent.com/50200188/186502709-c2ef05cf-e9b3-4657-8818-2a9cd0dc4b43.PNG)
![pic 9](https://user-images.githubusercontent.com/50200188/186502724-05b62326-0fe2-4516-8cf4-8c18984b5be4.PNG)

From these graphs many observations can be made including that for all the different
demographic comparisons, women tend to have higher mean duration times. Furthermore,
among different education levels it does not seem like the duration time changes much for either
gender. Certain outliers can also be visualized in these graphs, for example the women from
Egypt have a duration time that creates a bar much larger than the rest of the countries. Lastly, it
also does not seem as though the salary of a data scientist affects the duration time in taking the
survey.

**3) Point and Interval Estimates for Survey Completion Time and Normality Analysis**

In order to get the point and interval estimates for the survey completion times, we first
looked at the observed distribution of the data. We graphed the completion times in a histogram
that is shown on the left below. From it, it is evident that the distribution is not normal and is
skewed to one side. If we were to calculate the estimates from the observed sample, then it is
likely that the estimates will not be representative of the true population, therefore, we
bootstrapped samples from the original distribution of completion times to get a more accurate
representation of the true population. We created a sample from the original distribution and
bootstrapped it. Then we calculated the mean of each of the bootstrapped samples and graphed
their distribution along with the normal distribution as shown in the graph on the right below.



From the visualization on the right, it is evident that the sampled bootstrap means of
completion time closely follow a normal distribution. Before we generated the corresponding
estimates of survey completion time, we wanted to verify statistically that the bootstrapped
means actually followed a normal distribution. To do so, we used the KS-test on the bootstrapped
sample and tested it against the normal distribution. The results of this test are shown below:

The KS-test on the bootstrapped estimates and the standard normal distribution resulted
in a p-value of around 0.541, which means that we fail to reject the null hypothesis that the
distribution of the bootstrapped estimates approximately follows a normal distribution. Now that
we have confirmed that the bootstrapped samples follow a normal distribution and don’t just
appear to do so in the graphs, we generate the point and interval estimates from the bootstrapped
estimates. The estimates are as follows:

The point estimate of the average time taken to complete the survey among the
participants came out to be around 8816.230 seconds with a standard error of 417.684 seconds.
The interval estimate for the average survey completion time is around (8337.204, 9974.526)
seconds. It is clear that the point estimate of mean completion time falls within the 95%
confidence interval generated. These estimates should be unbiased estimators of the true
population.

**4) Regression and R^2: Calculating the Effect of Questions on Duration Time**

With nearly thirty five questions that are all categorical, the data set becomes extremely
wide after encoding. Therefore, a model that trains on the entire data set has too much
information to effectively predict To better understand the effects of individual questions on
duration time, we decided to train individual regression lines for each feature and calculate R
squared values to determine how much the change induration could be attributed to a given
feature. The first step in creating the model was to one hot encode each question. The questions
in the data set with multiple parts were already one hot encoded, so the data was cleaned to fit
the encoded format (filling 1’s and 0’s where appropriate).After creating these different
regression lines and calculating the R^2 for each, we found the five most important features
(according to their R squared values) were:

1. Favorite Computer Vision Method is Generative Networks:  0.1
2. The individual is from the Republic of Korea: 0.08
3. Their favorite cloud computing platform is Microsoft Azure: 0.0599
4. Their income is between ninety and one hundred thousand dollars: 0.052
5. They have not completed any data science courses: 0.051

    Most of these values were somewhat unexpected besides the none option for data science
courses taken. This makes sense to affect the duration time since if someone were to click none,
it would take less time than having to go through all the data science courses and finding the
ones they have taken. Overall, these individual models did not perform super well in predicting
the duration time, but this behavior is somewhat expected. That is, there are many outside factors
that affect the duration time of a survey, so the model can only account for a certain amount of
the variation in duration time. Moreover, since we chose not to take out outliers during our
preprocessing of the data, the values from these outliers could be heavily affecting the results of
our model. The performance may get worse as we get more and more information since with too
many data points the model would both overfit and become random because patterns would be
harder to detect. Lastly, as mentioned previously, duration time itself is dependent on many
outside factors, so any model that does not overfit will come with large amounts of error.

**5) Adv. Analysis: Analyzing Differences in Completion Time between Men and Women**

For the advanced analysis part of the study, we wanted to look at how the distributions of
survey completion time changed based on the gender. To do so, we first grouped the data by the
gender and computed the mean and median survey completion times by gender. These statistics
are shown on the left and right below, respectively. Upon examining the mean and median
statistics by gender, it is evident that there is a massive discrepancy between the two statistics.
The mean completion time is much greater than the median completion time in every gender
group as seen in the charts below. This indicates a presence of massive data points, which may
be outliers, where some respondents take an unusually large amount of time to complete the
survey. To move forward with the advanced analysis, we looked at the distribution of the gender
of participants shown in the third chart below that is under the other two charts.

We found that over 98% of the participants of the survey are male and female. Therefore,
we continued the rest of the advanced analysis on these two genders. Due to the discrepancy
where a majority of the participants were male, we wanted to see whether either gender was
faster at completing the survey on average. To test this analytically, we performed a permutation
test detailed below:

_Question:_ Is there a significant difference between the distributions of completion times of males
and females?

_Test:_ Permutation test with the absolute difference of means and medians

_Null Hypothesis:_ The distributions of completion times of males and females come from the
same overall distribution.

_Alternative Hypothesis_ : There is a significant difference in survey completion times between the
two gender groups.


_Significance Level_ : 0.

_Results_ :

The result of the permutation test with the absolute difference in means came out to be a
p-value of 0.773, which is much above the significance level. Therefore, we fail to reject the null
hypothesis that the distributions of survey completion times between males and females come
from the same overall distribution. The result of the permutation test with the absolute difference
in median, however, came out to be a p-value of 0.0,which indicates that we reject the null
hypothesis in favor of the alternative hypothesis that there is a significant difference between the
two distributions. These results should be interpreted carefully. From the graph on the right
above, it is evident that the observed absolute difference in median was a significant deviation
from the rest of the trials in the permutation test. In addition, since the permutation test that used
the absolute difference in median favored the alternative hypothesis, further statistical tests
should be performed to verify the result as the result of the permutation test does not guarantee
the discrepancy, but does evidence it.

## Conclusion

Our goal for this project is to analyze the different factors that have an impact on the
survey completion times among different individuals from the data science community. In doing
so we hope to provide insight that is useful in optimizing surveys that are one of the most
popular methods of collecting user data for analysis. Before analyzing the duration time, we
wanted to investigate how the actual survey was distributed. From our analysis of the survey data
we found that a large majority of the data came from kagglers aged 18-29 who were Master’s
students with little income. Acknowledging that this was the main profile of the survey data, we
moved on to analyzing interactions between gender and other demographic features. From this
analysis we found that on average women tended to take longer in terms of duration time while
other factors like income status had little effect on duration time. Next we looked at the specific
point and interval estimates for the average survey completion times. To do so we bootstrapped
samples from the original data to generate samples that are closer to the average completion
times of the true population. In doing so, we found unbiased estimators for the average survey
completion times. Modeling the data by creating simple regressions for each feature, we
determined the individual effect of different questions on duration time**.** The results of our model
were somewhat random, as all the columns had low R squared values. However, this behavior
was expected since the vast amount of questions made it difficult to determine patterns in the
data without overfitting. Overall, we found that the most important features were primarily when
individuals selected none on columns since this takes less time than selecting other options.
Lastly, we wanted to look at how the distributions of survey completion times changed by
gender. For this, we mainly looked at the male and female participants of the survey. We found a
massive discrepancy between the mean and median survey completion times across every
gender. It seems that a presence of large outliers or a cluster of data points linked to respondents
taking an unusually long time to complete the survey caused this discrepancy. Upon performing
a permutation test, we found that there was a significant difference in the median survey
completion times of males and females. While the results of this report provide a deeper insight
into what factors affected the survey completion times among the Kaggle community. It is
limited due to the choice of the dataset. This is because the survey is given specifically to data
scientists and machine learning specialists that are on the Kaggle platform. Therefore, this data
does not provide any insight into other occupations or people that fill our surveys. In addition,
every survey is different and our analysis is only relevant to optimizing this specific survey,
while parts of it can be generalized. Regardless of the limitations of the study, these insights can
be generalized to optimize future surveys that are put out by Kaggle with a similar target
audience.


