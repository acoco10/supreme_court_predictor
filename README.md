# Using NLP and Machine Learning to Predict the Outcome of Close SC Cases
![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/Scotusblogheader.png##) 

## Business problem

[SCOTUSblog](https://www.scotusblog.com/), an award winning publication that follows the Supreme Court, wants to expand its coverage of oral arguments. They want to focus on close cases which are [notoriously hard to predict](https://www.jstor.org/stable/4099370?seq=1). Natural Language Processing is a growing field of study for law and the SCOTUS blog wants to use language data from the transcripts of the oral arguments to create their own proprietary model. Past Machine learning models make predictions based on the factors of the case and justice behaviour. SCOTUSblog will use the model to predict the outcomes of upcoming cases. There can be a long time period inbetween when oral arguments are heard and SC opinions are published, posting predictions that are reliably accurate using the model during this time will draw new traffic to their site and enhance their commentary on ongoing cases in general. They also requested some exploratory data analysis (EDA) on the language and other factors that predict the outcome of Supreme Court cases that they can use for an initial blog post to introduce the model. 

Research has shown that legal experts only predict the outcome of a SC case right about [60% of the time](https://www.jstor.org/stable/4099370?seq=1). While much of the SCOTUS blog focuses on commentary, they have realized that expert opinion is not enough for predicting future cases. Other Machine learning models that make predictions based on the case factors, such as who won at a lower court and what the issue being argued was get it right about 70% of the time. SCOTUS blog wants their model to build on these models  by being tailored specifically towards close cases and leveraging  NLP in addition to information about the case. 


![sc_drawing](http://www.scotusdaily.com/wp-content/uploads/2017/12/xSC170626wide.jpg)
[credit to the incredible courtroom artist Art Lien](https://courtartist.com/)

## Supreme Court 101

The Supreme Court is the highest court in the United States. They hear appeals from the federal courts, as well as state supreme courts, and also hear certain cases in the first instance (for example, when one state sues another).. The Supreme Court is tasked with interpreting the Constitution and federal law, and their interpretations are binding on lower federal courts and state courts. Most of the cases the Supreme Court hears are appeals, meaning that one party is unhappy with a court ruling and wants to reargue their case at a higher level. The Supreme Court has the authority to decide which cases it wants to hear appeals in, and only hears a small number of cases each year compared to other courts. 

## Glossary 

*Appeal* - to apply to a higher court for the reversal of a decision

*Petitioner* -  The party that lost at the lower level and is attempting to appeal the case 

*Respondent*- The party that won at the lower level and wants that decision reaffirmed

*Liberal/ conservative outcome*- [Defined as follows in the SCDB](http://scdb.wustl.edu/), "In order to determine whether the Court supports or opposes the issue to which the case pertains, this variable codes the ideological "direction" of the decision. Specification of direction comports with conventional usage for the most part except for the interstate relations, private law, and the miscellaneous issues." **Note** that there is no consensus as to what exactly constitutes a liberal or conservative decision, follow the link to read the specificities of the SCDB definition. 

*Justice* - One of the nine judges on the Supreme Court who vote on whether or not the case should be reversed or upheld. Justices are nominated by the president and the Senate votes whether or not to confirm them. They have life tenure, so seats are only open when a current justice steps down or dies. 

*Chief Justice* - Has some additional authority, and is the administrator of many court proceedings but has no additional voting powers compared to the other justices. Appointed whenever there is an open Chief Justice seat in the same manner as the other justices, and not by seniority or a vote by the SC.

*Oral argument* - When the lawyers from both sides argue their case in front of the nine justices and are questioned by them. Oral arguments take place between October and April. The petitioner always goes first. Each side gets 30 minutes. The petitioner is allowed a five minute rebuttal at the end of argument, but still may only speak for 30 minutes total. 

*Opinion* - the written decision of the court, which is usually written by one justice on the winning side. Justices on either side may submit concurring or dissenting opinions, but they do not affect the overall outcome of the vote. Opinions typically take several months to write and are published as they are completed.

## Data 

The oral argument transcript data for this project was acquired on github and scraped from oyez.com. Many thanks go out to [Eric Wiener](https://github.com/EricWiener) for sharing his repo of scraped arguments. Additional information about the case, including the target variable of whether or not the petitioner won the case was found on [Washington University’s excellent Supreme Court Database](http://scdb.wustl.edu/). Professor Harold Spaeth and his team work are heros for working tirelessly to maintain and verify the accuracy of the SCDB. In the end about 3000 ‘close’ cases were used from 1946 until today. Close was defined as any case where the case was decided by a margin of 6-3 or less. 

### Cleaning

For cleaning the transcripts, I used JSON to break them down by speaker then filtered out only the petitioner words. This was an imperfect process as I could not find a complete list of the lawyers for each side, so the petitioner is just the first lawyer to speak even if they may have had multiple speakers or advocates. The data was then tokenized and stop words were removed. In addition to the english set of stop words I imported from the NLTK corpus I used the 20 most common words found in my data and some of the pleasentries used in every case, such as, "may it please the court."

### Variables Used for Modeling

*Lemmatized Words* - I used the Natural Language Toolkit (NLTK) to reduce each word to its base. An example would be removing ing from running to make it run. This makes it easier for models to percieve the possible differences between words. 

*TFIDF Scores* - I used NLTK to calculate TFIDF scores for each word that appeared in the petitioners argument or said to the petitioner for each case.

*Lower Court Disposition* - From the SCDB whether the outcome of the case was liberal or conservative as explained in the glossary. 

*Natural Court* - Included in the SCDB, "A natural court is a period during which no personnel change occurs." I made a dummy variable for each of the 34 natural courts present in my dataset. 

*Issue* - The legal issue the case revolves around, eg. search and siezure. I made all 249 issues present in my data set into dummy variables. 

*Jurisdiction* - There are 15 different ways cases can reach the SC for consideration. The most well known way is on appeal, but another example is cases like inter state disputes can originate in the SC. I made a dummy variable for each manner. 

*Case Source* - The Lower Court where the case orignated. There are 600 different options so I just let the Random Forest Model handle this category and did not include it in my logistic regression models. 

*Petitioner* - There are different types of petitioners, such as an international party or the attorney general. The SCDB includes 600 different options so I also kept this one as is and only used it for more advanced models that could differentiate between the categories on their own. 

*Winning Party* - My target variable, who recieved a favorable outcome in the case? Either the petitioner or the respondent. I dropped cases where a winner was unclear. 

## Exploratory Data Analysis Findings

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/outcome_direction.png)

**Throughout its History, in close cases, the SC has been fairly balanced in terms of the idealogies of its outcomes** Liberal outcomes were slightly more likely

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/wordswinning.png)
![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/wordslosing.png)

**Apparently the Supreme Court doesnt like New ideas!** But seriously, there is an alarming amount of similarities between the two classes. It appears lawyers at the highest level use mostly the same words. 

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/lcdisposition.png)

While there is little difference between the disposition of the lower court ruling for winning petitioner arguments, it is clear that losing petitioner arguments are more likely to be conservative lower court outcomes, in other words, **the Supreme Court is more likely to uphold liberal lower court decisions.**



## Modeling

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/petitioner.png)
**There is a slight class imbalance within the data** The Petitioner wins in close cases 20% more than the respondent

I used TF-IDF scores in a sparse matrix to vectorize the transcripts. I first tried a random forest classifier(RCF) as a baseline model, achieving an accuracy of .58%. If you guessed that the petitioner would win every case you would be right more often that this model. A naive bayes classifier and a support vector model barely did better achieving accuracy scores of .5803 and .59 respectively. None of the models based solely on the text were very accurate and you would be better off just guessing that the petitioner won. However I had hope as they all achieve F1 scores around .7, indicating that they do have some ability to distinguish between the classes based on TFIDF scores. Worryingly, every model I tried was extremely overfit. After trying to reduce variance with a cross validated grid search, the Random Forest Classifier that resulted always predicted that the petitioner would win. 

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/most%20important%20features.png)
**Every strongly predictive factor was an issue besides the case being heard from CJ Roberts  3rd Natural Court** Interestingly, this court only had 8 justices on it until August when President Obama appointed Justice Sotomayor. Additionally [https://www.oyez.org/courts](there were 7 Republicans and only 1 Democrat on this court) which may have contributed. 

after finding little advantage in using NLP, I turned to using categorical factors similar to other researchers. I used a logistic regression as a base line and achieved only 59% accuracy about the same as my poor NLP models. With an RFC model I managed an accuracy score of 63% better than any model before and better than legal expert predictions, but not great.

In a last attempt to implement some sort of NLP analysis, I ran the Vader Sentiment analysis package on justice questions and used that in combinations with my categorical RFC. This actually lead to worse results however and lowered the accuracy to .57.


## Conclusions 

Categorical models offer far more potential with less work, unfortunately I devoted alot of this project to NLP whish is mostly a deadend. My reccomendation would be to implement this model in the short term as 63% accurate on close cases is better than what experts can predict on all cases. Other machine learning models have had much more success then mine, so I think some significant feature engineering could lead to more promising results. My EDA would make for an interesting blog post but my model needs far more work to be useful for SCOTUSblog. There was just not significant language differences between the losing and the winning arguments.  

## Next Steps 

1. Tailor my model for this court and make predictions on cases that have recently been argued.
2. Implement deep learning to further learn the relationship between oral argument and the outcome of the case.
3. Further segment the transcripts into each justice’s words and make a predictor for how each individual justice will vote 
4. See if DOC2VEC can further improve my model since the transcripts are long and may be hard to quantify solely on TF-IDF scores. 

## Repository Structure
<pre>
├── data
│   ├── Final_justice.csv
│   ├── Final_merge.csv
│   ├── justices.csv
│   ├── SCDB_2020_01_caseCentered_Citation.csv
│   └──supreme-court-cases
│      ├── cases
│      ├── justices.js
│      ├── README.md
├── images
│   ├── common_issues.png
│   ├── lcdisposition.png
│   ├── most_common.png
│   ├── outcome_direction.png
│   ├── petioner.png
│   ├── scdisposition.png
│   ├── Scotusblogheader.png
│   ├── wordslosing.png
│   └── wordswinning.png
├── notebooks
|   ├── _pycache_   
│   ├── SC_Predictor_Data_Cleaning_Notebook.ipynb   
│   ├── SC_Predictor_EDA_notebook.ipynb
│   ├── SC_Predictor_Modeling_Notebook1.ipynb
│   ├── SC_Predictor_Modeling_Notebook2.ipynb
└── src.py
</pre>

 
 ## Presentation Slides for this Project
 
 https://docs.google.com/presentation/d/1QUVTNoiegmGnHzAMduNHYqL7gzv7whqTw4LTw02laC4/edit?usp=sharing

