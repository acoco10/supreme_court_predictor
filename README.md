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

## Exploratory Data Analysis Findings

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/outcome_direction.png)

**Throughout its History, in close cases, the SC has been fairly balanced in terms of the idealogies of its outcomes**

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/wordswinning.png)
![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/wordslosing.png)

**Apparently the Supreme Court doesnt like New ideas!** But seriously, there is an alarming amount of similarities between the two classes. It appears lawyers at the highest level use mostly the same words. 

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/lcdisposition.png)

While there is little difference between the disposition of the lower court ruling for winning petitioner arguments, it is clear that losing petitioner arguments are more likely to be conservative lower court outcomes, in other words, **the Supreme Court is more likely to uphold liberal lower court decisions.**

## Data 

The oral argument transcript data for this project was acquired on github and scraped from oyez.com. Many thanks go out to [Eric Wiener](https://github.com/EricWiener) for sharing his repo of scraped arguments. Additional information about the case, including the target variable of whether or not the petitioner won the case was found on [Washington University’s excellent Supreme Court Database](http://scdb.wustl.edu/). Professor Harold Spaeth and his team work are heros for working tirelessly to maintain and verify the accuracy of the SCDB. In the end about 3000 ‘close’ cases were used from 1946 until today. Close was defined as any case where the case was decided by a margin of 6-3 or less. 

## Data Cleaning

For cleaning the transcripts, I used JSON to break them down by speaker then filtered out only the petitioner words. This was an imperfect process as I could not find a complete list of the lawyers for each side, so the petitioner is just the first lawyer to speak even if they may have had multiple speakers or advocates. The data was then tokenized and lemmatized. 

## Modeling

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/petitioner.png)
**There is a slight class imbalance within the data**

I used TF-IDF scores in a sparse matrix to vectorize the transcripts. I first tried a random forest classifier(RCF) as a baseline model, achieving an accuracy of .58%. If you guessed that the petitioner would win every case you would be right more often that this model. A naive bayes classifier and a support vector model barely did better achieving accuracy scores of .5803 and .59 respectively. None of the models based solely on the text were very accurate and you would be better off just guessing that the petitioner won. However I had hope as they all achieve F1 scores around .7, indicating that they do have some ability to distinguish between the classes based on TFIDF scores. 

After finding little advantage in using NLP, I turned to using categorical factors similar to other researchers. I used a logistic regression as a base line and achieved only 49% accuracy, far worse than guessing the dominant class. With an RFC model I managed an accuracy score of 63% better than any model before and legal expert predictions, but not great.

The breakthrough came when I found a synergistic effect between the text data and the categorical data. By taking my predictions from the NLP based naive bayes classifier and adding them as a feature in my RFC model I was able to reach a cross validated accuracy score of 77% , better than any of the models I found in my research and far better than the expert predictions referenced in Ruger et al.’s work. 

## Conclusions 

While NLP is not a silver bullet for legal issues, when used in conjunction with other factors, it can be an invaluable asset and lead to highly accurate predictions for close Supreme Court case. It seems like there must be information about the case in the transcript of the arguments that categorical variables do not pick up on and vice versa.  My model would be suitable as evidence in a prediction on a Supreme Court case that is expected to be close, especially when the legal experts are generally inaccurate. Organizations can use my model to make accurate predictions about some of the issues that will shape our country. While I would never recommend it be used for legal decisions, hopefully it can have some use in helping predict contentious cases that change people’s lives. 

## Next Steps 

1. Tailor my model for this court and make predictions on cases that have recently been argued.
2. Implement deep learning to further learn the relationship between oral argument and the outcome of the case.
3. Further segment the transcripts into each justice’s words and make a predictor for how each individual justice will vote 
4. See if DOC2VEC can further improve my model since the transcripts are long and may be hard to quantify solely on TF-IDF scores. 

## Repository Structure
#### Notebooks
  - Data cleaning
  - EDA
  - Modeling1 (NLP)
  - Modeling2 (non NLP and combination model)
#### Data
  - Transcript Data
  - SCDB CSV
  - Text Data which was extracted for the justices and the petitioners in the data cleaning notebook
#### Images
  - graphs from the EDA which I used in my presentation and readme 
 
 
 ## Presentation Slides for this Project
 
 https://docs.google.com/presentation/d/1QUVTNoiegmGnHzAMduNHYqL7gzv7whqTw4LTw02laC4/edit?usp=sharing

