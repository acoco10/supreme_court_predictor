# Using NLP and Machine Learning to Predict the Outcome of Close SC Cases
![sc_drawing](http://www.scotusdaily.com/wp-content/uploads/2017/12/xSC170626wide.jpg)
## Business problem:

The Supreme Court decides on some of the most important legal and moral issues of our time and often shapes our nations approach to civil rights and liberties. Unfortunately the process for deliberating writing and releasing opinions can take [months](https://www.supremecourt.gov/about/procedures.aspx). Furthermore, the Court's behaviour can be hard to predict, especially in cases that have a close outcome. This can leave populations affected by the courts decisions in total limbo. By leveraging natural language processing (NLP), this project aims to predict the outcome of close cases based on the words spoken during the exchange between the petitioner and the Justices during oral argument. This will provide legal prognosticators and savvy news organizations another tool for making a prediction about close cases before the opinion is published. Research has shown that legal experts only predict the outcome of a SC case right about 60% of the [time](https://www.jstor.org/stable/4099370?seq=1). Other Machine learning models that make predictions based on the factors of the case and past justice behaviour are about [71% accurate](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174698#sec006). This project will build on previous machine learning efforts by **being tailored specifically towards close cases 
and leveraging  NLP in addition to information about the case.** 

## Exploratory Data Analysis Findings

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/Common-issues.png)

**These are the top ten issues argueed before the Supreme Court when the petioner recieved a favorable and outcome and when they did not.** The two classes are fairly similar, but some differences can be inferred. It seems petitioners do worse in Antitrust cases, double jeopardy cases right to council and federal taxation cases. The petitioner seemingly does better in Habeas Corpus, sufficiency of evidence, liability, Civil Rights Act liability and judicial review of administrative proccesses cases.

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/winningwords.png)
![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/wordslosing.png)

**Apparently the Supreme Court doesnt like New ideas!** But seriously, there is an alarming amount of similarities between the two classes. It appears lawyers at the highest level use mostly the same words. 

![img](https://github.com/acoco10/supreme_court_predictor/blob/main/images/lcdisposition.png)

While there is little difference between the disposition of the lower court ruling for winning petitioner arguments, it is clear that losing petitioner arguments are more likely to be conservative lower court outcomes, in other words, **the Supreme Court is more likely to uphold liberal lower court decisions.**

## Data 

The oral argument transcript data for this project was acquired on github and scraped from oyez.com. Many thanks go out to [Eric Wiener](https://github.com/EricWiener) for sharing his repo of scraped arguments. Additional information about the case, including the target variable of whether or not the petitioner won the case was found on [Washington University’s excellent Supreme Court Database](http://scdb.wustl.edu/). Professor Harold Spaeth and his team work are heros for working tirelessly to maintain and verify the accuracy of the SCDB. In the end about 3000 ‘close’ cases were used from 1946 until today. Close was defined as any case where the case was decided by a margin of 6-3 or less. 

## Data Cleaning

For cleaning the transcripts, I used JSON to break them down by speaker then filtered out only the petitioner words. This was an imperfect process as I could not find a complete list of the lawyers for each side, so the petitioner is just the first lawyer to speak even if they may have had multiple speakers or advocates. The data was then tokenized and lemmatized. 

## Modeling

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

