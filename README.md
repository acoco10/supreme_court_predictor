# supreme_court_predictor

Business problem:
	The Supreme Court decides on some of the most important legal and moral issues of our time but does so over multiple months after arguments are held. Furthermore, the Court's behaviour can be hard to predict, especially in cases that have a close outcome. This can leave populations affected by the courts decisions in total limbo. By leveraging natural language processing (NLP), this project aims to predict the outcome of close cases based on the words spoken during the exchange between the petitioner and the Justices during oral argument. This will provide legal prognosticators and savvy news organizations another tool for making a prediction about close cases before the opinion is published. Research has shown that legal experts only predict the outcome of a SC case right about 60% of the time. Other Machine learning models that make predictions based on the factors of the case and past justice behaviour get it right about 70% of the time. This project will build on those models by being tailored specifically towards close cases and leveraging  NLP in addition to information about the case. 

Data 

The oral argument transcript data for this project was acquired on github and scraped from oyez.com. Many thanks go out to Eric Wiener for sharing his repo of scraped arguments. Additional information about the case, including the target variable of whether or not the petitioner won the case was found on Washington University’s excellent Supreme Court Database. Professor Harold Spaeth and his team work tirelessly to maintain and verify the accuracy of SCDB. In the end about 3000 ‘close’ cases were used from 1946 until today. Close was defined as any case where the case was decided by a margin of 6-3 or less. 

Data Cleaning

For cleaning the transcripts, I used JSON to break them down by speaker then filtered out only the petitioner words. This was an imperfect process as I could not find a complete list of the lawyers for each side, so the petitioner is just the first lawyer to speak even if they may have had multiple speakers or advocates. The data was then tokenized and lemmatized. 

Modeling

I used TF-IDF scores in a sparse matrix to vectorize the transcripts. I first tried a random forest classifier(RCF) as a baseline model, achieving an accuracy of .58%. If you guessed that the petitioner would win every case you would be right more often that this model. A naive bayes classifier and a support vector model barely did better achieving accuracy scores of .5803 and .59 respectively. None of the models based solely on the text were very accurate and you would be better off just guessing that the petitioner won. However I had hope as they all achieve F1 scores around .7, indicating that they do have some ability to distinguish between the classes based on TFIDF scores. 

After finding little advantage in using NLP, I turned to using categorical factors similar to other researchers. I used a logistic regression as a base line and achieved only 49% accuracy, far worse than guessing the dominant class. With an RFC model I managed an accuracy score of 63% better than any model before and legal expert predictions, but not great.

The breakthrough came when I found a synergistic effect between the text data and the categorical data. By taking my predictions from the NLP based naive bayes classifier and adding them as a feature in my RFC model I was able to reach a cross validated accuracy score of 77% , better than any of the models I found in my research and far better than the expert predictions referenced in Ruger et al.’s work. 

Conclusions 

While NLP is a silver bullet for legal issues, when used in conjunction with other factors, it can be an invaluable asset and lead to highly accurate predictions for close Supreme Court case. It seems like there must be information about the case in the transcript of the arguments that categorical variables do not pick up on and vice versa.  My model would be suitable as evidence in a prediction on a Supreme Court case that is expected to be close, especially when the legal experts are generally inaccurate. Organizations can use my model to make accurate predictions about some of the issues that will shape our country. While I would never recommend it be used for legal decisions, hopefully it can have some use in helping predict contentious cases that change people’s lives. 

Next Steps 

Tailor my model for this court and make predictions on cases that have recently been argued.
Implement deep learning to further learn the relationship between oral argument and the outcome of the case.
Further segment the transcripts into each justice’s words and make a predictor for how each individual justice will vote 
See if DOC2VEC can further improve my model since the transcripts are long and may be hard to quantify solely on TF-IDF scores. 

