#This program takes a training set of sentiment manually classified sentiment (1,0,-1)
#and uses a Naive Bayes Classifier to determine the sentiment of other tweets.
#Input: Training set of Tweets, Tweets that need Sentiment Analysis
#Output: Predicted Sentiment of Tweets in text file
import nltk
import csv

#define cleaning functions for training set
def get_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def get_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words
    
def take_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# Load Training set into array
file = 'Training_Set.txt'
with open(file) as f:
    reader = csv.reader(f, delimiter="\t")
    tweet_set = list(reader)

#clean tweets
tweets = []

for(words,sentiment) in tweet_set:
	filtered = [e.lower() for e in words.split() if len(e) >= 3]
	tweets.append((filtered, sentiment))

	
word_features = get_features(get_tweets(tweets))


#build training sets using word_features
training_set = nltk.classify.apply_features(take_features, tweets)



#build classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)


#load tweet file into list 
variable = 'HowlinRays'

t_file = variable + '_input.txt'
f = open(t_file,'r')
tweets = f.read().splitlines()

#perform sentiment analysis on list and write them to new output
t_output = variable + '_output.txt'
file = open(t_output, "w")
for tweet in tweets:
	file.write(classifier.classify(take_features(tweet.split())) + '\n')
file.close()

#raw output can be analyzed in whatever software you desire since its written to a text file. 






	
	
    




