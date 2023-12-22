# Implement the six functions below

##############################################################################
############################ Question 2a  ####################################
##############################################################################

import json
def naive_output_probs(training_data):
    delta = 0.1
    content = open(training_data, "r", encoding='utf8')
    data = content.read().split()

    # All tokens converted to lowercase
    tokens = []
    tags = []
    for i, word in enumerate(data):
        if i%2 == 0:
            tokens.append(word.lower())
        else:
            tags.append(word)

    #Make a list of pairs of tags & tokens that occurred together
    pairs = []
    for i in range(len(tokens)):
        pairs.append([tags[i], tokens[i]])

    uniqueTokens = list(set(tokens))
    uniqueTags  = list(set(tags))

    #create a emissionNums dictionary with each of the 25 unique tags as keys.
    #Each tag opens up another dictionary that contains every unique token as keys.

    emissionNums = {}
    for tag in uniqueTags:
        emissionNums[tag] = {}
        for token in uniqueTokens:
            emissionNums[tag][token] = 0
    
        emissionNums[tag]['unknown_token'] = 0

            
    # emissionNums[tag][token] records the no. of times token occured tgt with tag
    for pair in pairs:
        emissionNums[pair[0]][pair[1]] += 1

    #create emissionProbs dictionary that is basically the same as emissionNums, but converts counts to probability
    # i.e. emissionProbs[tag][token] = P_naive(token = w | tag = j. Use smoothing output probability with delta
    
    emissionProbs = {}
    for tag in emissionNums:
        emissionProbs[tag] = {} 
        for token in emissionNums[tag]:
            emissionProbs[tag][token] = (emissionNums[tag][token] + delta)  / (sum(emissionNums[tag].values()) + delta * (len(uniqueTokens) + 1))

    with open('naive_output_probs.txt', 'w') as file:
        json.dump(emissionProbs, file)


##############################################################################
############################ Question 2b  ####################################
##############################################################################

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    # load the naive output probs dictionary
    with open(in_output_probs_filename) as file:
        output_probs = json.load(file)

    # read the file being tested, convert all tokens to lowercase
    with open(in_test_filename, "r", encoding='utf8') as test:
        tokens = []
        allTokens = test.read().split()

        for token in allTokens:
            tokens.append(token.lower())
        
    results = []
    for token in tokens:
        #set default token as A
        best_tag = "A"
        
        #since every output_probs[tag] have all unique tokens from training set as keys, if this new token is not inside,
        #then it must have not been in the training set. Set to _.
        #Otherwise, append to results the tag that has highest output_probs[tag][token] = P_naive(token=w | tag=j)
        if token not in output_probs["A"]:
            best_tag_prob = output_probs["A"]["unknown_token"]
            for tag in output_probs:
                if output_probs[tag]["unknown_token"] > best_tag_prob:
                    best_tag = tag
                    best_tag_prob = output_probs[tag]["unknown_token"]
            results.append(best_tag)

        else:
            
            best_tag_prob = output_probs["A"][token]
            for tag in output_probs:
                if output_probs[tag][token] > best_tag_prob:
                    best_tag = tag
                    best_tag_prob = output_probs[tag][token]
            results.append(best_tag)
            

    with open(out_prediction_filename, 'w') as output:
        for tag in results:
            output.write(tag + '\n')
    pass

##############################################################################
############################ Question 2c  ####################################
##############################################################################

# naive_predict('naive_output_probs.txt', 'twitter_dev_no_tag.txt',  'naive_predictions.txt')
# evaluate('naive_predictions.txt', 'twitter_dev_ans.txt')

#The result is (937, 1378, 0.6799709724238027)


##############################################################################
############################ Question 3a  ####################################
##############################################################################

#Using Bayes Rule: P(tag=j | token=w) = [P(token=w | tag=j) / P(token=w)] * P(tag=j)
# P(token=w | tag=j) to be retrieved from naive_output_probs dictionary
    # P(token=w)  --> We can ignore this component as it doesn't involve j. We just want to choose j to maximise the probability P(tag=j | token=w)
    # P(tag=j) = count(tag j in training set) / count(total number of tags in training set)


##############################################################################
############################ Question 3b  ####################################
##############################################################################
    
def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    # load the output probs dictionary
    with open(in_output_probs_filename) as file:
        output_probs = json.load(file)

    #read the training set, convert tokens to lowercase
    with open(in_train_filename, "r", encoding='utf8') as training:
        training = training.read().split()

    training_tokens = []
    training_tags = []
    for i, word in enumerate(training):
        if i%2 == 0:
            training_tokens.append(word.lower())
        else:
            training_tags.append(word)

    # read the file being tested, convert tokens to lowercase
    with open(in_test_filename, "r", encoding='utf8') as test:
        tokens = []
        allTokens = test.read().split()
        for token in allTokens:
            tokens.append(token.lower())

    results = []
    for token in tokens:
        # set A as default token
        best_tag = "A" 
        if token not in output_probs["A"]:
            best_tag_prob = output_probs["A"]["unknown_token"]
            for tag in output_probs:
                if output_probs[tag]['unknown_token'] * (training_tags.count(tag)/len(training_tags)) > best_tag_prob:
                        best_tag_prob = output_probs[tag]['unknown_token'] * (training_tags.count(tag)/len(training_tags))
                        best_tag = tag

            results.append(best_tag)
            
        else:
            # Bayes Rule
            best_tag_prob = output_probs["A"][token] * (training_tags.count("A")/len(training_tags))

            for tag in output_probs:
                if output_probs[tag][token] * (training_tags.count(tag)/len(training_tags)) > best_tag_prob:
                    best_tag_prob = output_probs[tag][token] * (training_tags.count(tag)/len(training_tags))
                    best_tag = tag
            results.append(best_tag)

    with open(out_prediction_filename, 'w') as output:
        for tag in results:
            output.write(tag + '\n')


##############################################################################
############################ Question 3c  ####################################
##############################################################################

# naive_predict2("naive_output_probs.txt", "twitter_train.txt", "twitter_dev_no_tag.txt", "naive_predictions2.txt")
# evaluate('naive_predictions2.txt', 'twitter_dev_ans.txt')

#The result is (980, 1378, 0.7111756168359942)
pass


##############################################################################
############################ Question 4a  ####################################
##############################################################################

def mle_probs(training_data):
    delta = 0.1
    content = open(training_data, "r", encoding='utf8')
    lines = content.read().split('\n\n')
    lines = [line for line in lines if line]

    #separate data into tag/token pairs within each tweet
    obs = []
    for line in lines:
        obs.append(line.split('\n'))

    for line in obs:
        for i in range(len(line)):
            line[i] = line[i].split('\t')

    for line in obs:
        for pair in line:
            pair[0] = pair[0].lower()

    obs_tagsOnly = []
    for line in obs:
        tags = []
        for pair in line:
            tags.append(pair[1])
        obs_tagsOnly.append(tags)


    #Get list of all tokens, tags, uniqueTokens, uniqueTags, and pairs of Tag/Token occuring tgt
    tokens = []
    tags = []
    for line in obs:
        for pair in line:
            tags.append(pair[1])
            tokens.append(pair[0])
    uniqueTokens = list(set(tokens))
    uniqueTags = list(set(tags))
    pairs = []
    for i in range(len(tokens)):
        pairs.append([tags[i], tokens[i]])

    #initialise transition nums dictionary
    transition_nums = {"START":{}}
    for from_tag in uniqueTags:
        transition_nums[from_tag] = {}

    for from_tag in transition_nums:
        for to_tag in uniqueTags:
            transition_nums[from_tag][to_tag] = 0
        transition_nums[from_tag]["END"] = 0

    #populate transition_nums dictionary
    for tweet in obs_tagsOnly:
        for i in range(len(tweet)):
            if i == 0:
                transition_nums["START"][tweet[i]] += 1
                if len(tweet) > 1:
                    transition_nums[tweet[i]][tweet[i+1]] += 1
            elif i == len(tweet) - 1:
                transition_nums[tweet[i]]["END"] += 1
            else:
                transition_nums[tweet[i]][tweet[i+1]] += 1

    #Convert transition_nums to transition_probs
    transition_probs = {}
    for from_tag in transition_nums:
        transition_probs[from_tag] = {}
        for to_tag in transition_nums[from_tag]:
            #Prob smoothing: (count(i -> j) + delta) / (count(i) + delta * (no. of unique tags + 1))
            transition_probs[from_tag][to_tag] = (transition_nums[from_tag][to_tag] + delta) / (sum(transition_nums[from_tag].values()) + delta * (len(uniqueTags) + 1))

    with open('trans_probs.txt', 'w') as file:
        json.dump(transition_probs, file)
        
    
                                        
    ################ Find output probs in same way as Q2 #####################
    emissionNums = {}
    for tag in uniqueTags:
        emissionNums[tag] = {}
        for token in uniqueTokens:
            emissionNums[tag][token] = 0
    
        emissionNums[tag]['unknown_token'] = 0

            
    # emissionNums[tag][token] records the no. of times token occured tgt with tag
    for pair in pairs:
        emissionNums[pair[0]][pair[1]] += 1

    #create emissionProbs dictionary that is basically the same as emissionNums, but converts counts to probability
    # i.e. emissionProbs[tag][token] = P_naive(token = w | tag = j. Use smoothing output probability with delta
    
    emissionProbs = {}
    for tag in emissionNums:
        emissionProbs[tag] = {} 
        for token in emissionNums[tag]:
            emissionProbs[tag][token] = (emissionNums[tag][token] + delta)  / (sum(emissionNums[tag].values()) + delta * (len(uniqueTokens) + 1))

    with open('output_probs.txt', 'w') as file:
        json.dump(emissionProbs, file)

##########################################################################



##############################################################################
############################ Question 4b  ####################################
##############################################################################
        
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
    # load the full set of tags
    with open(in_tags_filename, "r", encoding = "utf8") as tagfile:
        allTags = tagfile.read().split()

    #load the transition probs dictionary
    with open(in_trans_probs_filename) as transfile:
        trans_probs = json.load(transfile)

    # load the output probs dictionary
    with open(in_output_probs_filename) as outputProbsFile:
        output_probs = json.load(outputProbsFile)

    # load the file being tested, split into tweets, convert all tokens to lowercase
    with open(in_test_filename, "r", encoding='utf8') as test:
        lines = test.read().split('\n\n')
        lines = [line for line in lines if line]
        obs = []
        for line in lines:
            obs.append(line.split('\n'))

        for tweet in obs:
            for i in range(len(tweet)):
                tweet[i] = tweet[i].lower()


    result = []

    for tweet in obs:
        path = {}
        BP = {}

        for i in range(1, len(tweet) + 1):
            path[i] = {}
            BP[i] = {}
            for tag in allTags:
                path[i][tag] = 0
                BP[i][tag] = 'nil'
        path[len(tweet) + 1] = {'END'}
        BP[len(tweet) + 1] = 'nil'


        #Base cases
        for tag in path[1]:
            if tweet[0] in output_probs[tag]:
                path[1][tag] = trans_probs['START'][tag] * output_probs[tag][tweet[0]]
            else:
                path[1][tag] = trans_probs['START'][tag] * output_probs[tag]['unknown_token']
        for pointer in BP[1]:
            BP[1][pointer] = "START"

        #Recursive
        for i in range(1, len(tweet)):
            for current_tag in path[i + 1]:
                maxTag = ""
                maxProb = 0
                
                for prev_tag in path[i]:
                    if tweet[i] in output_probs[current_tag]:
                        prob = path[i][prev_tag] * trans_probs[prev_tag][current_tag] * output_probs[current_tag][tweet[i]]
                    else:
                        prob = path[i][prev_tag] * trans_probs[prev_tag][current_tag] * output_probs[current_tag]['unknown_token']
                    if prob >  maxProb:
                        maxProb = prob
                        maxTag = prev_tag

                path[i + 1][current_tag] = maxProb
                BP[i + 1][current_tag] = maxTag

        
        #Ending
        maxTag = ""
        maxProb = 0
        for prev_tag in path[len(tweet)]:
            prob = path[len(tweet)][prev_tag] * trans_probs[prev_tag]['END']
            if prob > maxProb:
                maxProb = prob
                maxTag = prev_tag

        path[len(tweet) + 1] = maxProb
        BP[len(tweet) + 1] = maxTag


        #Display path using backpointers

        currentpointer = BP[len(tweet) + 1]
        bestpath = [currentpointer,]
        for i in range(len(tweet), 0, -1):
            bestpath.append(BP[i][currentpointer])
            currentpointer = BP[i][currentpointer]

        bestpath = bestpath[:-1] #remove START tag
        bestpath.reverse()
        result.append(bestpath)


    with open(out_predictions_filename, 'w') as output:
        for path in result:
            for tag in path:
                output.write(tag + '\n')
            output.write('\n')


##############################################################################
############################ Question 4c  ####################################
##############################################################################\
             
# (1070, 1378, 0.7764876632801161)
             
        
##############################################################################
############################ Question 5a  ####################################
##############################################################################
# For the first improvement, we perform pre-processing by trying to obtain the suffix of each word (if the word has one).
# As an example, we want this to occur:
# testing -> ing
# tested -> ed
# We can leave nouns or plural nouns as it is
# test -> test
# tests -> test

# We believe that POS tag is based on the tense/grammatical form of the word, so extracting the suffix will make it easier for the function to assign the correct tag, 
# e.g. instead of having to learn the underlying pattern of 'testing' and 'dancing' belonging to the same tag, representing the word as 'ing' makes it easier, 
# as there will be less distinct tokens to learn.

# We acheive this by importing Stemmer from NLTK package
from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='english')
# Stemming is the process of reducing words to their stem or root form and is typically used in NLP to standardize words by removing suffixes. 
# Stemming algorithms apply a set of rules to achieve this. It aims to improve accuracy of models by treating similar words as the same.
# Stemming achieves the reverse of what we want, hence after stemming, we extract the part that was stemmed, 
# e.g. stemming 'testing' will return 'test', so we built a function to recover the part that was stemmed, which is 'ing'.
def get_suffix(word: str) -> str:
    """Helper function to obtain suffix of word"""
    stemmed = snowball.stem(word)
    
    # If word is already 'stemmed', no suffix, just return original word
    if len(word) == len(stemmed):
        return word
    suffix = word.removeprefix(stemmed)
    
    # If word is plural, just return original word
    if suffix == 's':
        return word
    else:
        return suffix

#For the next improvment we use word patterns to better handle unseen words by creating a rule based system to tackle unseen words
#For instance words that begin with "@USER_", we know that these words are highly likely to be tagged to the tag "@", 
#Hence if a given tag is "@" and the current word begins with "@USER_" we set it's ouput probability to 1 instead of giving it the general unknown probability 
#We have created a rule based system to detect URLs, Hashtags, Users, Numbers, Retweets and Punctuations, the functions are as defined in question 5b

#For the last improvement, we seek to better model transition probabilities by using second-order HMMs.
#This way, a given tag is dependant on not only the tag before it, but also two tags before.
#for instance for a given tweet "@ A N .", we model transition probabilities as P(N|A, @) instead of P(N|A) in a first order HMM
#This can improve predictions as the complexity of a sentence structure may be better captured by second order HMMs.

##############################################################################
############################ Question 5b  ####################################
##############################################################################        
def mle_probs_trigram(training_data):
    delta = 0.1
    content = open(training_data, "r", encoding='utf8')
    lines = content.read().split('\n\n')
    lines = [line for line in lines if line]

    #separate data into tag/token pairs within each tweet
    obs = []
    for line in lines:
        obs.append(line.split('\n'))

    for line in obs:
        for i in range(len(line)):
            line[i] = line[i].split('\t')

    for line in obs:
        for pair in line:
            pair[0] = get_suffix(pair[0].lower())

    obs_tagsOnly = []
    for line in obs:
        tags = []
        for pair in line:
            tags.append(pair[1])
        obs_tagsOnly.append(tags)


    tokens = []
    tags = []
    for line in obs:
        for pair in line:
            tags.append(pair[1])
            tokens.append(pair[0])
    uniqueTokens = list(set(tokens))
    uniqueTags = list(set(tags))
    pairs = []
    for i in range(len(tokens)):
        pairs.append([tags[i], tokens[i]])


    #initialise transition nums dictionary
    #Transition probabilities are stored in a dictionary of dictionaries of dictionaries
    #e.g. @{@{@{1.0}}} would refer to the Probability of @ given that the previous two words were tagged @ or P(@| @, @).
    # * here refers to empty
    transition_nums = {"*":{}}
    for from_tag in uniqueTags:
        transition_nums[from_tag] = {}

    #transition probs that start with ** means it is a starting probability
    #e.g. "**@" or P(@| *, *) is the starting probability for @
    transition_nums["*"]["*"] = {}

    for from_tag in transition_nums:
        for middle_tag in transition_nums:
            transition_nums[from_tag][middle_tag] = {}
            
            for to_tag in transition_nums:
                transition_nums[from_tag][middle_tag][to_tag] = 0
                
            
            transition_nums[from_tag][middle_tag]['END'] = 0  

    #transitions between three empty spaces do not exist
    transition_nums["*"]["*"]['*'] = 0

    #populate transition_nums dictionary
    for tweet in obs_tagsOnly:

        for i in range(len(tweet)+1):
        
            if i == 0:
                if len(tweet) > 1:
                    transition_nums['*']['*'][tweet[i]] += 1
                else:
                    transition_nums['*'][tweet[i]]['END'] += 1
            elif i == 1:
                if len(tweet) > 1:
                    transition_nums['*'][tweet[i-1]][tweet[i]] += 1
                else:
                    transition_nums['*'][tweet[i-1]]['END'] += 1

            elif i == len(tweet):
                
                    transition_nums[tweet[i-2]][tweet[i-1]]['END'] += 1

            else:
                transition_nums[tweet[i-2]][tweet[i-1]][tweet[i]] += 1
        

    #Convert transition_nums to transition_probs
    transition_probs = {}
    for from_tag in transition_nums:
        transition_probs[from_tag] = {}

        for middle_tag in transition_nums[from_tag]:
            transition_probs[from_tag][middle_tag] = {}

            for to_tag in transition_nums[from_tag][middle_tag]:
                #Prob smoothing: (count(i -> j -> k) + delta) / (count(i-j) + delta * (no. of unique tags + 1))
                transition_probs[from_tag][middle_tag][to_tag] = (transition_nums[from_tag][middle_tag][to_tag] + delta) /  \
                (sum(transition_nums[from_tag][middle_tag].values()) + delta * (len(uniqueTags) + 1))


    with open('trans_probs2.txt', 'w') as file:
        json.dump(transition_probs, file)


    ################ Find output probs in same way as Q2 #####################
    emissionNums = {}
    for tag in uniqueTags:
        emissionNums[tag] = {}
        for token in uniqueTokens:
            emissionNums[tag][token] = 0
    
        emissionNums[tag]['unknown_token'] = 0

            
    # emissionNums[tag][token] records the no. of times token occured tgt with tag
    for pair in pairs:
        emissionNums[pair[0]][pair[1]] += 1

    #create emissionProbs dictionary that is basically the same as emissionNums, but converts counts to probability
    # i.e. emissionProbs[tag][token] = P_naive(token = w | tag = j. Use smoothing output probability with delta
    
    emissionProbs = {}
    for tag in emissionNums:
        emissionProbs[tag] = {} 
        for token in emissionNums[tag]:
            emissionProbs[tag][token] = (emissionNums[tag][token] + delta)  / (sum(emissionNums[tag].values()) + delta * (len(uniqueTokens) + 1))

    with open('output_probs2.txt', 'w') as file:
        json.dump(emissionProbs, file)

##########################################################################

#Word patterns for handling unseen words
#Function that returns true if token starts with hashtag
def is_hashtag(word):
    return word[0] == "#"

#Function returns true if token is a URL
def is_URL(word):
    return word.startswith("http://") or word.startswith("https://")

#Function that returns true if token is a USER
def is_USER(word):
    return word.startswith("@USER_")

#Function that returns true if token is a retweet
def is_RT(word):
    return word == "RT"

#Function that returns true if token is a number
#includes fractions and decimals such as 2/3 or 1.2
def is_number(word):
    i = 0
    if not word[i].isdigit():
        return False
    i += 1
    if i <len(word) and word[i] == ".":
        i += 1
    if i <len(word) and (word[i] == "," or word[i] == '/' or word[i] == "\\"):
        i += 1
    while i < len(word) and word[i].isdigit():
        i += 1
    if i != len(word):
        return False
    return True

#Function that returns true if token is a punctuation mark
def is_punctuation(word):
    punctuations = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    return word in punctuations


def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):

    # load the full set of tags
    with open(in_tags_filename, "r", encoding = "utf8") as tagfile:
        allTags = tagfile.read().split()

    #load the transition probs dictionary
    with open(in_trans_probs_filename) as transfile:
        trans_probs = json.load(transfile)

    # load the output probs dictionary
    with open(in_output_probs_filename) as outputProbsFile:
        output_probs = json.load(outputProbsFile)

    # load the file being tested, split into tweets, convert all tokens to lowercase
    with open(in_test_filename, "r", encoding='utf8') as test:
        lines = test.read().split('\n\n')
        lines = [line for line in lines if line]
        obs = []
        for line in lines:
            obs.append(line.split('\n'))

        for tweet in obs:
            for i in range(len(tweet)):
                tweet[i] = get_suffix(tweet[i].lower())
        

    result = []
    allTags.append("*")
    
    for tweet in obs:
        path = {}
        BP = {}

        path[0] = {}
        #Base case for Viterbi trigram model
        for i, tag1 in enumerate(allTags):
            path[0][tag1] = {}
            for j, tag2 in enumerate(allTags):
                if tag1 == "*" and tag2 == "*":
                    path[0][tag1][tag2] = 1
                else:
                    path[0][tag1][tag2] = 0
        
        #Recursive case for Viterbi Trigram model
        for k, token in enumerate(tweet):
            k = k + 1
            path[k] = {}
            BP[k] = {}
            for tag1 in allTags:
                path[k][tag1] = {}
                BP[k][tag1] = {}
                for tag2 in allTags:
                    path[k][tag1][tag2] = 0
                    BP[k][tag1][tag2] = 0
                    if tag1 == "*":
                        continue
                    for tag3 in allTags:
                        if token in output_probs[tag1]:
                            prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * output_probs[tag1][token]
                            
                        #Rule based conditions to handle unseen words
                        elif is_hashtag(token):
                            #if tag and token are matched, we set output probability to 1
                            if tag1 == "#":
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 1
                            #else if tag and token are not matched, we set output prob to 0 as token should be tagged as hashtag 
                            else:
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 0 
                        elif is_URL(token):
                            if tag1 == "U":
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 1
                            else:
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 0 
                        elif is_USER(token):
                            if tag1 == "@":
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 1
                            else:
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 0 
                        elif is_RT(token):
                            if tag1 == "~":
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 1
                            else:
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 0 
                        elif is_number(token):
                            if tag1 == "$":
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 1
                            else:
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 0 
                        elif is_punctuation(token):
                            if tag1 == ",":
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 1
                            else:
                                prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * 0  
                        else:
                            prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * output_probs[tag1]['unknown_token']
                        
                        if prob > path[k][tag1][tag2]:
                            path[k][tag1][tag2] = prob
                            BP[k][tag1][tag2] = tag3

        max_prob = 0
        max_bp = (0, 0)
        #Ending
        for tag1 in allTags:
            if tag1 == "*":
                continue
            for tag2 in allTags:
                last = len(tweet)
                prob = path[last][tag1][tag2] * trans_probs[tag2][tag1]["END"]
                if prob >= max_prob:
                    max_prob = prob
                    max_bp = (tag2, tag1)
        
        curr = []
        #set y_n-1 and y_n to argmax (path(n, u , v) * q(STOP | u, v))
        y_1, y_2 = max_bp
        if len(tweet) == 1:
            curr.append(y_2)
        else:
            curr.append(y_2)
            curr.append(y_1)
            #for k = (n- 2) .... 1
            #y_k = BP(k + 2, y_k+1, y_k+2)
            for k in reversed(range(len(BP))[2:]):
                k = k + 1
                current = BP[k][y_2][y_1]
                curr.append(current) 
                y_2 = y_1
                y_1 = current
                
        for tag in reversed(curr):
            result.append(tag)
        result.append("")
    
            
    with open(out_predictions_filename, 'w') as output:
        for tag in result:
            if tag == "":
                output.write('\n')
            else:
                output.write(tag + '\n')
                
##############################################################################
############################ Question 5c  ####################################
##############################################################################
#Viterbi2 prediction accuracy:  1086/1378 = 0.7880986937590712




def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = r'C:/Users/danyel/Desktop/Bt3102Proj'#your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')


    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    ddir = r'C:/Users/danyel/Desktop/Bt3102Proj'#your working dir
    in_train_filename = f'{ddir}/twitter_train.txt'
    naive_output_probs(in_train_filename)
    mle_probs(in_train_filename)
    mle_probs_trigram(in_train_filename)
    run()
