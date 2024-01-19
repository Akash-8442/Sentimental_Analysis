import pickle 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer,LancasterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize, RegexpTokenizer 

from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer ,TfidfVectorizer


data = pd.read_csv("Equal.csv")
data = data[["review", "Sentiment"]]
# print(data.shape)
# print(data.loc[13,"Reviews"]) 
# print(data.loc[13,"Sentiment"]) 
# sns.countplot(x="Sentiment", data = data)
# plt.show()
# print(sent_tokenize(data.loc[16,"review"]))
# print(sent_tokenize(data.loc[156,"review"]))
# print(word_tokenize(data.loc[156,"review"]))

reviews = list(data["review"])
# print(sw)
# print(reviews)
# print(len(reviews)) 
reviews_lower = [str(r).lower() if isinstance(r, str) else '' for r in reviews]
# print(reviews_lower[156]) 
tokens = [word_tokenize(str(r)) if isinstance(r, (str, int, float)) else [] for r in reviews]
# print(tokens) 
sw = stopwords.words('english')
tokens = [[word for word in t if word not in sw] for t in tokens]
tokenizer = RegexpTokenizer(r'\w+')
tokens = [["".join(tokenizer.tokenize(word)) for word in t if len(tokenizer.tokenize(word)) >0] for t in tokens]
# print(tokens[10]) 
ps = PorterStemmer()
tokens = [[ps.stem(word) for word in t] for t in tokens]
# print(tokens[10])
flat_tokens = [word for t in tokens for word in t]
# print(len(flat_tokens))
counts = Counter(flat_tokens)
# print(len(counts))
# print(counts.most_common(10)) 
clean_reviews = [" ".join(t) for t in tokens]
# print(clean_reviews[15])
vect = CountVectorizer(binary=True, min_df= 5)
X = vect.fit_transform(clean_reviews)
# print(X.shape)
X_a = X.toarray()
# print(X_a[50:]) 


data["Sentiment"] = data["Sentiment"].apply(lambda x: 1 if x=="positive" else 0) 
y = data["Sentiment"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state =42) 

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
# print(accuracy_score(y_train,train_pred))
test_pred = model.predict(X_test)
# print(accuracy_score(y_test,test_pred))

# with open("Output/binary_count_vect.pkl", "wb") as f:
#     pickle.dump(vect, f) 
# with open("Output/binary_count_vect_lr.pkl", "wb") as f:
#     pickle.dump(model, f)    
# vect = CountVectorizer(min_df=5)
# X = vect.fit_transform(clean_reviews)
# X_a = X.toarray()
# print(clean_reviews[16]) 
# print(vect.get_feature_names().index("must")) 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y, random_state=42)

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)
# print(f"Train Accuracy:{accuracy_score(y_train, train_pred)}")
# print(f"Teat Accuracy:{accuracy_score(y_test, test_pred)}")


# vect = CountVectorizer(min_df=5, ngram_range=(1,3))
# X = vect.fit_transform(clean_reviews)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y, random_state=42)
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)
# print(f"Train Accuracy:{accuracy_score(y_train, train_pred)}")
# print(f"Teat Accuracy:{accuracy_score(y_test, test_pred)}") 

vect = TfidfVectorizer(min_df = 5)
X = vect.fit_transform(clean_reviews)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print(f"Train Accuracy:{accuracy_score(y_train, train_pred)}")
print(f"Test Accuracy:{accuracy_score(y_test, test_pred)}") 


# test_review_1 = '''this is a truly amazing app, best for those hwo havw content but don't know how to express it in a good and shareable manner.
# Thanks Team Canva for such a great app.''' 
# text_review_2 = '''Its the worst app I save my design Its not save'''

# test_review_1 = [test_review_1]
# test_review_2 = [text_review_2]
# # Change in lowercase
# test_review_1 = [r.lower() for r in test_review_1] 
# test_review_2 = [r.lower() for r in test_review_2]
# # Tokenize the text
# tokens_1 = [word_tokenize(r) for r in test_review_1]
# tokens_2 = [word_tokenize(r) for r in test_review_2]
# # Remove stopwords
# tokens_1  = [[word for word in t if word not in sw] for t in tokens_1]
# tokens_2  = [[word for word in t if word not in sw] for t in tokens_2]
# # Remove punctuations
# tokens_1 = [["".join(tokenizer.tokenize(word)) for word in t if len(tokenizer.tokenize(word))>0] for t in tokens_1]
# tokens_2 = [["".join(tokenizer.tokenize(word)) for word in t if len(tokenizer.tokenize(word))>0] for t in tokens_2]
# # Stemming 
# tokens_1 = [[ps.stem(word) for word in t] for t in tokens_1]
# tokens_2 = [[ps.stem(word) for word in t] for t in tokens_2]
# # Join the tokens to form a sentence 
# clean_review_1 = [" ".join(review) for review in tokens_1]
# clean_review_2 = [" ".join(review) for review in tokens_2]
# vect = CountVectorizer(min_df=5, ngram_range=(1,3))
# X_test = vect.fit_transform(clean_review_1)
# print(X_test.shape)











