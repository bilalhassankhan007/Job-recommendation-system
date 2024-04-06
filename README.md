# Job-recommendation-system
OBJECTIVE

1)Personalized Recommendations: To develop a system that provides personalized job recommendations to users based on their skills, experience, preferences, and other relevant factors.
2)Enhanced User Experience: Create a user-friendly platform that simplifies the job search process and empowers users to discover relevant job opportunities quickly and easily.
3)Scalability and Adaptability: Build a scalable and adaptable system capable of handling large volumes of job data and evolving user preferences, ensuring its effectiveness in dynamic job markets.
4)Improved Job Matching: Enhance the job search experience by accurately matching job seekers with job opportunities that align with their qualifications and career goals.


DATA COLLECTION
Importing the important libraires, classes and methods for data manipulation and further analysis. Libraries and classes imported as follow:
•	Pandas: It has functions for analysing, cleaning, exploring, and manipulating data.
•	Scikit Learn - It provides a selection of efficient tools for machine learning and statistical modelling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python. 
•	NLTK: NLTK (Natural Language Toolkit) is the go-to API for NLP (Natural Language Processing) with Python. It is a tool to preprocess text data for further analysis like with ML models for instance. It helps convert text into numbers, which the model can then easily work with.
•	Tokenize: The aim is to eliminate stop words, punctuation, and other irrelevant information from the text. Tokenizers transform the text into a list of words, which can be cleaned using a text-cleaning function.
•	PorterStemmer: It uses predefined rules to convert words into their root forms.
•	Corpus: It is a collection of text documents. It can be thought as just a bunch of text files in a directory, often alongside many other directories of text files.
•	Stopwords: It is used for text mining & to eliminate words that are so widely used that they carry very little useful information.
•	Regular Expression(re): It is used to match the strings of text such as particular characters, words or patterns of characters. It means that we can match and extract any string pattern from the text with the help of regular expressions.


DATA PREPROCESSING

Data Cleaning
•	Checking the NULL Values.
•	Checking Outliers in the dataset.
Text Preprocessing
•	Using techniques for text preprocessing such as tokenization, removing punctuation, lowercasing and stemming.
•	Used TF-IDF(Term Frequency-Inverse Document Frequency) scores or word embeddings. This process helped in converting textual data into numerical representation suitable for machine learning algorithm. 

Find Duplicates 
•	Examine the whole DataFrame if there is.
•	If duplicates values present in the Dataset, then remove it.
•	Use of “drop duplicates” to remove the values.
•	And rerun the DataFrame and check again. 

FEATURE ENGINEERING

•	Using PORTER STEMMER. The Porter stemming algorithm (or ‘Porter stemmer’) is a process for removing the commoner morphological and inflexional endings from words in English. Its main use is as part of a term normalisation process that is usually done when setting up Information Retrieval systems.
•	Then created a function which as follows –
 def cleaning(txt):
 	cleaned_txt = re.sub(r’[^a-zA-Z0-9\s]’, ‘’, txt)
	tokens = word_tokenize(cleaned_txt.lower())
	stemming = [ps.stem(word) for word in tokens if word not in 
	stopwords.words(‘english)]
	return “ “.join(stemming)

 VECTORIZATION
 •	In our model we have used Term Frequency-Inverse Document Frequency (TF-IDF): TF-IDF calculates a numerical value for each word in a document based on its frequency (TF) within the document and its rarity (IDF) across the entire corpus of documents. This technique assigns higher weights to words that are frequent in the document but rare across the corpus.
•	IMPORTING TfidVectorizer and cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
•	TfidfVectorizer: This module is used for vectorizing textual data using the TF-IDF (Term Frequency-Inverse Document Frequency) technique. TF-IDF is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents (corpus). The TfidfVectorizer class converts a collection of raw documents into a matrix of TF-IDF features. It preprocesses the text data, tokenizes it into individual words or tokens, computes the term frequency (TF) and inverse document frequency (IDF) values for each term, and generates a numerical matrix representing the TF-IDF features of the documents
•	cosine_similarity: This module is used to compute the cosine similarity between pairs of vectors. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. In the context of text data, cosine similarity is commonly used to measure the similarity between documents based on their TF-IDF vector representations. Higher cosine similarity values indicate greater similarity between documents, while lower values indicate dissimilarity.



	Purpose of “Cleaning” function -	
•	Removing non-alphanumeric characters.
•	Tokenizing the text into individual words
•	Converting words to lowercase for uniformity.
•	Stemming the words to their base form.
•	Removing stopwords to reduce noise in the text data.


Importing pickle module

•	In Python, the pickle module is used for serializing and deserializing Python objects. When working on a project involving dataframes, such as with pandas, pickle can be used to efficiently save and load dataframe objects.
•	Import pickle
pickle.dumple(df, open(‘df.pkl’, ‘wb’))
pickle.dump(similarity, open(‘similarity.pkl’, ‘wb’))
•	By using PICKLE to save and load dataframes, you can efficiently store intermediate results, trained models, or any other dataframe-based data structures in your Python projects. This helps in improving the overall efficiency and reproducibility of your data analysis or machine learning workflows.

