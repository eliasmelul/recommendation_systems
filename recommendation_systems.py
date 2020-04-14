import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import re

movies_df = pd.read_csv('C:/Users/melul/Desktop/Recommender_System/movies.csv')
ratings_df = pd.read_csv('C:/Users/melul/Desktop/Recommender_System/ratings.csv')
movies_df.head()

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))')
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)')
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))','')
movies_df['title'] = movies_df['title'].apply(lambda s: s.strip())
movies_df.head()

movies_df['genres'] = movies_df.genres.str.split('|')
movies_df

# Now we need to make the genres column into binary variables
movies_genres = movies_df.copy()

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movies_genres.at[index, genre] = 1
movies_genres = movies_genres.fillna(0)
movies_genres.head()

ratings_df.head()

ratings_df = ratings_df.drop('timestamp', axis=1)

ratings_df.head()

class rated_movie:
    def __init__(self, title, rating):
        self.title = title
        self.rating = rating

def input_movies():
    add_movie = True
    userInput = []
    while add_movie == True:
        title = input("Name of movie: ")
        rating = input("Rating of movie: ")
        movie = rated_movie(title, rating)
        userInput.append(movie.__dict__)
        cont = input("Do you want to add another movie?")
        add_movie = cont.lower() in ['yes','true','of course','y']
    inputMovies = pd.DataFrame(userInput)
    return(inputMovies)

movie1 = rated_movie("Lion King, The",5)
movie2 = rated_movie("Toy Story", 4.5)
movie3 = rated_movie("Iron Man", 4.5)
movie4 = rated_movie("Jumanji", 2.5)
movie5 = rated_movie("V for Vendetta", 5)

userInput = [movie1.__dict__,
            movie2.__dict__,
            movie3.__dict__,
            movie4.__dict__,
            movie5.__dict__
            ]
inputMovies = pd.DataFrame(userInput)
inputMovies

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('genres',1).drop('year',1)
inputMovies

userMovies = movies_genres[movies_genres['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies

userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId', 1).drop('title',1).drop('genres',1).drop('year',1)
userGenreTable

inputMovies['rating'].astype(float)

userProfile = userGenreTable.transpose().dot(inputMovies['rating'].astype(float))
userProfile

genTable = movies_genres.set_index(movies_genres['movieId'])
genTable = genTable.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)
genTable.head()

recom_df = ((genTable*userProfile).sum(axis=1))/userProfile.sum()
recom_df = recom_df.sort_values(ascending=False)
recom_df.head()

movies_df.loc[movies_df['movieId'].isin(recom_df.head(10).keys())]

# Once we have the input of the movies watched and rated, we need to look for the subset of users that have watched and reviewed the same movies!
users_watch = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
users_watch.head(10)

users_subsets = users_watch.groupby('userId')

# Note: consider creating a for loop that loops through all the userId and counts how many rated movies they have total.

users_subsets = sorted(users_subsets, key = lambda x: len(x[1]), reverse=True) # This sorts the groups in the data by # of reviews in common with input

users_subsets[0:5]

users_subsets = users_subsets[1:100] # We do this for computational purposes

corr_dict = {}

for name, group in users_subsets:
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    
    nrat = len(group)
    
    common_rev_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    
    input_revs_common = common_rev_df['rating'].tolist()
    
    group_revs_common = group['rating'].tolist()
    
    Sxx = sum([i**2 for i in input_revs_common]) - pow(sum(input_revs_common),2)/float(nrat)
    
    Syy = sum([i**2 for i in group_revs_common]) - pow(sum(group_revs_common),2)/float(nrat)
    
    Sxy = sum(i*j for i,j in zip(input_revs_common,group_revs_common)) - sum(input_revs_common)*sum(group_revs_common)/float(nrat)
    
    if Sxx != 0 and Syy != 0:
        corr_dict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        corr_dict[name] = 0

corr_dict.items()

corr_df = pd.DataFrame.from_dict(corr_dict, orient='index')
corr_df.columns = ['similarityCorr']
corr_df['userId'] = corr_df.index
corr_df.index = range(len(corr_df))
corr_df.head()

# Who are the most similar users?
mostSimilar = corr_df.sort_values(by='similarityCorr', ascending=False)[0:50]
mostSimilar.head()

mostSimilarRatings = mostSimilar.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
mostSimilarRatings.head()

mostSimilarRatings['weightedRating'] = mostSimilarRatings['similarityCorr']*mostSimilarRatings['rating']
weighted_ratings = mostSimilarRatings.groupby('movieId').sum()[['similarityCorr','weightedRating']]
weighted_ratings.columns = ['sum_similarity_corr','sum_weighted_rating']
weighted_ratings.head()

recommendation_df = pd.DataFrame()

recommendation_df['Weighted Avg Score'] = weighted_ratings['sum_weighted_rating']/weighted_ratings['sum_similarity_corr']
recommendation_df['movieId'] = weighted_ratings.index
recommendation_df = recommendation_df.sort_values(by='Weighted Avg Score', ascending=False)
recommendation_df.head()

movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(20)['movieId'].tolist())]['title']


