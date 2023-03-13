# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:10:04 2018

@author: Frank
"""
import pandas as pd
from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
testSubject = '85'
k = 10

# Load our testdata set and compute the user similarity matrix
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()
a = pd.read_csv("../ml-latest-small/ratings.csv")
print(a.head())

trainSet = data.build_full_trainset()


sim_options = {'name': 'cosine', # 코사인 유사도를 기반.. MSD, 피어슨 모두 가능
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()
# print(f"simsMatrix: {simsMatrix[84]}")

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(testSubject)
print(f"testUserInnerID:{testUserInnerID}")
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
print(f"kNeighbors: {kNeighbors}") # 비슷한 유저의 id와 유사도를 저장

# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    print(f"theirRatings: {theirRatings}")
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore

# print(f"candidates: {candidates}")
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getMovieName(int(movieID)), ratingSum)
        pos += 1
        if (pos > 10):
            break



