# Simple Spotify Recommender System

### The data is scraped through Spotify API. There are 17 features in the data.
  #### track_name: the name of the track (song)
  #### artist_name: the name of the artist
  #### popularity: how popular the song is
  #### market: the country which the data scrapes from
  #### genres: the list of types of the song
  #### durations_mins: the length of the song
  #### release_year: the year the song released
  #### tempo: the tempo of the song
  #### speechiness: to classify the podcast, talkshow, ...
  #### loudness: the decibels
  #### liveness: to identify if the song is live performed
  #### instrumentalness
  #### energy: 
  #### acousticness
  #### valence: to determine whether the song is sad or happy
  #### like*: to determine whether the audience like the song or not (1-like, 0-not like)
  
### There are steps in this project:
  #### Preprocessing
  #### EDA
  #### Building simple song recommender system
  #### Evaluating

### Approaches:
  #### Aprroach 1: The audience will score the songs in the scale from 0 to 10 to make a list. We are going to use that list to train the model and make song predictions.
  #### Approach 2: Create a new feature 'like' to determine whether the audiences like the song or not (1-like, 0-not like). With this feature, we use classifying algorithms
  #### to train the model.
  
