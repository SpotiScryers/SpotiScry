![Header](https://i.pinimg.com/originals/d2/c6/29/d2c629d9295ff516375ef2ec3ac25bc8.png)

## About the Project
What is the target variable? Popularity

### Goals
- Build a dataset of songs using Spotify's API
- Identify the drivers of song popularity
- Create a regression model to predict how popular a song will be that has an RMSE lower than the baseline

### Background
What makes a song popular? According to Splinter News [here](https://splinternews.com/how-does-a-song-become-number-one-1793850261),
>    "making a 'good' number one song is not necessarily the same as making a 'good' song in general. It's not about artistry (though 
> sometimes artistry does hit number one). It's about popularity. And not long-term popularity. But popularity right here, right now."  

By analyzing Spotify's API data, we will determine ourselves what influences a song's popularity.

### Deliverables
- Video presentation
- Presentation slides via Canva [here](https://www.canva.com/design/DAEQUdzBtqM/JW1AI9WU9ad01VO14yr2dg/view?utm_content=DAEQUdzBtqM&utm_campaign=designshare&utm_medium=link&utm_source=homepage_design_menu)
- GitHub repository with analysis

### Timeline
- [X] Project Planning: December 8th
- [X] Aquisition and Prep: December 10th
- [X] Exploration: December 14th
- [X] Modeling: December 15th
- [X] Finalize Minimum Viable Product (MVP): EOD December 15th
- [ ] Improve/Iterate MVP: December 16th
- [ ] Finalize Presentation: December 17th

### Acknowledgments
* [Continuous data stratification](https://danilzherebtsov.medium.com/continuous-data-stratification-c121fc91964b) by Danil Zherebtsov
* [Using Spotipy Library](https://towardsdatascience.com/how-to-create-large-music-datasets-using-spotipy-40e7242cc6a6) by Max Hilsdorf

## Data Dictionary
| Feature                | Description                                                                                                                                                                                                                                                                                          |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| artists                | The artists who performed the track                                                                                                                                                                                                                                                                  |
| album                  | The album in which the track appears                                                                                                                                                                                                                                                                 |
| track_name             | The name of the track                                                                                                                                                                                                                                                                                |
| track_id               | The spotify ID for the track                                                                                                                                                                                                                                                                         |
| danceability           | A value of 0 - 1 that represents a combination of tempo, rhythm stability, beat strength, and overall regularity                                                                                                                                                                                     |
| energy                 | A value of 0 - 1 that represents a perceptual measure of intensity and activity. The faster, louder, noisier a track is the higher the energy                                                                                                                                                        |
| key                    | The estimated overall key of the track, integers map to pitches using standard Pitch Class notation. If no key was detected, value is -1. 0 = C 1 = C# 2 = D etc.                                                                                                                                    |
| loudness               | The overall loudness of a track in decibels (dB). Values typically range between -60 and 0                                                                                                                                                                                                           |
| mode                   | The modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major = 1, Minor = 0                                                                                                                                                                          |
| speechiness            | A value of 0 - 1 that represents how exclusively speech-like the recording is. Values above .66 are made almost entirely of spoke words, .33 - .66 values may contain both music and speech, either in sections or layered. Values .33 most likely represent music and other non-speech-like tracks. |
| instrumentalness       | Predicts whether a track contains no vocals, The close the instrumentalness value is to 1 the greater the likelihood the track contains no vocal content. Values above .5 are intended to represent instrumental tracks, but confidence is higher as the value aproaches 1.                          |
| liveness               | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.                                                                              |
| valence                | A measure from 0 - 1 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (happy, cheerful, euphoric), while tracks with low valence sound more negative (sad, depressed, angry).                                                                   |
| tempo                  | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.                                                                                                           |
| duration_ms/duration_s | The duration of the song in ms or s respectively                                                                                                                                                                                                                                                     |
| time_signature         | An estimated overall time signature of a track, the time signature is a notational convention to specify how many beats are in each bar.                                                                                                                                                             |
| release_date           | The date the album was first released, if only year was given as precision it defaults to YYYY-01-01                                                                                                                                                                                                 |
| popularity             | Target variable, value between 0 - 100 that measures how many views the track has gotten in relation to how current those views are.                                                                                                                                                                 |
| explicit               | Boolean variable for whether or not the track has explicit lyrics.                                                                                                                                                                                                                                   |

## Initial Thoughts & Hypotheses
### Thoughts
* What are the drivers of popularity on Spotify?
* Is there a seasonality to the popularity of tracks?
* Are originals or remixes more popular?
* Since 2020 has been the year of the pandemic, are more people listening to sad songs right now?
* Are people's musical tastes expanding or experimenting due to the "new normal" of stay-at-home culture?
* Does loudness have a relationship with popularity?
* Does the instrumental-to-lyrical ratio of a track have an effect on its popularity?

### Hypotheses

𝐻0: Mean of song popularity of explicit tracks = Mean of song popularity of non-explicit tracks<br>
𝐻𝑎: Mean of song popularity of explicit tracks > Mean of song popularity of non-explicit tracks

𝐻0: Mean of popularity of major key songs =< Mean of popularity of minor key songs<br>
𝐻𝑎: Mean of popularity of major key songs > Mean of popularity of minor key songs

𝐻0: Mean of popularity of time signature 4 =< Mean of popularity of all songs<br>
𝐻𝑎: Mean of popularity of time signature 4 > Mean of popularity of all songs

𝐻0: There is no linear relationship between song length and popularity.<br>
𝐻𝑎: There is a linear relationship between song length and popularity.

𝐻0: There is no linear relationship between liveness and popularity.<br>
𝐻𝑎: There is a linear relationship between liveness and popularity.

𝐻0: There is no difference in popularity between tracks released by the top 10 labels or not.<br>
𝐻𝑎: Tracks released by the top 10 labels are more likely to be popular.

𝐻0: There is no difference in popularity between tracks released by the worst 5 labels or not.<br>
𝐻𝑎: Tracks released by the worst 5 labels are more likely to be unpopular.

𝐻0: there is no difference between songs released in 2020 popularity and the overall average.<br>
𝐻𝑎: there is a difference between songs released in 2020 popularity and the overall average.

## Project Steps
### Acquire
Data was acquired from Spotify API using the spotipy library. Going to this website https://developer.spotify.com/dashboard/login let us create a spotify web app that gave us a client id and client secret. This allowed us to use the create_spotipy_client function to create our own spotipy client that could access the API.  

The dataframe is saved as a csv file and has around 5900 observations, otherwise in the acquire.py file there is function for grabbing the entire capstone playlist as well as a function for acquiring any additional playlists should you choose. There are 24 columns in the original data frame, this ranges from track and album metadata to audio features for that track. There are very few nulls which have been marked as null in the data acquisition function for ease of removal later in prepare.
### Prepare
Functions to prepare the dataframe are stored in two seperate files depending on their purpose, prepare.py and preprocessing.py:  

**prepare.py:** Functions for cleaning and ordering data
* release dates that only specify the year are set to '01-01' for month and day
* nulls are dropped
* set track id to index
* change dtypes to correct type

**preprocessing.py:** Functions for adding features we found interesting / modyifying data for ease of use in exploration
* convert track length from ms to seconds & minutes
* lowercase artist, album, and track name
* create column for year, month, and day for release date
* bin release year by decade

### Explore
During exploration we looked at these features:
* if a track is explicit
* liveness
* song length
* time signature
* key
* loudness
* original vs remix
* instrumentalness
* danceability

### Model
### Conclusions
## How to Reproduce
1. ~Read through the README.md file~ :white_check_mark:  
2. Download acquire.py, prepare.py, preprocessing.py, and data folder.
3. If you don't have spotipy installed run this in your terminal: ~~~pip install spotipy~~~  
4. Login/Sign up at https://developer.spotify.com/dashboard/login to create a Spotify webapp that'll give you your client id and client secret.
5. Create an env.py file in your working directory and save this code after swaping out your individual client id and secret: 
~~~
cid = YOURCLIENTID
c_secret = YOURCLIENTSECRET
~~~
6. Using the functions in acquire create a spotipy client.
7. Use the functions in prepare.py and preprocessing.py to clean and set up your data.
8. Enjoy exploring the data!
### Steps
### Tools & Requirements
## License
## Creators
[Brandon Martinez](https://github.com/Brandon-Martinez27), [Bethany Thompson](https://github.com/ThompsonBethany01), [Kwame V. Taylor](https://github.com/KwameTaylor), [Matthew Mays](https://github.com/Matthew-Mays)
