![Header](https://i.pinimg.com/originals/d2/c6/29/d2c629d9295ff516375ef2ec3ac25bc8.png)
## Table of Contents
1. About the Project  
[Goals](https://github.com/SpotiScryers/SpotiScry#goals) | [Background](https://github.com/SpotiScryers/SpotiScry#background) | [Deliverables](https://github.com/SpotiScryers/SpotiScry#deliverables) | [Outline](https://github.com/SpotiScryers/SpotiScry#project-outline)  

2. Data Dictionary  
[Original Features](https://github.com/SpotiScryers/SpotiScry#original-features) | [Engineered Features](https://github.com/SpotiScryers/SpotiScry#engineered-features)  

3. Initial Thoughts & Hypotheses  
[Thoughts](https://github.com/SpotiScryers/SpotiScry#thoughts) | [Hypotheses](https://github.com/SpotiScryers/SpotiScry#hypotheses)  

4. Project Steps  
[Acquire](https://github.com/SpotiScryers/SpotiScry#acquire) | [Prepare](https://github.com/SpotiScryers/SpotiScry#prepare) | [Explore](https://github.com/SpotiScryers/SpotiScry#explore) | [Model](https://github.com/SpotiScryers/SpotiScry#model) | [Conclusions](https://github.com/SpotiScryers/SpotiScry#conclusions)  

5. How to Reproduce & More  
[Steps](https://github.com/SpotiScryers/SpotiScry#steps) | [Tools & Requirements](https://github.com/SpotiScryers/SpotiScry#tools--requirements) | [License](https://github.com/SpotiScryers/SpotiScry#License) | [Creators](https://github.com/SpotiScryers/SpotiScry#Creators)

## About the Project
What makes a song reach the top of the charts while others flop? Using data from Spotify, our team will determine what features influence song popularity - such as the danceability or song length. We will then predict a song’s popularity. You can check out our presentation [here](https://www.canva.com/design/DAEQUdzBtqM/JW1AI9WU9ad01VO14yr2dg/view?utm_content=DAEQUdzBtqM&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton) and our Spotify playlist our data comes from [here](https://open.spotify.com/playlist/3P6Pr6iEqvK5fl4UkgdQ7T?si=6EzeUwUwTF61b2fpo7uNGg).

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
- Tableau Storybook [here](https://public.tableau.com/profile/thompson.bethany.01#!/vizhome/Spotiscry/Story1)
- GitHub repository with analysis

### Project Outline
The files within the repository are organized as follows. The /images and /sandbox contents are not necessary for reproduction.  
![Outline](https://github.com/SpotiScryers/SpotiScry/blob/main/images/Outline.png?raw=true)

### Timeline
- [X] Project Planning: December 8th
- [X] Aquisition and Prep: December 10th
- [X] Exploration: December 14th
- [X] Modeling: December 15th
- [X] Finalize Minimum Viable Product (MVP): EOD December 15th
- [X] Improve/Iterate MVP: December 17th
- [X] Finalize Presentation: December 31st

### Acknowledgments
* [Continuous data stratification](https://danilzherebtsov.medium.com/continuous-data-stratification-c121fc91964b) by Danil Zherebtsov
* [Using Spotipy Library](https://towardsdatascience.com/how-to-create-large-music-datasets-using-spotipy-40e7242cc6a6) by Max Hilsdorf
* [The Most Successful Labels in Hip Hop:](https://pudding.cool/2017/03/labels/) Every hip hop record label, since 1989, sorted by their artists' chart performance on Billboard, by Matt Daniels and Kevin Beacham
* [What Is “Escape Room” And Why Is It One Of My Top Genres On Spotify?:](https://festivalpeak.com/what-is-escape-room-and-why-is-it-one-of-my-top-genres-on-spotify-a886372f003f) Using data to understand how genres understand us, by Cherie Hu
* [Tunebat](https://tunebat.com/Info/WAP-feat-Megan-Thee-Stallion-Cardi-B-Megan-Thee-Stallion/4Oun2ylbjFKMPTiaSbbCih)
* [The Case For Lil Jon As One of Hip-Hop’s Greatest Producers](https://medium.com/@SermonsDomain/the-case-for-lil-jon-as-one-of-hip-hops-greatest-producers-ace21b04ab2b) by Erich Donaldson

<kbd>[Back to Table of Contents](https://github.com/SpotiScryers/SpotiScry#table-of-contents)</kbd>

## Data Dictionary
### Original Features
Below are the features included in the orginal data acquired from the Spotify API.  
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
| duration_ms            | The duration of the song in ms                                                                                                                                                                                                                                                                        |
| time_signature         | An estimated overall time signature of a track, the time signature is a notational convention to specify how many beats are in each bar.                                                                                                                                                             |
| release_date           | The date the album was first released, if only year was given as precision it defaults to YYYY-01-01                                                                                                                                                                                                 |
| popularity             | Target variable, value between 0 - 100 that measures how many views the track has gotten in relation to how current those views are.                                                                                                                                                                 |
| explicit               | Boolean variable for whether or not the track has explicit lyrics.                                                                                                                                                                                                                                   |

### Engineered Features
Using domain knowledge and exploration insights, we also engineered features using the original data. These created features are below.  
| Feature Name             | Description                                                                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| duration_seconds/minutes | Converting the track duration in milliseconds to seconds and minutes, rounded integers                                                        |
| is_featured_artist       | Boolean value if the track name includes 'feat', meaning an additional artist is on the track                                                 |
| decade                   | The decade the track was released in based on the release year, 80s - 90s - 2000s - 2010s - 2020s                                             |
| top_ten_label            | Boolean if the track is produced by a top record label (based on count of songs produced by the record and the average popularity)            |
| popularity_bins          | Binned values on popularity feature using domain knowledge: <br>0-10 as 'Very Low', 11-40 as 'Low', 41-70 as 'moderate', and 71-100 as 'High' |
| danceability_bins        | Binned values on danceability feature using qcut to create three equal bins:<br>0-.69 as 'Low', .70-.80 as 'Medium', .81-1.0 as 'High'        |

<kbd>[Back to Table of Contents](https://github.com/SpotiScryers/SpotiScry#table-of-contents)</kbd>
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

<kbd>[Back to Table of Contents](https://github.com/SpotiScryers/SpotiScry#table-of-contents)</kbd>
## Project Steps
### Acquire
Data was acquired from Spotify API using the spotipy library. Going to this website https://developer.spotify.com/dashboard/login let us create a spotify web app that gave us a client id and client secret. This allowed us to use the create_spotipy_client function to create our own spotipy client that could access the API.  
![Acquire-Visual](https://github.com/SpotiScryers/SpotiScry/blob/main/images/aquisition.png?raw=true)

The dataframe is saved as a csv file and has around 5900 observations, otherwise in the acquire.py file there is function for grabbing the entire capstone playlist as well as a function for acquiring any additional playlists should you choose. There are 24 columns in the original data frame, this ranges from track and album metadata to audio features for that track. There are very few nulls which have been marked as null in the data acquisition function for ease of removal later in prepare.
### Prepare
Functions to prepare the dataframe are stored in two seperate files depending on their purpose, prepare.py and preprocessing.py:  

**prepare.py:** Functions for cleaning and ordering data
* release dates that only specify the year are set to '01-01' for month and day
* nulls are dropped
* set track id to index
* change dtypes to correct type
* fix tempos
    * From Kwame: "As a hip-hop artist and producer, I know firsthand how BPM (beats per minute, aka the tempo of a song) can often be miscalculated as twice their actual value. This is because most song tempos fall in-between 90 and 160 BPM, and a computer can wrongly detect tempo as double-time in slower tempos below 90. There are some genres that have faster BPM, such as 170 to 190 for Drum ’n’ Bass, however, in Hip-Hop I’ve found that the BPM is wrongly miscalculated in this way when it’s shown as 170 and above. Therefore, in our data, I chose to halve the tempos of all tracks with 170 BPM or greater for a more accurate look at tempo."

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

![Subgenre Popularity](https://github.com/SpotiScryers/SpotiScry/blob/main/images/Genre_with_Title.png?raw=true)

![Popular Tempos](https://github.com/SpotiScryers/SpotiScry/blob/main/images/Tempo_with_Title.png?raw=true)

![Popular Key Signatures](https://github.com/SpotiScryers/SpotiScry/blob/main/images/Key_with_Title.png?raw=true)

### Model
First we made a baseline model to compare our model performances. The baseline was based on the average popularity for a track in our train split, which means our baseline prediction came out to a popularity of 38. The baseline model had an RMSE of 22.8 on the train split. We created various regression models and fit to the train data.  

**Feature Groups**
We used three sets of feauture groups. 
- Select K best: selects features according to the k highest scores (top 5)
- Recursive Feature Elimination: features that perform best on a simple linear regression model (top 5)
- Combination (unique features from both groups, 7 features)

**Models Evaluated**
* OLS Linear Regression
* LASSO + LARS
* Polynomial Squared + Linear Regression
* Support Vector Regression using RBF Kernel
* General Linear Model with Normal Distribution

**Evaluation Metric**  
Models are evaluated by calculating the root mean squared error (RMSE) or residual of the predicted value to the actual observation. The smaller the RMSE, the better the model performed. A visual of this error is below.  
![Model-Error](https://github.com/SpotiScryers/SpotiScry/blob/main/images/model_error.png?raw=true)

**Final Model:**  
Polynomial Squared + Linear Regression was our final model we performed on test, predicting 6% better than the baseline.  

| Model                         | Train RMSE | Validate RMSE | Test RMSE |
|-------------------------------|------------|---------------|-----------|
| Polynomial 2nd Degree         | 21.599581  | 21.5257       | 21.5236   |
| OLS Linear Regression         | 21.796331  | 21.7566       |           |
| Support Vector Regression     | 21.812662  | 21.6988       |           |
| General Linear Model - Normal | 21.821093  |               |           |
| Baseline - Average            | 22.897138  |               |           |
| LASSO + LARS                  | 22.897138  |               |           |  

**How It Works:**  
Polynomial Regression: a combination of the Polynomial features algorithm and simple linear regression. Polynomial features creates new variables from the existing input variables. Using a degree of 2, the algorithm will square each feature, take the combinations of them, and use the results as new features. The degree is a parameter that is a polynomial used to create a new feature. For example, if a degree of 3 is used, each feature would be cubed, squared, and combined with each other feature. Finally, a regression model is fit to the curved line of best fit depending on the degree. An example of determining best fit is below.  

![Model_Evaluation](https://github.com/SpotiScryers/SpotiScry/blob/main/images/Model_Evaluation.png?raw=true)

### Conclusions  
Key drivers for popularity include **danceability with speechiness**, whether a track is **explicit**, **energy**, **track number**, and whether a track has **featured artists** or not. The best performing model was our **2nd Degree Polynomial Regression** model with an RMSE of **21.5236** on the testing dataset. The most popular songs were about ~2 minutes long.

<kbd>[Back to Table of Contents](https://github.com/SpotiScryers/SpotiScry#table-of-contents)</kbd>
## How to Reproduce  
### Steps  
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

### Tools & Requirements
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
## License
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
## Creators
[Brandon Martinez](https://github.com/Brandon-Martinez27), [Bethany Thompson](https://github.com/ThompsonBethany01), [Kwame V. Taylor](https://github.com/KwameTaylor), [Matthew Mays](https://github.com/Matthew-Mays)  
<kbd>[Back to Table of Contents](https://github.com/SpotiScryers/SpotiScry#table-of-contents)</kbd>
