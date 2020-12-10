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
- Presentation slides
- GitHub repository with analysis

### Timeline
- [X] Project Planning: December 8th
- [ ] Aquisition and Prep: December 10th
- [ ] Exploration: December 14th
- [ ] Modeling: December 15th
- [ ] Finalize Minimum Viable Product (MVP): EOD December 15th
- [ ] Improve/Iterate MVP: December 16th
- [ ] Finalize Presentation: December 17th

### Acknowledgments
## Data Dictionary
- artists: The artists who performed the track
- album: The album in which the track appears
- track_name: The name of the track
- track_id: The spotify ID for the track
- danceability: A value of 0 - 1 that represents a combination of tempo, rhythm stability, beat strength, and overall regularity
- energy: A value of 0 - 1 that represents a perceptual measure of intensity and activity. The faster, louder, noisier a track is the higher the energy
- key: The estimated overall key of the track, integers map to pitches using standard Pitch Class notation. If no key was detected, value is -1.
    - 0 = C
    - 1 = C#
    - 2 = D
    - etc.
- The overall loudness of a track in decibels (dB). Values typically range between -60 and 0
- mode: The modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major = 1, Minor = 0
- speechiness: A value of 0 - 1 that represents how exclusively speech-like the recording is. Values above .66 are made almost entirely of spoke words, .33 - .66 values may contain both music and speech, either in sections or layered. Values .33 most likely represent music and other non-speech-like tracks.
- instrumentalness: Predicts whether a track contains no vocals, The close the instrumentalness value is to 1 the greater the likelihood the track contains no vocal content. Values above .5 are intended to represent instrumental tracks, but confidence is higher as the value aproaches 1.
- liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- valence: A measure from 0 - 1 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (happy, cheerful, euphoric), while tracks with low valence sound more negative (sad, depressed, angry).
- tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
- duration_ms/duration_s: The duration of the song in ms or s respectively
- time_signature: An estimated overall time signature of a track, the time signature is a notational convention to specify how many beats are in each bar.
- release_date: The date the album was first released, if only year was given as precision it defaults to YYYY-01-01
- popularity: Target variable, value between 0 - 100 that measures how many views the track has gotten in relation to how current those views are.
- explicit: Boolean variable for whether or not the track has explicit lyrics.
## Initial Thoughts & Hypotheses
### Thoughts
### Hypotheses
## Project Steps
### Acquire
### Prepare
### Explore
### Model
### Conclusions
## How to Reproduce
### Steps
### Tools & Requirements
## License
## Creators
Brandon Martinez, Bethany Thompson, [Kwame V. Taylor](https://github.com/KwameTaylor)
