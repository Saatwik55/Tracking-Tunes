# Tracking-Tunes
A music recommender system that takes a song from user and generates similar songs based on the genre predicted.
The system at its core is a neural network (currently CNN) analysing the temporal features of audio of each song in Free Music Archive (FMA) - https://github.com/mdeff/fma
Many further improvements are required for the model such as using spectograms as input instead of numerical features for CNN.
Current accuracy of predicting the correct genre in the top 3 prediction ~82%.
