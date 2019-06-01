<p align="center">
  <img src=./imgs/oscars.jpg width="400">
</p>

# Project PTO - Building a Computer Algorithm to Predict the Oscars

## Overview
This project was a month-long side project I did for my highschool final project. PTO is a computer algorithm using machine learning and data science to determine which movies will be oscar nominees and winners. I have created a flexible dataset for anyone in the community to try and get better results!

## Documentation
A jupyter notebook (and its HTML file for easy viewing) under the name "predicting_the_oscars.ipynb" is included in this repo explaining in detail my process.

## Contents
The project consists of 4 major sections: 
1. Data Collection
2. Data Manipulation/Organization
3. The Algorithm
4. Algorithm Optimization

## File Descriptions
data/ - contains datasets that were created and used, as well as pickled numpy arrays
imgs/ - images used in the jupyter noteboook
best.h5 - best performing saved model in h5 format
collect_data.py - contains functions that main.py uses to collect/web-scrape movie data
main.py - the main file of this project. 
predicting_the_oscars.html - HTML version of the jupyter notebook
predicting_the_oscars.ipynb - jupyter notebook

## Usage
The project was programmed with Anaconda Python 3.6 along with the sklearn and keras packages.

To use, simply clone this repo in any directory and run:
```
python3 main.py
```

## License
Code is released under the MIT License.
