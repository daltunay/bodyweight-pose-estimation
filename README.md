bodyweight-pose-estimation
==============================

_Classify live bodyweight exercises via 3D body pose estimation_

The purpose of this package is to classify an input webcam video into three exercise labels :
- **push up**
- **pull up**
- **squat**

The classifier consists in a k-NN, which takes as an input the multi-dimensional time series of the body angles evolution over time.  
In order to compute a distance between these multi-dimensional time series, an adaptation of the _Dynamic Time Warping (DTW)_ was used : the _Longest Common Subsequence (LCSS)_.

# Input data

If you want to you your own training data, you just need to add `.mp4` video files in the corresponding folders in `/data/raw/`. This implies doing the **dataset creation** and **model training** steps.  
Otherwise, a pre-trained model (`/models/model.pkl`) with my own training data will be used.

# Usage

## Initialization

First, install poetry :
```
pip install poetry
```

Then, choose the right version (**3.8.10**) of `python` for `poetry` to use :
```
poetry env use /<PATH_TO_PYTHON_3.8.10>/python.exe
```

Then, create and activate a virtual environment :
```
poetry shell
```

Install the packages and their dependencies :
```
poetry install
```

You are now ready to go !

## Dataset creation (optional)

The purpose of this step is to preprocess the videos in `/data/raw/`, and export the output as `.csv` files in `/data/processed/`.

```
poetry run python src/data/make_dataset.py
```

## Model training (optional)
The purpose of this step is to train the model on the newly created dataset. The pickled model is exported as `/models/model.pkl`.
```
poetry run python src/models/train_model.py
```

## Run the app
In order to try the application, run the following command :
```
poetry run python app.py
```
A webcam window will pop up, in which you will be able to see yourself. Then, do the exercise you want (either **push ups**, **pull ups**, or **squats**) and press `Esc` to close the window.  
The script will print the probabilities of each label in the terminal.