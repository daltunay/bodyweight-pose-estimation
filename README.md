bodyweight-pose-estimation
==============================

Classify live bodyweight exercises via 3D body pose estimation

# Usage

## 1. Initialization

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
poetry install
```

You are now ready to go !

## 2. Make the dataset

Run the script via the following command :
```
poetry run python src/data/make_dataset.py
```

## 3. Train the model

Run the script via the following command :
```
poetry run python src/models/train_model.py
```

## 4. Run the app

Run the script via the following command :
```
poetry run python app.py
```