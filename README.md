bodyweight-pose-estimation
==============================

Classify live bodyweight exercises via 3D body pose estimation

Project Organization
------------

``` 
├── LICENSE
├── README.md
├── models
├── notebooks
│  └── draft.ipynb
├── poetry.lock
├── pyproject.toml
├── references
├── reports
│  └── figures
└── src
   ├── __init__.py
   ├── data
   │  ├── __init__.py
   │  └── make_dataset.py
   ├── distances
   │  ├── __init__.py
   │  └── lcss.py
   ├── features
   │  ├── __init__.py
   │  └── build_features.py
   ├── models
   │  ├── __init__.py
   │  ├── predict_model.py
   │  └── train_model.py
   ├── resources
   │  ├── __init__.py
   │  ├── joints.json
   │  ├── joints.py
   │  └── smoothers.py
   ├── utils
   │  ├── __init__.py
   │  ├── miscellanous.py
   │  └── trigonometry.py
   └── visualization
      ├── __init__.py
      └── visualize.py
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
