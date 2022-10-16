bodyweight-pose-estimation
==============================

Classify live bodyweight exercises via 3D body pose estimation

Project Organization
------------

```
.flake8
.gitignore
LICENSE
README.md
├─ data
│  ├─ processed
│  │  ├─ pullup
│  │  │  ├─ VID_20221015_170933_Trim.csv
│  │  │  ├─ VID_20221015_171207_Trim.csv
│  │  │  ├─ VID_20221015_171347_Trim.csv
│  │  │  └─ VID_20221015_171534_Trim.csv
│  │  ├─ pushup
│  │  │  ├─ VID_20221015_171647_Trim.csv
│  │  │  ├─ VID_20221015_171750_Trim.csv
│  │  │  ├─ VID_20221015_171911_Trim.csv
│  │  │  ├─ VID_20221015_173237_Trim.csv
│  │  │  ├─ petal_20221015_233600.csv
│  │  │  └─ petal_20221015_233640.csv
│  │  └─ squat
│  │     ├─ VID_20221015_172629_Trim.csv
│  │     ├─ VID_20221015_172722_Trim.csv
│  │     ├─ VID_20221015_173029_Trim.csv
│  │     ├─ VID_20221015_173317_Trim.csv
│  │     ├─ petal_20221015_233712.csv
│  │     └─ petal_20221015_233751.csv
│  └─ raw
│     ├─ pullup
│     │  └─ .gitkeep
│     ├─ pushup
│     │  └─ .gitkeep
│     └─ squat
│        └─ .gitkeep
├─ models
│  └─ .gitkeep
├─ notebooks
│  └─ draft.ipynb
├─ poetry.lock
├─ pyproject.toml
├─ references
│  └─ .gitkeep
├─ reports
│  ├─ .gitkeep
figures
│     └─ .gitkeep
├─ src
│  ├─ __init__.py
│  ├─ data
│  │  ├─ __init__.py
│  │  ├─ angle_series.py
│  │  ├─ coordinate_series.py
│  │  ├─ make_dataset.py
│  │  └─ video.py
│  ├─ distances
│  │  ├─ __init__.py
│  │  └─ lcss.py
│  ├─ features
│  │  ├─ __init__.py
│  │  └─ build_features.py
│  ├─ models
│  │  ├─ __init__.py
│  │  ├─ model.py
│  │  ├─ predict_model.py
│  │  └─ train_model.py
│  ├─ resources
│  │  ├─ __init__.py
│  │  ├─ joints.json
│  │  └─ joints.py
│  ├─ utils
│  │  ├─ __init__.py
│  │  ├─ miscellanous.py
│  │  └─ trigonometry.py
│  └─ visualization
│     ├─ __init__.py
│     └─ visualize.py
└─ tests
   └─ .gitkeep
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
