## Titanic-Machine Learning from disaster with deployment <br>
Work Flow--

    requirements.txt and setup.py:
        Create a requirements.txt file listing the project dependencies.
        Optionally, create a setup.py file for packaging and distribution.

    Notebooks:
        Begin your work in Jupyter Notebooks for data exploration, analysis, and model development.
        Save and organize your notebooks in a dedicated directory, such as notebooks/.

    Source Code (src):
        Move your code from notebooks to a structured source code directory (e.g., src/).
        Divide the code into modules or scripts for better organization.

    Application Setup (app.py, templates, static):
        Create an app.py file for your Flask application.
        Organize HTML templates in a templates/ directory.
        Place static files (like images or CSS) in a static/ directory.

    Documentation and Version Control:
        Write a comprehensive README.md to document your project, including instructions for running the application and any other relevant information.
        Create a .gitignore file to specify files or directories that should be ignored by version control.

*** Project Structure" ***

titanic_ml_from_disaster/
|-- notebooks/
|   |-- titanic_EDA.ipynb
|   |-- model_train.ipynb
|   |-- submission.csv
|   |-- test_prepare_data.csv
|   |-- train_cleaned_data.csv
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- load_data.py
|   |-- preprocessing.py
|   |-- outlier_handling.py
|   |-- features_eng.py
|   |-- train_model.py
|   |-- predict_model.py
|-- data/
|   |--model
|   |  |-- rf_pipeline.pkl
|   |-- preprocessed_data
|   |   |-- cleaned_train_data.csv
|   |-- raw_data
|       |-- train_titanic.csv
|-- deploy/
|   |-- __init__.py
|   |-- cleaned_data.csv
|   |-- rf_pipeline_dep.pkl
|   |-- deploy_train_model.py
|-- static/
|   |-- not_survive.jpg
|   |-- ship_titanic.jpg
|   |-- survive.jpg
|   |-- styless.css
|-- templates/
|      |-- index.html
|      |-- result.html
|      |-- error.html
|-- app.py
|-- README.md
|-- requirements.txt
|-- setup.py
|-- .gitignore
