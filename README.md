# movie-review-sentiment-classifier

Project Setup Commands
1. Install Python 3 (if not already installed)
On Ubuntu/Debian:
Apply to test_preproc...
Run
2. Clone or Copy Your Project Directory
If using git:
Apply to test_preproc...
Run
Or copy the folder manually.
3. Create and Activate a Virtual Environment
Apply to test_preproc...
Run
Note: Always activate your venv before installing or running anything.
4. Upgrade pip (recommended)
Apply to test_preproc...
Run
5. Install All Required Libraries
Apply to test_preproc...
Run
Note: joblib is only needed if you want to save/load models.
6. (Optional) Install Additional Tools for Development
For running tests:
Apply to test_preproc...
Run
or just use Python’s built-in unittest (no extra install needed).
7. Download NLTK Data (if not already downloaded)
In a Python shell or at the top of your script (run once):
Apply to test_preproc...
Note: This ensures NLTK’s stemmer and stopwords work.
8. Install Jupyter Notebook (Optional, for EDA)
Apply to test_preproc...
Run
9. Run Your Project
To run the main pipeline:
Apply to test_preproc...
Run
10. Run Unit Tests
Apply to test_preproc...
Run
or
Apply to test_preproc...
Run
11. View Confusion Matrix Plots
If running on a local machine with GUI, plots will pop up.
If on a server/headless, open the saved .png files in your project directory.
12. (Optional) Freeze Requirements for Reproducibility
Apply to test_preproc...
Run
Use this file to install exact versions on another machine:
Apply to test_preproc...
Run
Summary Table
Step	Command	Note
1	sudo apt install python3 python3-venv python3-pip	Install Python 3 and tools
3	python3 -m venv .venv<br>source .venv/bin/activate	Create & activate venv
4	pip install --upgrade pip	Upgrade pip
5	pip install pandas scikit-learn matplotlib seaborn nltk joblib	Install libraries
7	import nltk; nltk.download('punkt'); nltk.download('stopwords')	Download NLTK data
9	python3 src/main.py	Run main pipeline
10	python3 -m unittest discover tests	Run all tests
12	pip freeze > requirements.txt	Save requirements
