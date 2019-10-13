Running the Algorithms

All of the algorithms, related charts, and cross validation result data can be found at: https://github.com/jagrusy/RandomizedOptimization

All of the project requirements are in the `requirements.txt` file

`pip install -r requirements.txt`

The version of mlrose that is used is forked from: https://github.com/parkds/mlrose and the full library is included in the repo.

The code for each algorithm is contianed in the aptly named files

Running an algorithm (continuous_peaks.py, traveling_salesman.py, four_peaks.py, ANN_optimization.py) file will generate the associated
charts which will be saved in the `Figs` folder.

There are additional experiments for tuning certain parameters that are named by the experiment name followed by the algorithm they are tuning and the problem they are solving

Each algorithm file utilizes the `util.py` file to gather and preprocess the data. In order for the data to be
processed correctly it should be stored in the `Data` folder.
