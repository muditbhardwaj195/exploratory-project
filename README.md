## Exploratory Project

### Objective
The objective of this project is the early classification of a multivariate time series with different sampling rates using divide-and-conquer approach, while mantaining the `alpha` accuracy.

### Dataset

Daily and Sports Activities Data Set is used in this project.
[Link](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)

### Algorithms Used
- Gaussian Process Classifier
- Agglomerative Hierarchical Clustering
- Divide and Conquer Approach for label prediction

### Running the code

Execute the following commands to run this code
- `python main.py`

### Summary
- I was able to achieve `93.25%` accuracy and an earliness of `15.09`.
- I used `80%` of the dataset for training and `20%` for testing.
- `ALPHA = 0.9`
- I used `5` sensors from the original dataset.

### File Structure
- **datasets/** - These file contain dataset for ith sensor.
- **mrls/** - This contains the mrls with different values of `alpha` and `training fraction`.
- **create_dataset.py** - This file is responsible for creating all the sensors dataset and the dataste for testing purpose.
- **gaussian_process_classifier.py** - This file contains all the code related to `gaussian_process_classifier`.
- **time_series_prediction.py** - This file implements divide and conquer approach for time series prediction.
- **load_dataset.py** - This file contains the functions to load the datasets in a format that they can be used from the `datasets` directory.
- **load_final_mrl.py** - This file contains the functions to load the final_mrls calculated previously in a format that they can be used for the prediction purpose.
- **main.py** - This file runs different sections of this project.
- **pre_processing.py** - This file was used to convert the original dataset to the format present in the `datasets` directory.

### Final Output
```
SUMMARY
--------------------  ------------------------------------------------------------------------
Algorithms Used       Gaussian Process Classifier, Agglomerative Hierarchical Clustering,Divide and Conquer Approach for label prediction
Accuracy              95.45%
Earliness             15.09%
Sensors               5
Training fraction     0.8
Alpha                 0.9
--------------------  ------------------------------------------------------------------------
