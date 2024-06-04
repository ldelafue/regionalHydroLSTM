# Regionalization

By using the Code or the HydroLSTM representation in your publication(s), you agree to cite:

> *De la Fuente, L. A., Gupta, H. V., and Condon, L. E.: XXX, XXX [preprint], XXX, 2024.*
> *De la Fuente, L. A., Ehsani, M. R., Gupta, H. V., and Condon, L. E.: Toward interpretable LSTM-based modeling of hydrological systems, Hydrology and Earth System Sciences, 28(4), 945-971, https://doi.org/10.5194/hess-28-945-2024, 2024.*

This repository is splited in 4 different sections.
  - Data
  - Codes
  - Results
  - Notebooks

## Getting started

To run the code you must set up the conda environment first. You must use the follow lines to clone the repo and create the environment (the last step may take some time):
```
git clone git@github.com:ldelafue/regionalization.git
cd regionalization
conda env create -f environment.yml
```

### Data

There are two sources of data needed to run all the scripts and notebooks. The first one if the raw data that can be found in the data folder. This folder contains the three sources of data used in this paper (USGS, CAMELS attributes, CAMELS time series). That information is completely available in this repository, so you do not need to request or download information from another source.

By using the CAMELS attributes in your publication(s), you agree to cite:

> *Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrology and Earth System Sciences, doi:10.5194/hess-2017-169, 2017.*

By using the CAMELS time series in your publication(s), you agree to cite:

> *Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., Viger, R., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T. and Duan, Q.: Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional variability in hydrologic model performance, Hydrology and Earth System Sciences, 19, 209–223, doi:10.5194/hess-19-209-2015, 2015.*

The second source of data are the results presented in the paper. This data must be download in the folder in order of running the jupyter notebook with the figures of the paper. You can do it by running the following lines:

```
cd regionalization
XXXX
```







  

 
To run the code you can use the provided conda environment. You must use the follow lines to clone the repo and create the environment (the last step may take some time):
```
git clone git@github.com:ldelafue/regionalization.git
cd regionalization
conda env create -f environment.yml
```

Once you have the environment set up and the data downloaded, you can train a model with the code. You have two options. You can train a single catchment with the HydroLSTM repsentation:
```
conda activate regionalization
cd Codes
python main.py --code 9223000 --cells 1 --memory 256 --epochs 10 --model HYDRO
```

The second option is training a the regional HydroLSTM with the 569 catchments used in the paper.
```
conda activate regionalization
cd Codes
python main.py --code 9223000 --cells 1 --memory 256 --epochs 10 --model HYDRO   ??????????????
```


To see the available options, you can use:
```
python main.py -h, --help
```

### Data
This folder contains the three sources of data used in this paper (USGS, CAMELS attributes, CAMELS time series). That information is completely available in this repository, so you do not need to request or download information from another source.

By using the CAMELS attributes in your publication(s), you agree to cite:

> *Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrology and Earth System Sciences, doi:10.5194/hess-2017-169, 2017.*

By using the CAMELS time series in your publication(s), you agree to cite:

> *Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., Viger, R., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T. and Duan, Q.: Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional variability in hydrologic model performance, Hydrology and Earth System Sciences, 19, 209–223, doi:10.5194/hess-19-209-2015, 2015.*


### Codes
This forlder has the 5 python code used in the training and evaluation (testing) of the results of the models.
  - main.py: This code is calling all the other files. This script has some parameterization such as gauge ID, #cells, #memory (days), learning rate, #epochs, and the model used (LSTM or HYDRO)
  - LSTM.py and Hydro_LSTM.py: They create a class with the specific structure. Its equation can be found in this script.
  - importing.py: This script create the dataset with the data from the thre sources.
  - utils.py: This script has some specific functions used to create the datset and train the model.

Moreover, the folder has the Anaconda environment used to run the codes, and a txt file with an example about how to run the main.py script using comand lines in a terminal.

### Results
This folder has the summary results for each structure used. Hydro and LSTM refers to the experiment with 10 catchments. Hydro_CONUS refers to the experiment with 587 catchment using Hydro_LSTM. In the cases where the weight distribution is analyzed, The folder has the actual model saved in a pkl file.

### Notebooks
This folder has the files used to create each of the figures presented in the paper. All the figures are contained in the jupiter notebook Figures.ipynb.


