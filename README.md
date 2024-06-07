# Regional HydroLSTM

By using the Code or the HydroLSTM representation in your publication(s), you agree to cite:

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
cd your_local_github_repository/regionalization
conda env create -f environment.yml
```

### Data
There are two sources of data needed to run all the scripts and notebooks. The first one if the raw data that can be found in the Data folder. This folder contains the three sources of data needed to train a model(USGS, CAMELS attributes, CAMELS time series). That information is completely available in this repository, so you do not need to request or download information from another source.

By using the CAMELS attributes in your publication(s), you agree to cite:

> *Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrology and Earth System Sciences, doi:10.5194/hess-2017-169, 2017.*

By using the CAMELS time series in your publication(s), you agree to cite:

> *Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., Viger, R., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T. and Duan, Q.: Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional variability in hydrologic model performance, Hydrology and Earth System Sciences, 19, 209–223, doi:10.5194/hess-19-209-2015, 2015.*

The second source of data are the results presented in the paper. This data must be downloaded in your local github folder in order of running the jupyter notebook which creates the paper figures. You can download this data from the following links.

  - neuralhydrology.zip (18.2 MB): https://drive.google.com/uc?export=download&id=1--C6MJf7G1OrM8IHj55OJOI5jQuozp1u
  - RF_mean_0.0.0.0.zip (5.56 GB): https://drive.google.com/uc?export=download&id=13qwiI8Sj4XplXQt712y8NUcdOMU35cJs
  - hydroLSTM.zip (5.46 GB): https://drive.google.com/uc?export=download&id=16wTinNOgK9ItixG0DsD1qqLawKP5OitE

The unzipped folder must be stored in Results folder.

### Codes
This folder has 7 python scripts used in the training and testing.
  - main.py: This code is calling all the other files during training. The script has some parameterization such as gauge ID, #cells, #memory (days), learning rate, #epochs, and the model used (HYDRO or regionalHYDRO)
  - Hydro_LSTM.py, HydroLSTM_regional.py, and HydroLSTM_regional_testing: They create a class with the specific structure. Its equation can be found in this script. The last script is used to read the models saved in Results.
  - importing.py: This script read the data for each catchment and create a dataframe.
  - utils.py: This script has some specific functions used to create the datsaet and train the model.
  - testing.py: This code run the saved model in results to generate the testing results. The script has the same parameterization than main.py

### Training
Once you have the environment set up and the data downloaded, you can train a model with the code. You have two options. You can train a single catchment with the HydroLSTM representation:
```
conda activate regionalization
cd Codes
python main.py --code 9223000 --cells 1 --memory 256 --epochs 5 --model HYDRO
```

The second option is training a the regional HydroLSTM with the 569 catchments used in the paper.
```
conda activate regionalization
cd your_local_github_repository/regionalization/Codes
python main.py --epochs 100 --model regionalHYDRO
```

To see the available options, you can use:
```
python main.py -h, --help
```

The files of the training will be stored in the same folder where the codes are. These are the list of files that will be created by the script:

  - HydroLSTM:
    - code_cells_memory_hydro_models.pkl: it has a list of the ensamble models. By default are 20 models.
    - code_cells_memory_hydro_predictions.csv: Predictions for the training and validation period.
    - code_cells_memory_hydro_state.csv: Values of the internal state variable when you train only one cell.
    - code_cells_memory_hydro_summary.csv: Summary of different metrics during the validation period.

  - Regional HydroLSTM:
    - 1000000_C1_L512_regionalhydro_models.pkl: It has the last HydroLSTM model.
    - 1000000_C1_L512_regionalhydro_predictions.csv: Predictions for the training and validation period for all the catchments.
    - 1000000_C1_L512_regionalhydro_state.csv: Values of the internal state variable for all the catchments.
    - 1000000_C1_L512_regionalhydro_summary.csv: Overall summary of different metrics in the validation period.
    - 1000000_C1_L512_regionalhydro_summary_per_catchment.csv: Summary of different metrics in the validation period for each catchment.
    - 1000000_C1_L512_regionalhydro_RF_model.pkl: it has the RF model to predict the weights for HydroLSTM.
    - 1000000_C1_L512_regionalhydro_RF_regression.pkl: it has the RF model to predict the weights for the linear layer after HydroLSTM.
    - 1000000_C1_L512_regionalhydro_weights.csv: it has the weights predicted by RF in the best epoch.
    - 1000000_C1_L512_regionalhydro_regression.csv: It has the weights predicted by RF regression in the best epoch.          

### Testing
Similar to training, you have 2 options for creating the results in the testing period, HydroLSTM and regional HydroLSTM. In the case of HydroLSTM, the script runs the best model saved in the folder '/Results/hydroLSTM/best model'. Therefore, for any specific catchment you should match the number of cells and memory (lag) of the model saved there.

```
conda activate regionalization
cd Codes
python testing.py --code 1022500 --cells 1 --memory 128 --model HYDRO
```

The second option is evaluating the 569 catchments with the regional HydroLSTM model. This script read the model saved in '/Results/RF_mean_0.0.0.0/'
```
conda activate regionalization
cd Codes
python testing.py --model regionalHYDRO
```
The files of the testing will be stored in the same folder where the codes are. The files are similar to the training but with the suffix 'testing'.

### Results
This folder has the files used in the paper. There are 3 folders containing the results of different runs. hydroLSTM folder has the results of training the 569 catchment by hydroLSTM model. The neuralhydrology folder has the result of training one regional LSTM model for the same 569 catchemnt. The folder RF_mean_0.0.0.0 has the result of running the regional HydroLSTM model for the same catchments.

### Notebooks
This folder has the files used to create each of the figures presented in the paper. All the figures are contained in the jupyter notebook Figures.ipynb.







