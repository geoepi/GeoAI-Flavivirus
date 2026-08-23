# GeoAI-Flavivirus

Code and supporting materials for **Leveraging recurrent graph neural networks to improve geospatial estimation of equine West Nile Virus outbreaks**.  

Data archived at the Open Science Framework [https://osf.io/79rne/overview](https://doi.org/10.17605/OSF.IO/79RNE)  

This project evaluates how geographic structure, temporal dependence, environmental conditions, vector and host ecology, and landscape characteristics can be integrated to characterize reported equine WNV occurrence across the U.S. Southern Climate Region.

## Overview

West Nile virus transmission varies substantially across both space and time. Temperature, precipitation, drought, vegetation, land cover, host communities, and mosquito ecology all contribute to this variation, while observations from neighboring locations and preceding time periods are not independent.

The modeling framework implemented here uses a recurrent graph neural network to represent these dependencies explicitly. U.S. counties are represented as nodes in a spatial graph, with neighboring counties connected using Queen's contiguity. Graph convolutional layers allow information to be shared among neighboring counties, while a long short-term memory (LSTM) component represents temporal dependence among epidemiological weeks.

The analysis focuses on reported equine WNV occurrence from 2002–2019 in Kansas, Oklahoma, Texas, Arkansas, Louisiana, and Mississippi. Environmental and ecological predictors include temperature, precipitation, drought, vegetation, land cover, habitat heterogeneity, topography, avian species richness, horse abundance, and WNV surveillance information from birds and mosquitoes.

The primary objectives are to:

* quantify spatial and temporal patterns associated with reported equine WNV occurrence;
* evaluate whether explicitly representing geographic and temporal dependence improves model performance;
* identify environmental and ecological variables that contribute most strongly to model discrimination; and
* assess model performance in held-out years representing both an unusually high-incidence period and later observations.

This repository is a demonstrative workflow rather than as a general-purpose software package.

## Repository contents

* `GLSTM_analysis.ipynb` — primary analysis notebook demonstrating data preparation, model fitting, evaluation, and interpretation.
* `GLSTM_models.py` — graph neural network and related model definitions used for training and hyperparameter tuning.
* `GLSTM_training.py` — model training and evaluation functions.
* `GLSTM_utils.py` — supporting utilities used throughout the analysis.
* `osf_data_fetch.py` — helper code for accessing project data stored outside the GitHub repository.
* `Figures/` — figures and graphical outputs associated with the analysis.
* `requirements.txt` — Python dependencies used by the analysis.

## Modeling framework

The central model combines graph convolution with recurrent neural-network structure. Spatial adjacency allows conditions in one county to inform representations of neighboring counties, while the LSTM component allows the model to retain information from preceding time steps.

Model performance is compared with a non-spatiotemporal neural-network baseline to evaluate the contribution of explicitly representing spatial and temporal dependence. Predictor importance is subsequently evaluated using permutation importance, providing an estimate of how strongly model discrimination depends on individual environmental and ecological variables.
