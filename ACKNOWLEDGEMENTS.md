# Acknowledgements

This project uses publicly available datasets and builds upon prior research in predictive maintenance and prognostics.

## Datasets

### 1. NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS)

**Citation:**
```
A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
```

**Source:**
- NASA Prognostics Center of Excellence Data Repository
- Direct download: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- NASA Open Data Portal: https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data
- Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

**Description:**
The C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset consists of multivariate time series from turbofan engine simulations. The dataset includes run-to-failure trajectories under various operational conditions and fault modes, making it ideal for remaining useful life (RUL) prediction and predictive maintenance research.

**Dataset Details:**
- 4 sub-datasets with increasing complexity
- Multiple operational settings and fault modes
- 21 sensor measurements per time step
- Training and test sets with known and unknown failure times

### 2. PHM Society Data Challenge Datasets

**Source:**
- PHM Society Data Repository: https://data.phmsociety.org/
- 2025 EHM Data Challenge: https://data.phmsociety.org/phm-north-america-2025-conference-data-challenge/

**Description:**
The Prognostics and Health Management (PHM) Society hosts annual data challenges focusing on various aspects of predictive maintenance, including bearing fault diagnosis, battery degradation, and engine health monitoring.

**Recent 2025 Challenge:**
- Commercial jet engine maintenance prediction
- Sensor data from 8 engines across 2001 flights
- 16 sensed engine variables
- Targets: Remaining cycles to three maintenance events

### 3. Kaggle Predictive Maintenance Datasets

#### IoT-Integrated Predictive Maintenance Dataset
**Source:** https://www.kaggle.com/datasets/ziya07/iot-integrated-predictive-maintenance-dataset
**Year:** 2025
**Features:** Time-series IoT sensor data with fault labels for machine failure prediction

#### Machine Predictive Maintenance Classification
**Source:** https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
**Year:** 2021
**Features:** Synthetic dataset modeled after real predictive maintenance data with binary and multiclass failure prediction

#### Machine Failure Prediction using Sensor Data
**Source:** https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data
**Year:** 2024
**Features:** Sensor readings with machine failure labels

## Research References

### Key Papers in Predictive Maintenance

1. **Deep Learning for RUL Prediction:**
   - "A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset" (2022)
   - ResearchGate: https://www.researchgate.net/publication/363217802

2. **LSTM-based Approaches:**
   - Multiple implementations available on Kaggle demonstrating LSTM networks for predictive maintenance
   - Example: https://www.kaggle.com/code/nafisur/predictive-maintenance-using-lstm-on-sensor-data

3. **Comprehensive Dataset Review:**
   - Fabian Mauthe, Luca Steinmann, and Peter Zeiler. "Overview and analysis of publicly available degradation data sets for tasks within prognostics and health management." 35th European Safety and Reliability Conference (2025).
   - arXiv: https://arxiv.org/html/2403.13694v2

4. **Responsible AI in Predictive Maintenance:**
   - Balamurugan Balakreshnan. "Responsible AI in Predictive Maintenance — Using NASA Turbofan Engine Degradation Dataset" (Medium)
   - Link: https://balabala76.medium.com/responsible-ai-in-predictive-maintenance-using-nasa-turbofan-engine-degradation-dataset-using-e386b49355e5

5. **GitHub Resources:**
   - exploring-nasas-turbofan-dataset: https://github.com/kpeters/exploring-nasas-turbofan-dataset
   - PHM_datasets: https://github.com/hustcxl/PHM_datasets
   - phm_public_data: https://github.com/ShaunRBK/phm_public_data

## Tools and Libraries

### PHMD - PHM Data Access Library
**Source:** https://www.sciencedirect.com/science/article/pii/S2352711025000068
**Description:** Python library for seamless access to 59+ PHM datasets from diverse domains

## Machine Learning Frameworks
- TensorFlow/Keras: Deep learning implementation
- PyTorch: Alternative deep learning framework
- scikit-learn: Classical machine learning algorithms
- XGBoost: Gradient boosting for tabular data
- SHAP: Model interpretability and explainability

## Community and Support

Special thanks to:
- NASA Prognostics Center of Excellence for maintaining open datasets
- PHM Society for organizing data challenges and maintaining repositories
- Kaggle community for sharing datasets and notebooks
- Research community for advancing predictive maintenance methodologies

## License and Usage

This project is for educational and research purposes. Please refer to individual dataset licenses:
- NASA C-MAPSS: Public domain (U.S. Government work)
- PHM Society datasets: Check individual challenge terms
- Kaggle datasets: Refer to dataset-specific licenses

When using this code or datasets, please cite the original sources appropriately.

---

**Last Updated:** October 2025
