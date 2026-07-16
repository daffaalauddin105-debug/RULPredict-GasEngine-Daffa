# RUL Prediction for Gas Engine Compressor Using Deep Learning

## Abstract
Gas engine is one of the critical equipment in the oil and gas industry, making remaining useful life (RUL) prediction using deep learning essential to prevent unplanned downtime and ensure operational availability. However, the majority of existing studies rely on vibration data from ideal simulation rigs, while the utilization of real field temperature data remains scarce. This study aims to evaluate RUL prediction performance using mean absolute error (MAE) and root mean square error (RMSE), while also investigating the effect of outlier conditions and sensor combination variations on prediction results. A data driven approach based on long short-term memory (LSTM) was employed, with principal component analysis (PCA) applied to construct the health indicator (HI). Three sensor groupings were defined: Combustion (five exhaust cylinder temperature sensors), Systemic (jacket water temperature and exhaust manifold temperature for both right and left banks), and Global (a combination of Combustion and Systemic sensors), each evaluated under two data conditions: with and without outlier removal. Results show that the Combustion variation without outlier removal achieved the best performance, with an RMSE of 42.89 hours and MAE of 35.27 hours. This variation also demonstrated generalizability to different operating conditions, yielding an RMSE of 24.00 hours and MAE of 20.25 hours after fine tuning. This study concludes that sensor groupings within a single, strongly correlated system yield the lowest prediction error and maintain performance across different operating conditions. Furthermore, outlier removal is not universally beneficial. It reduces error in low-correlation, high-sensor-count fusion scenarios but can cause LSTM model failure when applied to single-system sensor groups.

## General Information
This repository contains the source code, datasets, and pipeline for predicting the Remaining Useful Life (RUL) of a Gas Engine Compressor. This project is part of an undergraduate thesis (Tugas Akhir) focused on leveraging deep learning models and logged temperature data to predict equipment health.

## Project Structure
* `Data RTF Mentah/` - Raw log data from the field (.rtf format).
* `Data Siap Tampil/` - Extracted, filtered, and processed data visualizations.
* `K101 Data/` - Main temperature logger datasets (.csv format).
* `Preprocess/` - Scripts for data cleaning and feature engineering.
* `Testing/` - Model evaluation scripts and validation datasets.
* `Training/` - Deep learning model training configurations and saved weights.
* `Testing/GUI_New.py` - User-friendly Graphical User Interface (GUI) linked to the best performing model.
* `requirements.txt` - Python environment dependencies.

## Academic & Usage Notice
This repository is set to **No License**. The intellectual property belongs entirely to the author. 
However, **full academic permission** is granted to internal students and researchers from the same institution to download, modify, and build upon this work for non-commercial academic purposes, provided that proper citation to the original thesis is included. Contact daffaalauddin105@gmail.com for inquiries.

## AI Generative Disclosure & Claim
* **AI Use Disclaimer**: Parts of the source code, data preprocessing pipeline, GUI development, and web deployment scripts within this repository were generated, optimized, or debugged with the assistance of Generative AI tools (including Gemini/Claude). 
* **Human Oversight**: All AI-assisted code blocks have been thoroughly reviewed, modified, and validated by the author to ensure scientific accuracy, structural integrity, and alignment with engineering domain logic.
