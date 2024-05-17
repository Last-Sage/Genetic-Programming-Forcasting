# Genetic Programming for Sales Prediction

This repository contains code for predicting sales using Genetic Programming (GP). 

## Overview

Genetic Programming (GP) is an evolutionary algorithm-based technique for solving optimization problems. In this project, GP is employed to predict sales based on various input features such as price, discount, advertisement, and cost.

## Prerequisites

Ensure you have the following libraries installed:
- `csv`
- `random`
- `operator`
- `matplotlib`
- `deap`
- `numpy`
- `pandas`
- `sklearn`

## Usage

1. Clone the repository:
   ```bash
   git clone git clone https://github.com/Last-Sage/Genetic-Programming-Forcasting.git

2. Navigate to the project directory:

    ```bash
    cd Genetic-Programming-Forcasting

3. Run the main Python script:

    ```bash
    python sales_prediction_gp.py

## Importing Data

The training and testing data should be provided in CSV format. Ensure that the file paths to the training and testing data are correctly specified in the script.


## Genetic Programming Settings and other Parameter Configuration

| Parameter               | Description                                                                                      |
|-------------------------|--------------------------------------------------------------------------------------------------|
| `individual_size`        | The size of each individual in the population.                                                  |
| `population_size`         | The size of the population.                                                                     |
| `generations`             | The number of generations for which the genetic algorithm will run.                             |
| `cxpb`                    | Crossover probability.                                                                          |
| `mutpb`                   | Mutation probability.                                                                           |
| `tournsize`               | Tournament size for selection.                                                                   |
| `file_name`               | File path to the CSV file containing training and testing data.                                  |
| `attr_price`              | Range for generating random values for the 'price' attribute.                                    |
| `attr_discount`           | Range for generating random values for the 'discount' attribute.                                 |
| `attr_advertisement`      | Range for generating random values for the 'advertisement' attribute.                            |
| `attr_cost`               | Range for generating random values for the 'cost' attribute.                                      |


## Fitness Function

The fitness function evaluates the performance of each individual in the population based on the mean squared error (MSE) between the predicted and actual sales. The fitness is minimized.


## Visualization

```visualize_data``` : Scatter plot of input data for verification.  
```visualize_predictions``` : Line plot comparing actual sales with predicted sales.  


## Results

The script outputs various performance metrics:  <br>


| Metric                           | Description                                                                                            |
|----------------------------------|--------------------------------------------------------------------------------------------------------|
| `Mean squared error (MSE)`        | The average of the squares of the differences between predicted and actual values.                      |
| `R-squared value`                  | A statistical measure indicating the proportion of the variance in the dependent variable that is predictable from the independent variables. |
| `Mean absolute error (MAE)`       | The average of the absolute differences between predicted and actual values.                           |
| `Root mean squared error (RMSE)`  | The square root of the average of the squares of the differences between predicted and actual values.   |
| `Mean percentage error (MPE)`     | The average of the percentage differences between predicted and actual values.                          |
| `Mean absolute percentage error (MAPE)` | The average of the absolute percentage differences between predicted and actual values.              |
| `Root mean squared percentage error (RMSPE)` | The square root of the average of the squares of the percentage differences between predicted and actual values. |
| `Coefficient of variation (CV)`   | The ratio of the standard deviation to the mean, expressed as a percentage.                             |



# Run this project in Google Colab
https://colab.research.google.com/github/Last-Sage/Genetic-Programming-Forcasting/blob/main/genetic_programming_colab.ipynb
