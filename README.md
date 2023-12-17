# Hypothesis Evaluator

## Author
**Name:** Matan Amichai

## Introduction
This repository contains the implementation for the Hypothesis Evaluator assignment. The assignment involves creating an algorithm (Empirical Risk Minimization - ERM) that works with data samples and finds the best hypothesis based on different criteria. The main class for the assignment is `HypothesisEvaluator`, which contains various methods for sampling data, running experiments, and evaluating hypotheses.

## Dependencies
The code relies on the following dependencies:
- `numpy`: For handling numerical operations.
- `matplotlib`: For creating plots.
- `intervals`: A custom module for finding the best interval in the data.

Please make sure to install these dependencies before running the code.

```bash
pip install numpy matplotlib
```
HypothesisEvaluator Class


Methods
sample_from_D(m)

This method samples m data samples from the distribution D and returns a 2D array.

experiment_m_range_erm(m_first, m_last, step, k, T)

Runs the Empirical Risk Minimization (ERM) algorithm for a range of data sample sizes. Calculates and plots the average empirical and true errors.

cal_emp_true_error_and_emp(sample, n, k)

Calculates the empirical and true errors based on the given sample, sample size n, and number of intervals k.

experiment_k_range_erm(m, k_first, k_last, step)

Finds the best hypothesis for different values of k. Plots the empirical and true errors as a function of k.

cross_validation(m)

Finds the best k value using cross-validation.

Additional Methods
if_x_in_intervals(intervals, x)

Checks if a given value x falls within the provided intervals.

zero_one_lost(intervals, x, y)

Calculates the zero-one loss for a given data point (x, y) based on the provided intervals.

cal_emp_error(hypothesis, data)

Calculates the empirical error for a given hypothesis and data.

calc_true_error_for_I(l)

Calculates the true error for a given set of intervals.

cal_overlap_with_interval_array_of_interval(intervals1, array_of_interval)

Calculates the overlap of a set of intervals with an array of intervals.

interval_overlap(intervals1, intervals2)

Calculates the overlap between two sets of intervals.

cal_y_probability_per_x(x)

Calculates the probability distribution of y given x.

Usage

To use the HypothesisEvaluator class and its methods, instantiate the class and call the desired methods. Ensure that the required dependencies are installed.
```
evaluator = HypothesisEvaluator()
evaluator.sample_from_D(10)
evaluator.experiment_m_range_erm(10, 100, 5, 3, 100)
evaluator.experiment_k_range_erm(1500, 1, 10, 1)
print(evaluator.cross_validation(1500))
