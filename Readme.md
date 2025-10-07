# andycshang-mlops-f25
This is my first assignment of 596C. Before building the model, my goal is to first study the tutorials, then complete 
an initial setup of the model to obtain preliminary prediction results, and afterward proceed with parameters testing 
and more complex data processing.

### The process of building my model is as following:

First, by studying the tutorial I learned the model-building workflow, the purpose and underlying principles of each 
step, and produced an initial model. At this stage I set the number of iterations and learning rate to iterations = 1000
and alpha = 1.0e-6, and initialized the weight vector w and bias term b to zero. The minimum cost I obtained was 39.35, 
and the MSE was relatively high at 66.89.

By adjusting different iteration counts and learning rates, it can be observed that within a certain range, a higher 
number of iterations results in a lower cost. When the number of iterations is set to 10,000 and the learning rate 
to 1e-6, the cost reaches 32.94 and the training MSE is 32.48. However, the learning rate needs to be kept 
within an appropriate range — if it is too small, the convergence will be slow and accuracy may be affected with fewer 
iterations, while if it is too large, the model may diverge.


Next, I started to consider which factors influence the cost — in other words, how to improve model performance. In the 
initial model I had not normalized or standardized the dataset.I noticed that the feature values in the dataset varied 
greatly, such as between 'B' and 'AGE'. Therefore, following the approach in the tutorial, I standardized the 
features and obtained the following results:

*Iteration  900: Cost   302.61*  
*Training result: w = [-0.00353237  0.00287454 -0.00437161  0.00177588 -0.00386185  0.00661378
 -0.00315909  0.00217926 -0.00359845 -0.00427178 -0.00456878  0.00315981
 -0.00684981], b = 0.022785151571493724*  
*Training MSE = 302.53316961619663*

I observed that using the standardized data led to a significant increase in both the cost and the training MSE. I 
suspect that the reason might be that I only standardized the 'X' data and did not standardize 'y', while the MSE 
is calculated based on the comparison between the two. This likely caused the noticeable discrepancy in the results.
Therefore, I attempted to standardize the y data as well, using the following code:

*y_scaler = StandardScaler()*  
*y_train_norm = y_scaler.fit_transform(y_train.values.reshape(-1, 1))*  
*y_test_norm = y_scaler.transform(y_test.values.reshape(-1, 1))*

However, after standardizing both X and y, a dimension mismatch occurred between the data matrix and the weight
vector w. This issue is relatively complex for me at the moment, and I have not yet fully resolved it. I am 
currently continuing to revise and debug the model. And this is the most difficult part so far.







