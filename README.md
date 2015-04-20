##Distrubuted Stochastic Gradient Descent for Matrix Factorization in Spark

This algorithm is based on "Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent" (Gemulla et al.). The basic concept is to select several "interchangable" blocks for each strata in each iteration and run SGD sepearatly in each block independently.As long as those blocks are "interchangable", the sum of the results would be correct since they don't use same row or same column among blocks. 

To execute the program
	>> ./bin/spark-submit numOfFactors numOfWorkers numOfIterations beta lambda training_data outputRowModel outputColumnModel
	>> ./bin/spark-submit 20 3 100 0.6 0.1 nf_subsampling.csv w.csv h.csv

All of the code had been successfully ran and tested on Spark 1.3.0.



Big Matrix Factorization  Spark implement Large-Scale Matrix Factorization with Distributed Stochastic
Gradient Descent (DSGD-MF) in Spark. The paper sets forth a solution for matrix factorization
using minimization of sum of local losses. The solution involves dividing the matrix into strata
for each iteration and performing sequential stochastic gradient descent within each stratum in
parallel. The two losses considered are the plain non-zero square loss and the non-zero square loss
with L2 regularization of parameters W and H:-

