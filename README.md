
<!-- PROJECT LOGO -->
<br />
<p align="center">
 
   

  <h3 align="center">A Dynamical System Based Framework for Learning a Low-dimensional Representation of Data</h3>

  <p align="center">
    Document is available below
    <br />
    <a href="xxx.pdf"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    
  </p>
</p>




<!-- ABOUT THE PROJECT -->
## About The Project

This paper proposes a framework for learning a low-dimensional representation of data based on the nonlinear dynamical system. This method is called dynamical dimension reduction (DDR). In this model, we find the dynamical system which evolves a data in high-dimensional space toward the low-dimensional subspace. And the lower representation, called the latent vector, is defined by projecting the solution to the ODE at the fixed final time onto the onto the orthonomal subspace. 



* Based on the Equation discovery method, we parametrize the underlying vector filed by the linear combination of pre-specified dictionary.
* We solve the ODE-constraint optimization problem by minimizing both the mean-squared residual error and the regularity of the flow.
* We measure the regularity of the flow by total mean kinetic energy of the trajectories. With the idea of the optimal transportation, it can be viewed as 2-Wesserstain distance between the initial data measure and terminal data measure.
* Using the adjoint method, compute the gradients of objective by solving alternative adjoint equation. 
* We prove the existence of optimizer and stability of the forward embedding to the noise under a given data.



## Requirements
* JAX library is required to run the code. 

 ```
 pip install jaxlib
 ```
 
<!-- Usage Example -->
## Usage Example
Explore the example which shows how this code run for the dimension reduction. Using the algorithm, we train both "coefficient matrix in the dictionary representation of vector field" and "orthonomal projection matrix" 

### 1. Dataset

Here, we use the synthetic data generating S-shaped manifold by the known ODE. The code for creating synthetic data is here [generating](https://github.com/rkyoon12/NAED/blob/master/GenerateData/osc.pickle)

Also, we examine the DDR method on the real example data, Iris and Hand-written digits. You can download pickled data. 

If you have own datasets, input(X) should be written as the tensor of the shape ( n_batch, original_dimension ). 

### 2. Dictionary choice. 

In DDR method, the right-hand side of dynamical system is written by dictionay representation. A natural choice of candidate functions is polynomials up to some degree.  
### 

This shows how to customize the dictionary in the main code. 

In the code [Main](https://github.com/rkyoon12/DDR/blob/main/Main.py), you can change the dictionay elements and its gradient. 

  ```sh
  # N  : a number of data example
  # d  : a original dimension of data
  # k  : a reduced dimension of data
  # dn : a number of candidate functions in dictionary
  
  def bigxi_true(x):
    # dictionary : [1, h, h^2 ,h^3]
    # rhs : N x dn
    rhs = jnp.concatenate([jnp.ones((x.shape[0],1)),x,jnp.power(x,2),jnp.power(x,3)],1)
    
  
  def grad_ft_true(sol_at_t, dn):
     # gradient : N x dn x d

      (N,d) = sol_at_t.shape
      gradient = np.zeros((N, dn, d))

      #linear dictionary
      gradient[:,1:1+d, 0:d] = jnp.eye(d)

      # nonlinear dictionary
      for i in range(d):
        gradient[:,d+1+i ,i ] = jnp.multiply(jnp.power(sol_at_t[:,i],1),2)
        gradient[:,2*d+1+i ,i ] = jnp.multiply(jnp.power(sol_at_t[:,i],2),3)
     

      return jnp.array(gradient)


  ```


### 3. Optimization

In this model, we have two parameters, beta and Q. We recommend to initialize beta to satisfies the Assumption 3.2 in paper, which guarantee the existence of the solution to the dynamical system.


Then it is trained using the gradient-based method "ADAM" with the constant learning rate. Depending on your problem, change learning rate and optimizer. 




<!-- Results -->
## Results  

We depict the learned lower-dimensional representation of data. 
![Digits](https://github.com/rkyoon12/DDR/blob/main/MNIST4.png)

This is comparision of lower-dimensional representation of the DDR to the PCA, t-SNE, and Umap. We show that the nonlinearity in the vector field encourage the mapping to preserve more features in the data. We also know that our method maintain both local and global structure of data. 


For more examples, please refer to the [paper](https://xxx.pdf).





<!-- CONTACT -->
## Contact
Ryeongkyung Yoon - rkyoon@math.utah.edu

Project Link: [https://github.com/rkyoon12/DDR](https://github.com/rkyoon12/DDR)


<!--stackedit_data:
eyJoaXN0b3J5IjpbNzUwNDgyOTMwLC0yMDYxNDMyNzgsLTEzMT
I5MTU0MzAsLTEzMTI5MTU0MzAsLTk4MTkwMTEzN119
-->
