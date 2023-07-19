## GMM pointset registaration with encoding class scores
This work focuses on reformulating point set registration using Gaussian Mixture Models while considering attributes associated with each point. Our approach introduces class score vectors as additional
features to the spatial data information. By incorporating these attributes, we enhance the optimization process by penalizing incorrect matching terms.

### Registration with GMMs

 [Jian&Vemuri (2011)](https://www.researchgate.net/publication/224207506_Robust_Point_Set_Registration_Using_Gaussian_Mixture_Models) have proposed to estimate deformation $T$ between the **model** point sets $\mathcal{V}_1$  and the **scene** $\mathcal{V}_2$ by minimising the Euclidian distance ($L_2$ distance) between two Gaussian Mixtures Models (GMMs) fitted on each point set. Here rigid transformation (rotation, translation) is considered in $\mathbb{R}^2$, in which case the estimation of $T$ is performed as:

```math
\hat{T}=\arg\max_{T} \sum_{i=1}^{|\mathcal{V}_1|}\sum_{s=1}^{|\mathcal{V}_2|} \mathcal{N}(0;T(v_1^{(i)}) -v_2^{(s)}, \Sigma)
```
where $T$ is a transform function of parameter $\theta$ = $[t_1,t_2,\phi]$ representing the translation and rotation, $\mathcal{N}(x;\mu,\Sigma)$ indicates the normal distribution for random vector $x$ with mean $\mu$ and covariance $\Sigma$.\
 For simplicity,  we have chosen isotropic covariance $\Sigma=\sigma^2 \mathrm{I}_2$ in this work ($\mathrm{I}_2$ identity matrix in $\mathbb{R}^2$). \
 $v_1^{(i)}$ is the spatial coordinate in $\mathbb{R}^2$ used as attribute for the node $i$ in $\mathcal{V}_1$ (resp. $v_2^{(s)}$ is the spatial coordinate in $\mathbb{R}^2$ used as attribute for the node $s$ in $\mathcal{V}_2$).

for 2D case we have :
```math
\hat{T}=\arg\max_{T} \sum_{i=1}^{|\mathcal{V}_1|}\sum_{s=1}^{|\mathcal{V}_2|} 
\exp\left(\frac{-\left\|T\left(v_1^{(i)} \right)  -v_2^{(s)}  \right\|^2}{4 \sigma^2}\right)
```

### Registration with class attributes


We propose to extend the GMMreg by concatenating a class  vector (noted $c$)  to spatial coordinate ($v$) as part of the attribute describing the nodes such that the  estimation becomes:

$$
\hat{T}=\arg\max_{T} \sum_{i=1}^{|\mathcal{V}_1|}\sum_{s=1}^{|\mathcal{V}_2|} 
\exp\left(\frac{-\left\|T\left(v_1^{(i)} \right)  -v_2^{(s)}  \right\|^2}{4 \sigma^2}\right)\times \exp\left(\frac{-\|c^{(i)}_1 -c^{(s)}_2\|^2}{4 \sigma_c^2}\right)$$

### Implementation
The main code is  `gmmreg_Extenstion.py` utlizing following functions:
#### **Pre-processing**
We first normalize the datapoints with z score method. Then augment the class score vector to each point, for example : 
$$
\begin{bmatrix}

 X & Y & class0 & class1 & class3\\
 x1 & y1 &1 & 0 &0 \\
 x2 & y2 & 0 & 1 & 0\\
 x3 & y3 & 0 & 0 & 1\\

\end{bmatrix}$$
 
we assume that each point represents one class.For this example we have three datapoints associated with three classes.

#### **transforms**
The affine transformation between shapes is defined by three basic transformations: rotation, translation and scaling. In the case of 2D shapes for instance, the latent variable
to estimate can be defined by the following parameters :
$$\theta = [t_1,t_2,\phi]$$ 
Where $\phi$ is rotation parameter,and $t_1$ and $t_2$ are rotation parameters.

$$\mu_i(\theta) =\begin{pmatrix}
  cos(\phi) & -sin(\phi) \\
  sin(\phi) &  cos(\phi)
\end{pmatrix}\mu_i^0 + \begin{pmatrix}
  t_1  \\
  t_2
\end{pmatrix}   $$

#### **L2_objective**
To compute the L2 distance between the two Gaussian mixture  densities constructed from a '**model**' point set and a '**scene**' point set at a given 'scale'(sigma), we need to the inner product between two spherical Gaussian mixtures, computed using the Gauss Transform.The centers of the two mixtures are given in terms of two point sets A and B (of same dimension d)represented by an $m$ x $d$ matrix and an   $n$ x $d$ matrix, respectively.
It is assumed that all the components have the same covariance matrix represented by a scale parameter (sigma). The inner products are implemented in `gauss_transform` function.

To optimize the $L_2$ distance computing from `gauss_transform` function, simulating annealing with temperature parameter $\sigma$ (or scale) is used due to dact that for large $\sigma$s the $L_2$ distance tends to be non-convex. 



