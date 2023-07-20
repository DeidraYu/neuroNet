# NeuroNet
This page describes the link between the software implementation and the mathematics behind a simple neural network, together with a derivation. It is assumed that the reader already knows what weights and biases are. For the ease of presentation and the connection between the presentation and the implementation we don't model the input layer as a separate layer.

## Feed forward

Let $\ell$ be the index of the layer in the network, $\ell \in \{0, 1, \cdots, L-1\}$. Each layer $\ell \in \{0, 1, \cdots, L-1\}$ has as input the ${\bf x}_\ell$ that is produced by the previous layer (or the input vector ${\bf x}_0 for the first layer). The layer produces the output, also called $activation$, ${\bf x}_{\ell+1}$ which is the input of the next layer, or is the output of the network in case for the last layer (possibly after some processing). For its operation, each layer $\ell$ a weight matrix $W_\ell$ and a bias vector ${\bf b}_\ell$.

Layer $\ell$ performs the following two operations:

- ${\bf z}_\ell = W_\ell x_\ell + {\bf b}_\ell$
- ${\bf x}_{\ell + 1} = \sigma({\bf z}_\ell)$

The intermidiate vecto ${\bf z}_\ell$ is not required explicitly for the feed forward computation. However, it will be used later in the section about back propagation.

As a result, the feed forward operation of the layers $\ell \in \{0, 2, \cdots, L-1\}$ is given by:

$${\bf x}_{\ell+1} = \sigma(W_\ell {\bf x}_\ell + {\bf b}_\ell)$$

In this document two types of presentations are mixed. We use the one above, with explicit reference to the layer, or the one that is introduced below, that removes the layer index. We want this to make some room for subscribts to indicate matrix or vector elements. This done by introducing ${\bf y}^{\ell} =  {\bf x}^{\ell+1}$. This is substituted for the left hand side of the above equation and see that all subscribts have become $\ell$. For a layer $\ell$ we now use the following equivalence.

$${\bf x}_{\ell+1} = \sigma(W_\ell {\bf x}_\ell + {\bf b}_\ell) \quad \Longleftrightarrow \quad {\bf y}= \sigma(W {\bf x} + {\bf b})$$

## Cost
In order to judge how good the output of the network is, we need a cost function. Here we base the cost directly on the error ${\bf x}_L - {\bf {h}}$ where ${\bf h}$ represents the desired network output for the current input vector. For a classification problem where classes are indicated by a numeric value, ${\bf h}$ is the one-hot encoding of this value. The cost function that will be used 0.5 times the squared $\ell_2$-norm:

$$
C({\bf x}_L) = \frac{1}{2} ||{\bf x}_L - {\bf {h}}||^2
$$

I believe that the motivation to use this norm is that it works out nicely in the derivation of the back propagation.

## Stogastic gradient descent
So, we have a network, via the feedforward computation we can derive the output ${\bf x}_L$, and we have the cost function $C({\bf x}_L) = \frac{1}{2} ||{\bf x}_L - {\bf {h}}||^2$ that quantifies the error that the networks makes. Now the job is to find good values values for the weights $W_\ell$ and the biasses ${\bf b}_\ell$. Good values are those that minimize the cost.

In order to find this local minimum gradient descent is used. That is, for each layer $\ell \in \{0, 1, \cdots, L-1\}$

$$
\begin{split}
W_\ell \quad &\leftarrow \quad W_\ell  -  \eta \frac{\partial C}{\partial W_\ell}\\\\
{\bf b}_\ell \quad &\leftarrow \quad {\bf b}_\ell  -  \eta \frac{\partial C}{\partial {\bf b}_\ell}\\\\

\end{split}
$$

where $\eta$ is the learning rate; a value of choice that tells how fast we want to approach the local minimum. A too large value may cause this method to overshoot, so be careful.

What remains to be done is to compute the above partial derivatives. This is the topic of back propagation in the next section.

## Back propagation

Back propagation relies on the chain rule for derivation.

$$
\begin{cases}
\dfrac{\partial C}{\partial W_\ell} & = \dfrac{\partial C}{\partial {\bf x}_{\ell + 1}^T} \dfrac{\partial {\bf x}_{\ell + 1}}{\partial W_\ell}\\\\

\dfrac{\partial C}{\partial {\bf b}_\ell^T} & = \dfrac{\partial C}{\partial {\bf x}_{\ell + 1}^T} \dfrac{\partial {\bf x}_{\ell + 1}}{\partial {\bf b}_\ell^T}\\\\

\dfrac{\partial C}{\partial {\bf x}_\ell^T} & = \dfrac{\partial C}{\partial {\bf x}_{\ell + 1}^T} \dfrac{\partial {\bf x}_{\ell + 1}}{\partial {\bf x}_\ell^T}\\
\end{cases}
$$

Now introduct the vector ${\bf u}_\ell^T = \dfrac{\partial C}{\partial {\bf x}_\ell^T}$. We then rewrite the above system into

$$
\begin{cases}
\dfrac{\partial C}{\partial W_\ell} & = {\bf u}_{\ell+1}^T \dfrac{\partial {\bf x}_{\ell + 1}}{\partial W_\ell}\\\\

\dfrac{\partial C}{\partial {\bf b}_\ell^T} & = {\bf u}_{\ell+1}^T \dfrac{\partial {\bf x}_{\ell + 1}}{\partial {\bf b}_\ell^T}\\\\

{\bf u}_\ell^T & = {\bf u}_{\ell+1}^T \dfrac{\partial {\bf x}_{\ell + 1}}{\partial {\bf x}_\ell^T}\\
\end{cases}
$$


The third equation actually computes a component that propagates back from layer $\ell$ to layer $\ell-1$. If we start at the last layer, that is layer $L-1$ we see that the first term at the left hand side is

$$
{\bf u}_L^T = \dfrac{\partial C({\bf x}_L)}{\partial {\bf x}_L^T} = \dfrac{\partial \frac{1}{2} \|{\bf x_L} - {\bf h}\|^2}{\partial {\bf x}_L^T} = ({\bf x_L} - {\bf h})^T\\
$$

That was easy. Now, if we are able to compute the second term at the right hand side for all $\ell$ we can use the above result to compute ${\bf u}_{L-1}$ and use that to compute ${\bf u}_{L-2}$ etc.. Hence, the work to be done is to find the partial derivatives of the second term at the right hand side of the equations above.

Because below we need to refer to the individual elements of the matrix $W_\ell$ it is practical to get rid of the subscript $\ell$. Like before let ${\bf y}_{\ell} = {\bf x}_{\ell+1}$. Similarly let ${\bf v}_{\ell} = {\bf u}_{\ell+1}$. Now we drop $\ell$ in the notation. We then get


$$
\begin{cases}
\dfrac{\partial C}{\partial W} & = {\bf v}^T \dfrac{\partial {\bf y}}{\partial W}\\\\

\dfrac{\partial C}{\partial {\bf b}^T} & = {\bf v}^T \dfrac{\partial {\bf y}}{\partial {\bf b}^T}\\\\

{\bf u}^T & = {\bf v}^T \dfrac{\partial {\bf y}}{\partial {\bf x}^T}\\
\end{cases}
$$

In the following three subsection we will compute eacht of them.





## Computing the cost gradient with respect to the weights: $\dfrac{\partial C}{\partial W}$

We want to compute

$$\dfrac{\partial C}{\partial W} = {\bf v}^T \dfrac{\partial {\bf y}}{\partial W} = \dfrac{\partial {\bf v}^T {\bf y}}{\partial W}$$

To simplify the derivation we moved ${\bf v}^T$ behind differantial operator such that the numerator becomes a scalar. We then can compute gradient that has the same shape as $W$. To compute the element $(i,j)$ of the gradient we get


$$
\begin{split}
\dfrac{\partial C}{\partial W_{ij}} &= \dfrac{\partial {\bf v}^T {\bf y}}{\partial W_{ij}}\\\\
&= \dfrac{\partial \sum_k v_k y_k}{\partial W_{ij}}\\\\
&= \dfrac{\partial \sum_k v_k \sigma({\bf w}_k{\bf x}+b_k)}{\partial W_{ij}}\\\\
&= \dfrac{\partial v_i \sigma({\bf w}_i{\bf x}+b_i)}{\partial W_{ij}}\\\\
&= \dfrac{\partial v_i \sigma(z_i)}{\partial z_i} \dfrac{\partial {\bf w}_i{\bf x}+b_i}{\partial W_{ij}}\\\\
&= v_i \sigma^\prime(z_i) x_j\\\\
\end{split}
$$

The full gradient matrix now is

$$
\begin{split}
\dfrac{\partial C}{\partial W} &=
\begin{bmatrix}
v_0 \sigma^\prime(z_0) x_0 & v_0 \sigma^\prime(z_0) x_1 & \cdots & v_0 \sigma^\prime(z_0) x_{M-1} \\
v_1 \sigma^\prime(z_1) x_0 & v_0 \sigma^\prime(z_0) x_1 & \cdots & v_0 \sigma^\prime(z_0) x_{M-1} \\
\vdots & \vdots & \ddots & \vdots \\
v_{N-1} \sigma^\prime(z_{N-1}) x_0 & v_{N-1} \sigma^\prime(z_{N-1}) x_1 & \cdots & v_{N-1} \sigma^\prime(z_{N-1}) x_{M-1} \\
\end{bmatrix}\\\\
&= \left({\bf v} \odot \sigma^\prime({\bf z})\right) {\bf x}^T
\end{split}
$$

## Computing the cost gradient with respect to the activation values: $\dfrac{\partial C}{\partial {\bf b}^T}$

We want to compute
$$
\begin{split}
\dfrac{\partial C}{\partial {\bf b}^T} &= {\bf v}^T \dfrac{\partial {\bf y}}{\partial {\bf b}^T} \\\\
&= {\bf v}^T \dfrac{\partial \sigma({\bf z})}{\partial {\bf z}^T}  \dfrac{\partial (W{\bf x} + {\bf b})}{\partial {\bf b}^T}\\\\
&= {\bf v}^T \text{diag}(\sigma^\prime({\bf z})) I \\\\
&= ({\bf v} \odot \sigma^\prime({\bf z}))^T
\end{split}
$$

Or alternatively we can write $\frac{\partial C}{\partial {\bf b}^T}$ as a column vector:

$$
\dfrac{\partial C}{\partial {\bf b}} = {\bf v} \odot \sigma^\prime({\bf z})
$$


## Computing the cost gradient with respect to the activation values: ${\bf u}^T = \dfrac{\partial C}{\partial {\bf x}^T}$

We want to compute

$$
\begin{split}
{\bf u}^T &= {\bf v}^T \dfrac{\partial {\bf y}}{\partial {\bf x}^T}\\\\
&= {\bf v}^T \dfrac{\partial \sigma({\bf z})}{\partial {\bf z}^T}  \dfrac{\partial (W{\bf x} + {\bf b})}{\partial {\bf x}^T}\\\\
&= {\bf v}^T \text{diag}(\sigma^\prime({\bf z})) W\\\\
&= ({\bf v} \odot \sigma^\prime({\bf z}))^T W
\end{split}
$$

Or alternatively we can write ${\bf u}$ as a column vector:

$${\bf u} = W^T({\bf v} \odot \sigma^\prime({\bf z}))$$

## The resulting equations

$$
\begin{cases}
    \dfrac{\partial C}{\partial W} &= \left({\bf v} \odot \sigma^\prime({\bf z})\right) {\bf x}^T\\\\
    \dfrac{\partial C}{\partial {\bf b}} &= {\bf v} \odot \sigma^\prime({\bf z})\\\\
    {\bf u} &= W^T({\bf v} \odot \sigma^\prime({\bf z}))
\end{cases}
\quad \Rightarrow \quad 
\begin{cases}
    \dfrac{\partial C}{\partial W_\ell} &= \left({\bf u}_{\ell+1} \odot \sigma^\prime({\bf z}_\ell)\right) {\bf x}^T_\ell\\\\
    \dfrac{\partial C}{\partial {\bf b}_\ell} &= {\bf u}_{\ell+1} \odot \sigma^\prime({\bf z}_\ell)\\\\
    {\bf u}_\ell &= W^T({\bf u}_{\ell+1} \odot \sigma^\prime({\bf z}_\ell))
\end{cases}
$$

Which can be written into

$$
\begin{cases}
    \pmb\gamma = \dfrac{\partial C}{\partial {\bf b}}&= {\bf v} \odot \sigma^\prime({\bf z})\\\\
    \dfrac{\partial C}{\partial W} &= \pmb\gamma {\bf x}^T\\\\
    {\bf u} &= W^T\pmb\gamma
\end{cases}
\quad \Rightarrow \quad 
\begin{cases}
    \pmb\gamma_\ell = \dfrac{\partial C}{\partial {\bf b}_\ell}&= {\bf u}_{\ell+1} \odot \sigma^\prime({\bf z}_\ell)\\\\
    \dfrac{\partial C}{\partial W_\ell} &= \pmb\gamma_\ell {\bf x}^T_\ell\\\\
    {\bf u}_\ell &= W^T\pmb\gamma_\ell
\end{cases}
$$


## How does this relate to the Python code

Call ${\pmb\delta}_\ell = {\bf u}_{\ell+1} \odot \sigma^\prime({\bf z}_\ell)$. The system of equations then becomes

$$
\begin{cases}
{\pmb\delta}_\ell &= {\bf u}_{\ell+1} \odot \sigma^\prime({\bf z}_\ell) \\\\
\dfrac{\partial C}{\partial {\bf b}_\ell} &= {\pmb\delta}_\ell\\\\
\dfrac{\partial C}{\partial W_\ell} &= {\pmb\delta}_\ell {\bf x}^T_\ell\\\\
{\bf u}_\ell &= W^T {\pmb \delta}_\ell
\end{cases}
$$

Further note that

$$
\begin{split}
{\pmb\delta}_\ell &= {\bf u}_{\ell+1} \odot \sigma^\prime({\bf z}_\ell) \\\\
{\pmb\delta}_{\ell-1} &= {\bf u}_{\ell} \odot \sigma^\prime({\bf z}_{\ell-1}) \\\\
\end{split}
$$

And by using ${\bf u}_\ell = W^T {\pmb \delta}_\ell$ we get

$$
{\pmb\delta}_{\ell-1} = (W^T {\pmb \delta}_\ell) \odot \sigma^\prime({\bf z}_{\ell-1})
$$

The resulting system of equations of layer $\ell \in \{0, 1, \cdots, L-1\}$ then is similar to that is used in the python code

$$
\begin{cases}
{\pmb\delta}_{\ell-1} &=

\begin{cases}
{\bf u}_{L} \odot \sigma^\prime({\bf z}_{L-1}) & ,\text{when} \quad \ell = L-1\\\\
(W^T {\pmb \delta}_\ell) \odot \sigma^\prime({\bf z}_{\ell-1}) & ,\text{otherwise}

\end{cases}\\\\

\dfrac{\partial C}{\partial {\bf b}_\ell} &= {\pmb\delta}_\ell\\\\
\dfrac{\partial C}{\partial W_\ell} &= {\pmb\delta}_\ell {\bf x}^T_\ell
\end{cases}
$$

What I dislike from the Python version of the equations is that the last layer has a slightly different functionaly from the lower layers. Note that the definition of $\pmb\gamma$ is the same as for $\pmb\delta$.





