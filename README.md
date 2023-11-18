<head>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>

# Ontologizer
A method of inductively inferring abstractions from unlabeled data with the goal of interpreting internal representations of deep learning models. The goal is to build an "ontological" representation of input data.

We define an _ontology_ as a function that encodes a random vector $\mathbf{x}$ as a composition of typologies applied to $\mathbf{x}$.
We define a _typology_ as an injective mapping of a random vector to a set of _centroids_.

These definitions are based on the assumptions that
1. Clustering, denoising, and abstraction are isomorphic.
2. The set of all abstractions accessible to a model defines its ontology.
3. In deep learning, abstractions emerge _de novo_ as a model learns to generalize.
4. Abstraction is a form of lossy compression.
5. Abstraction quality can be assessed by the quality of data that can be decoded from it.

Let $\Omega$ be an ontology of data $X$ consisting of $n$ typologies.
Let $\Theta$ be a typology in $\Omega$ and $\mathbf{\hat{x}}$ be an encoding of a vector $\mathbf{x} \in X$.
Let $\Theta_0(\mathbf{x}) = \mathbf{0}$.

$$
\Omega(\mathbf{x}) = \bigoplus_{i=1}^{n}\Theta_i(\mathbf{x} - \Theta_{i-1}(\mathbf{x}))
$$

We propose that by minimizing the self-supervised loss function

$$
\mathcal{L}(\Omega) = \mathbb{E}||\mathrm{sum}[\Omega(\mathbf{x})] - \mathbf{x}||^2
$$

we can obtain a maximally informative ontology.

The ontology is generated by stacking [DeePWAK](https://github.com/kewiechecki/DeePWAK) layers.
Each DeePWAK layer generates a typology.



# Overview
![overview](https://github.com/kewiechecki/ontologizer/blob/master/flowchart.png?raw=true)
