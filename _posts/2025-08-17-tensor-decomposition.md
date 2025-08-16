---
title: 'Tensor Decomposition'
date: 2199-01-01
permalink: /posts/2025/08/tensor-decomposition/
tags:
  - machine learning 
  - compression model
  - cool posts
  - category1
  - category2
---


# Motivation 

## Goal of Tensor Decomposition

Many layers in deep learning — especially **fully-connected (dense) layers** and **embedding layers** — contain large weight matrices.  
These matrices consume a lot of memory and computation.

The goal of **tensor decomposition** is to:

> *Approximate these large matrices/tensors with simpler (low-rank) structures that maintain comparable accuracy.*

---

## Core Idea: Low-Rank Approximation

Suppose you have a weight matrix:

$$
X \in \mathbb{R}^{n \times m}
$$

You can approximate it by the product of two smaller matrices:

$$
X \approx A B, \quad A \in \mathbb{R}^{n \times r}, \quad B \in \mathbb{R}^{r \times m}, \quad r \ll \min(m, n)
$$

 Instead of storing the full $m \times n$ parameters, you only need to store:

$$
r(m + n)
$$

This significantly reduces both **memory usage** and **computational cost**.
![image](https://hackmd.io/_uploads/BkRhrMGuxl.png)

---

# Understanding Matrix Rank

## Definition of Rank

The **rank** of a matrix `A`, denoted as `rank(A)`, is the **maximum number of linearly independent rows or columns** in `A`. This value reflects the dimension of the **row space** and the **column space** of the matrix, which are always equal.

## Why `rank(A) ≤ min(m, n)`

Let `A` be an `m × n` matrix. The rank of `A` must be less than or equal to the smaller of the number of rows (`m`) or columns (`n`), i.e.,
```
rank(A) ≤ min(m, n)
```

### Reasoning

We can transform `A` into its row-reduced form (RREF), using a series of elementary row operations:

```
A → Eₙ·Eₙ₋₁·...·E₁·D
```

Here, `D` is a matrix in row-reduced echelon form (RREF), typically consisting of leading 1s and zeros elsewhere.

Key observations:

- **Row operations** do **not** change the linear independence of rows.
- Likewise, **column dependencies** are preserved under these operations.

### Column Dependency Preservation

Suppose column `l` of `A` can be expressed as a linear combination of `k` other columns (e.g., `x₁, ..., xₖ`):

```
a_il = ∑ cₕ·a_ih   and   a_jl = ∑ cₕ·a_jh
```

Then, performing an elementary row operation (e.g., adding row `j` to row `i`) keeps column `l` as a linear combination of the same columns. Therefore, column independence remains unchanged through RREF.

### Conclusion

- Computing by rows: `rank(A) ≤ m`
- Computing by columns: `rank(A) ≤ n`

Hence:

```
rank(A) ≤ min(m, n)
```

---

## Rank of a Product: `rank(AB) ≤ min(rank(A), rank(B))`

Let `A` be an `m × k` matrix and `B` a `k × n` matrix. Then the product `AB` is an `m × n` matrix.

A fundamental result in linear algebra — sometimes attributed to **Sylvester's Rank Inequality** — states that:

```
rank(AB) ≤ min(rank(A), rank(B))
```

---

### Intuitive Explanation

Let us analyze the product `AB` in terms of the structure of its columns.

Suppose we write matrix `B` as a collection of its column vectors:

```
B = [b₁ b₂ ... bₙ]
```

Each column `bⱼ` is a vector in ℝᵏ. Then the product `AB` can be expressed as:

```
AB = [A·b₁  A·b₂  ...  A·bₙ]
```

This means that **each column of `AB` is a linear combination of the columns of `A`**. The coefficients of the linear combination are given by the entries of `bⱼ`.

Therefore, all columns of `AB` lie within the **column space of `A`**, implying that:

```
ColSpace(AB) ⊆ ColSpace(A)
→ rank(AB) ≤ rank(A)
```

Similarly, if we analyze the rows:

- Each row of `AB` is a linear combination of the rows of `B`, with coefficients determined by rows of `A`.
- Thus:

```
RowSpace(AB) ⊆ RowSpace(B)
→ rank(AB) ≤ rank(B)
```

Combining both results:

```
rank(AB) ≤ min(rank(A), rank(B))
```

---

# Approximate in Tensor Decomposition 

## Why Not Every Matrix Can Be Exactly Factorized

Although matrix factorization and tensor decomposition techniques are widely used, it is important to recognize that **not every matrix `W` can be exactly written as a product of low-rank components**.

This limitation stems directly from the **rank constraint**. For example, suppose we attempt to write a matrix `W` as the product:

```
W ≈ U · Vᵀ
```

where `U ∈ ℝ^{m×r}` and `V ∈ ℝ^{n×r}`, with `r << min(m, n)`.

In general, **such an exact decomposition is only possible if `rank(W) ≤ r`**. However, for high-rank matrices, this is not true, so **we must settle for an approximation**:

```
W ≈ U · Vᵀ    (approximation, not equality)
```

This leads us to the concept of **low-rank approximation**, where the goal is to **find the best approximation of `W` using a structure-constrained format** (e.g., fixed low rank `r`).

---

# Common Tensor Decomposition Techniques for Approximation

In practical applications — especially in deep learning — full-rank matrices or tensors are often approximated by **structured, lower-rank decompositions** to reduce memory and computation cost. Some commonly used techniques include:

---

# 1. Low-Rank Matrix Factorization (SVD)

**Definition**:  
For any real matrix $A \in \mathbb{R}^{m \times n}$, there exists a decomposition:

$$
A = U \Sigma V^\top
$$

Where:
- $U \in \mathbb{R}^{m \times m}$: orthogonal matrix (columns are orthonormal)
- $V \in \mathbb{R}^{n \times n}$: orthogonal matrix
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal (non-negative singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$)

---

### Proof of Existence of SVD

**Main idea**:
1. Consider $A^\top A$ ($n \times n$), which is **symmetric positive semi-definite (PSD)**.
2. By the **Spectral Theorem**, there exists an orthonormal basis of eigenvectors $v_1, \dots, v_n$ with eigenvalues $\lambda_i \ge 0$.
3. Sort $\lambda_i$ in descending order, define $\sigma_i = \sqrt{\lambda_i}$.
4. Form $V = [v_1, \dots, v_n]$, and build $\Sigma$ with $\sigma_i$ on the diagonal.
5. Construct $U$ by normalizing $Av_i / \sigma_i$.

---

**Detailed steps**:

1. $A^\top A$ is symmetric ⇒ all eigenvalues $\lambda_i$ are **real**.
2. Since $A^\top A$ is PSD ⇒ $\lambda_i \ge 0$.
3. By the Fundamental Theorem of Algebra (FTA) ⇒ the characteristic equation $\det(A^\top A - \lambda I) = 0$ has $n$ roots include real and complex.
4. By the Spectral Theorem ⇒ there exists an orthogonal $V$ such that:

$$
A^\top A = V \Lambda V^\top
$$

Assume $A \in \mathbb{R}^{m\times n}$. Let
$$
A^\top A = V \Lambda V^\top
$$
be the spectral decomposition of the symmetric positive semi-definite matrix $A^\top A$. Here
- $V = [v_1,\dots,v_n]$ is orthogonal ($V^\top V = I$),
- $\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_n)\) with \(\lambda_i \ge 0$.

---

1. $A v_i$ is an eigenvector of $A A^\top$

Starting from the eigen-equation for $A^\top A$:
$$
A^\top A v_i = \lambda_i v_i.
$$
Multiply both sides on the left by $A$:
$$
A(A^\top A v_i) = A(\lambda_i v_i)
\quad\Longrightarrow\quad
(AA^\top)(A v_i) = \lambda_i (A v_i).
$$
Thus $A v_i$ is an eigenvector of $AA^\top$ corresponding to eigenvalue $\lambda_i$, unless $A v_i = 0$. In the zero case the vector is the zero vector (associated with $\lambda_i=0$).

---

2. Norm relation: $\|A v_i\| = \sqrt{\lambda_i}$ (when $v_i$ is unit)

Assume $\|v_i\| = 1$ (we may choose eigenvectors of $A^\top A$ to be unit length). Then
$$
\|A v_i\|^2 = (A v_i)^\top (A v_i) = v_i^\top (A^\top A) v_i = v_i^\top (\lambda_i v_i) = \lambda_i v_i^\top v_i = \lambda_i.
$$
Hence
$$
\|A v_i\| = \sqrt{\lambda_i} \equiv \sigma_i.
$$
We define the singular value $\sigma_i := \sqrt{\lambda_i} \ge 0$.

---

3. Construct left singular vectors $u_i$ and the relation $A v_i = \sigma_i u_i$

For each index $i$ with $\sigma_i > 0$, define
$$
u_i := \frac{A v_i}{\sigma_i}.
$$
By the previous step $u_i$ has unit norm:
$$
\|u_i\| = \frac{\|A v_i\|}{\sigma_i} = \frac{\sigma_i}{\sigma_i} = 1,
$$
and different $u_i$ are orthogonal (because the corresponding $v_i$ are orthogonal and $AA^\top$ is symmetric). Therefore the vectors $u_i$ form an orthonormal set (and can be completed to an orthonormal basis of $\mathbb{R}^m$). The key relation is
$$
A v_i = \sigma_i u_i,
$$
 $A v_i$ is **proportional** to $u_i$ with factor $\sigma_i$). **Note:** it is $\sigma_i=\sqrt{\lambda_i}$(not $\lambda_i$) that appears here.

---

4. Matrix form: $A V = U \Sigma$ and hence $A = U \Sigma V^\top$

Stack the column relations $A v_i = \sigma_i u_i$ for $i=1,\dots,n$. Let
- $V = [v_1,\dots,v_n]$ (orthogonal),
- $U = [u_1,\dots,u_m]$ (orthogonal; the first $r$ columns are from nonzero singular values, remaining columns complete the basis),
- $\Sigma$ be the $m\times n$ diagonal matrix with $\sigma_1,\dots,\sigma_r$ on the diagonal (zeros elsewhere).

Then column-stacking gives
$$
A V = [A v_1,\dots,A v_n] = [\sigma_1 u_1,\dots,\sigma_n u_n] = U \Sigma.
$$
Multiplying on the right by $V^\top$ (and using $V V^\top = I$) yields
$$
A = U \Sigma V^\top.
$$

This is the Singular Value Decomposition of $A$.

---

##  Low-Rank Factorization and Truncated SVD

###  Low-Rank Factorization
We approximate $A$ with a low-rank matrix $A_k$:

$$
A \approx U_k \Sigma_k V_k^\top
$$

Where:
- $U_k$ contains the first $k$ columns of $U$
- $\Sigma_k$ is $k \times k$
- $V_k$ contains the first $k$ columns of $V$

---

### Choosing k (Truncated SVD)
Criteria:
- **Energy capture**: choose $k$ such that

$$
\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \ge \tau
$$
(where $\tau$ is typically 0.9 or 0.95)

- **Trade-off**: smaller $k$ ⇒ fewer parameters ⇒ higher approximation error.

---

### 3. Eckart–Young–Mirsky Theorem
**Theorem:**  
Let $A \in \mathbb{R}^{m \times n}$ have singular values $\sigma_1 \ge \dots \ge \sigma_r > 0$, $r = \mathrm{rank}(A)$.  
Among all matrices $B$ with $\mathrm{rank}(B) \le k < r$, the truncated SVD $A_k$ uniquely minimizes:

- **Frobenius norm error**:
$$
\min_{\mathrm{rank}(B) \le k} \|A - B\|_F = \left( \sum_{i=k+1}^r \sigma_i^2 \right)^{1/2}
$$
- **Spectral norm error**:
$$
\min_{\mathrm{rank}(B) \le k} \|A - B\|_2 = \sigma_{k+1}
$$

---

### Proof of Eckart–Young–Mirsky (Frobenius Norm)

**Write the SVD**
Let  
$$
A = U \Sigma V^\top,
$$
where  
$$
\Sigma = \mathrm{diag}(\sigma_1, \dots, \sigma_r, 0, \dots)
$$
with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.

---

**Use Frobenius Norm Invariance**
The Frobenius norm is invariant under orthogonal transformations:

$$
\|A - B\|_F = \| U^\top (A - B) V \|_F.
$$

Let  
$$
C := U^\top B V.
$$
Since orthogonal transformations preserve rank:  
$$
\mathrm{rank}(B) \le k \quad \Rightarrow \quad \mathrm{rank}(C) \le k.
$$

Therefore, the minimization problem:
$$
\min_{\mathrm{rank}(B) \le k} \|A - B\|_F
$$
is equivalent to:
$$
\min_{\mathrm{rank}(C) \le k} \|\Sigma - C\|_F.
$$

---

**Choose the Optimal $C$**
The matrix $\Sigma$ is diagonal with $\sigma_i$ on the diagonal.  
The best rank-$k$ approximation to a diagonal matrix (in Frobenius norm) is obtained by **keeping the top $k$ diagonal entries** and setting the rest to zero:

$$
C = \mathrm{diag}(\sigma_1, \dots, \sigma_k, 0, \dots).
$$

Any other choice would introduce larger squared error because Frobenius norm sums the squared differences element-wise.

---

**Transform Back**
Returning to the original coordinates:

$$
B = U C V^\top.
$$

With $C$ as above:
$$
B = U_k \Sigma_k V_k^\top = A_k.
$$

---

**Compute the Error**
Since  
$$
\Sigma - C = \mathrm{diag}(0, \dots, 0, \sigma_{k+1}, \dots, \sigma_r),
$$
we have:
$$
\|\Sigma - C\|_F^2 = \sum_{i = k+1}^r \sigma_i^2.
$$

Thus:
$$
\|A - A_k\|_F^2 = \sum_{i = k+1}^r \sigma_i^2.
$$

---


**Optimality**:
Any rank-$k$ matrix $B$ must live in a $k$-dimensional subspace.  
Projecting $A$ onto the subspace spanned by $\{u_1, \dots, u_k\}$ gives the minimal Frobenius error due to orthogonality (Pythagoras theorem). Thus $A_k$ is optimal.

---


## Randomized SVD for Large Matrices

### Core Idea
We want to approximate the range of $A$ using a low-dimensional subspace obtained from random projections.

Algorithm:
1. **Random Projection**  
   Generate a random Gaussian matrix  
   $$
   \Omega \in \mathbb{R}^{n \times (k+p)}
   $$
   where $k$ is target rank and $p$ is oversampling parameter ($p = 5$ or $10$).

2. **Sketch the Column Space**  
   Multiply:
   $$
   Y = A \Omega
   $$
   Now, $\mathrm{range}(Y) \approx \mathrm{range}(A)$.

3. **Orthonormalize**  
   Compute QR decomposition:
   $$
   Y = QR
   $$
   so $Q \in \mathbb{R}^{m \times (k+p)}$ has orthonormal columns.

4. **Projection**  
   Project $A$ to smaller space:
   $$
   B = Q^\top A
   $$
   Now $B \in \mathbb{R}^{(k+p) \times n}$.

5. **Small SVD**  
   Compute full SVD of $B$:
   $$
   B = \tilde{U} \Sigma V^\top
   $$

6. **Reconstruct Approximation**  
   $$
   U \approx Q \tilde{U}_{[:, 1:k]}, \quad
   \Sigma_k = \Sigma_{1:k, 1:k}, \quad
   V_k = V_{[:, 1:k]}
   $$
   Then:
   $$
   A \approx U_k \Sigma_k V_k^\top
   $$

---

### Complexity
- **Full SVD**: $O(mn\min(m,n))$
- **Randomized SVD**: $O(mn(k+p))$  
  Much smaller if $k \ll \min(m,n)$.

---

### Accuracy Enhancement — Power Iterations
For matrices with slowly decaying singular values, improve accuracy by:
$$
Y = (AA^\top)^q A \Omega
$$
where $q \in \{1,2,3\}$. This amplifies the decay of singular values.

---

## Modern Techniques

### 1. ARSVD (Adaptive Randomized SVD)
- **Goal**: Choose $k$ adaptively instead of fixing beforehand.
- Procedure:
    - Start with small $k$, increase gradually.
    - After each step, estimate energy ratio:
      $$
      \frac{\sum_{i=1}^k \sigma_i^2}{\|A\|_F^2}
      $$
      Stop when this exceeds threshold $\tau$ (e.g., 0.9 or 0.95).
    - Reduce power iterations if singular values decay quickly.

---

### 2. SVD-Free Optimization
- **Motivation**: In ML training, computing SVD in every step is expensive.
- **Idea**: Maintain orthonormality constraints without full SVD by:
    - **Orthogonal Regularization**: Add $\|W^\top W - I\|_F^2$ to loss.
    - **Iterative Orthonormalization**: Use QR or Cayley transform updates.
- Benefit: Avoids $O(mn^2)$ cost per iteration; suitable for very large models.

---
---
## 2. CP Decomposition 

![CP Decomposition](https://hackmd.io/_uploads/HJcQLg7dll.png)

**Definition**  
Approximates a tensor as a sum of outer products of vectors:
```
T ≈ ∑_{i=1}^r aᵢ ⊗ bᵢ ⊗ cᵢ
```
- `aᵢ ∈ ℝ^I`, `bᵢ ∈ ℝ^J`, `cᵢ ∈ ℝ^K`
- `⊗` denotes the **outer product**
- `r` is the CP rank (number of components)

![image](https://hackmd.io/_uploads/B1R0sxQOxl.png)


**Factor matrix form:**
```
T ≈ ⟦ A, B, C ⟧
A ∈ ℝ^(I×r), B ∈ ℝ^(J×r), C ∈ ℝ^(K×r)
```
Here, the *i-th* column of `A, B, C` corresponds to component `i`.

**Optimization goal:**
$$
\min_{A,B,C} \|X - ⟦A, B, C⟧\|_F^2
$$
That is, find factor matrices $A, B, C$ minimizing the Frobenius norm of the error.


**Common algorithm:**  
- **ALS (Alternating Least Squares)**:  
  Iteratively fix two factor matrices and solve for the third using least squares.

**Applications:**
- Compressing RGB images/video tensors
- Topic modeling from 3D term-document-context tensors
- Multi-way recommender systems
- Neural network weight compression

**Python Example (TensorLy):**
```python
import tensorly as tl
from tensorly.decomposition import parafac

# Create synthetic tensor
tensor = tl.tensor(tl.randn((4, 3, 2)))

# CP decomposition with rank=2
weights, factors = parafac(tensor, rank=2)

print("Factor matrices:")
for f in factors:
    print(f.shape)
```

---

## 3. Tucker Decomposition

![Tucker Decomposition](https://hackmd.io/_uploads/ByYA8xXuxg.png)

**Definition**  
Decomposes a tensor into:
1. A smaller **core tensor** `G`
2. Multiplied by factor matrices along each mode:
```
T ≈ G ×₁ A ×₂ B ×₃ C
```
- `×ₙ` is the mode-n product between a tensor and a matrix
- `A ∈ ℝ^(I×P)`, `B ∈ ℝ^(J×Q)`, `C ∈ ℝ^(K×R)` are factor matrices
- `G ∈ ℝ^(P×Q×R)` is the core tensor

**Difference from CP:**
- CP = sum of rank-1 tensors  
- Tucker = core tensor + projection matrices  
- Tucker allows different ranks along each mode

**Optimization goal:**
```
min_{G,A,B,C} || T - G ×₁ A ×₂ B ×₃ C ||_F²
```

**Applications:**
- Dimensionality reduction for multi-modal data
- Image compression (each mode = width, height, channel)
- Feature extraction from spatiotemporal data

**Python Example:**
```python
from tensorly.decomposition import tucker

tensor = tl.tensor(tl.randn((5, 4, 3)))

# Tucker decomposition with ranks along each mode
core, factors = tucker(tensor, ranks=[2, 2, 2])

print("Core tensor shape:", core.shape)
for f in factors:
    print("Factor matrix shape:", f.shape)
```

---

## 4. Tensor Train (TT) Decomposition

**Definition**  
Factorizes a high-dimensional tensor into a chain of **3D tensors** (TT-cores):
```
T[i₁, i₂, ..., i_d] = G₁[i₁] G₂[i₂] ... G_d[i_d]
```
- Each `G_k` is a 3D tensor of shape `(r_{k-1}, n_k, r_k)`
- `r₀ = r_d = 1`, `r_k` are TT-ranks

**Advantages:**
- Memory complexity: `O(d * n * r²)` instead of `O(n^d)`
- Suitable for extremely high-order tensors (e.g., NLP, quantum physics)
- Can represent very large tensors without storing them fully

**Applications:**
- Compressing embedding layers in NLP models
- Large-scale PDE solutions
- Quantum many-body systems

**Python Example:**
```python
import tensorly as tl
from tensorly.decomposition import tensor_train

tensor = tl.tensor(tl.randn((4, 4, 4, 4)))

# TT decomposition with max TT-rank=3
cores = tensor_train(tensor, rank=3)

print("Number of TT-cores:", len(cores))
for g in cores:
    print("Core shape:", g.shape)
```

---

# Experiments 
## CP / Tucker / Tensor Train (TT) Decomposition Comparison

| Criteria             | CP (CANDECOMP/PARAFAC)                                                      | Tucker (M-mode SVD)                                                   | Tensor Train (TT)                                                      |
|----------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|
| **Structure**        | Sum of rank-1 tensors: $\sum_{r=1}^R a_r \circ b_r \circ c_r$               | Core tensor + factor matrices: $\mathcal{G} \times_1 A \times_2 B \times_3 C$ | Sequence of 3-way core tensors (TT-cores) $G_k$                        |
| **Rank**             | Single scalar $R$ — number of rank-1 components                            | Vector of mode ranks $(R_1, R_2, R_3)$                                | TT-ranks: $r_1, r_2, \dots, r_{d-1}$                                   |
| **Compression**      | Very high if rank is small, simple structure                               | Flexible: separate rank per mode, but core size $\prod R_i$ may grow large | Extremely compact: total storage scales linearly with $d$              |
| **Computation / Stability** | Simple to compute but can be unstable for large $R$                      | More stable due to HOSVD/HOOI initialization, similar to PCA on tensors | Stable like Tucker but compact like CP; sequential computation friendly |
| **Interpretability** | Clear: each rank-1 term is an independent latent factor                    | Can be viewed as PCA extension for tensors, easy to interpret factors | Less intuitive to interpret individual components                      |
| **Key Advantages**   | Simple model; easy to implement; strong compression                        | Flexible; preserves multi-mode structure                             | Extremely efficient for high-order tensors; strong compression         |
| **Main Drawbacks**   | Sometimes unstable; rank selection is NP-hard                              | Core tensor can be large if ranks are not small                       | Harder to interpret; rank selection more complex; requires specialized algorithms |
| **Common Applications** | Topic modeling, latent variable models, CNN layer compression               | Multi-modal data reduction, tensor PCA, neural network compression   | Neural network compression (embeddings, fully connected layers), scientific high-dimensional data |
| **Real Examples**    | Fast and strong compression in RNNs (e.g., CP-GRU, TT-GRU outperform baseline) | Used with CP for optimizing tensor compression in binary networks    | TT-format effective for very high-order tensors, memory-efficient      |



# Reference 
1. Introduction to tensor decomposition [https://arxiv.org/pdf/1711.10781]
2. The Singular Value Decomposition, Applications and Beyond [https://arxiv.org/pdf/1510.08532]
3. Randomized algorithms for low-rank matrix approximation: Design, analysis, and applications
4. Fundamental Tensor Operations for Large-Scale Data Analysis in Tensor Train Formats [https://arxiv.org/pdf/1405.7786]



