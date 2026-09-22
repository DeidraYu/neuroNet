# Vector and Matrix operations

The `math` namespace defines the vector and matrix types the network is built from. Elements may be of any arithmetic type, such as `uint16_t`, `int`, `float` or `double`. Where an expression mixes two types the result takes the wider of them, so an `int` vector times a `float` scalar gives a `float` vector.

In the descriptions below let `u` and `v` be of type `Vector`, let `A` and `B` be of type `Matrix`, and let `s` be a scalar.

## How the data is stored

A `Vector<T>` holds its elements in one contiguous block.

A `Matrix<T>` also holds all of its elements in a single contiguous block, laid out one row after another, so that element `(i, j)` lives at index `i * numCols + j`. It is not a collection of row objects. Three things follow from that, and all three are the reason for it:

- A row can be walked from start to end without following a pointer, which is what allows the inner loop of a product to be vectorized.
- The whole matrix can be handed to something else, a file or eventually a graphics card, as one buffer.
- There is one allocation per matrix rather than one per row.

## Evaluating expressions

Every operation is carried out immediately and returns a new object. There is nothing lazy about them, so there is no difference between the value an expression has and the value it is later assigned to:

```cpp
Vector<int> u{1, 2, 3};
Vector<int> v{4, 5, 6};

Vector<int> w = u + v;  // w is [5, 7, 9]
auto x = u + v;         // x is a Vector<int>, the same thing
```

An earlier version of this code did the opposite: an expression built a small object describing the work, and the arithmetic only happened when that object was assigned to a vector. That avoided the intermediate results, but it does not carry over to the flat, one-operation-at-a-time layout that a graphics card needs, and measurement showed it was not buying anything here, because nearly all of the arithmetic is in the matrix-vector product rather than in the element-wise work around it. The earlier version is kept, out of the build, under `expression-templates/` for reference.

The one exception is `outer`, described at the end of this page.

## The `Vector` class

### Two vectors

- `u + v` element-wise addition
- `u - v` element-wise subtraction
- `u * v` element-wise multiplication
- `u.dot(v)` the inner product, a scalar

### A vector and a scalar

- `u + s` adds `s` to every element
- `u - s` subtracts `s` from every element
- `u * s` multiplies every element by `s`

### Element-wise functions

- `sigmoid(u)`, `sigmoid_prime(u)`, `sin(u)`, `cos(u)` apply the function to every element and return a new vector

### Compound assignment

These change the vector in place and allocate nothing. They accept either a vector of the same length or a scalar:

- `u += v`, `u -= v`, `u *= v`
- `u += s`, `u -= s`, `u *= s`

### Other operations

- `u[i]` access to one element
- `u.size()`, `u.fill(value)`, `u.resize(n)`
- `u.argmax()` the index of the largest element
- `u.data()` a pointer to the first element
- `Vector<T>::rand(n, min, max)` a new vector of uniformly distributed values
- `u.toString()`, and `operator<<` for streams

## The `Matrix` class

### Access

- `A(i, j)` access to one element, the usual way to reach into a matrix
- `A[i]` a `RowView` onto row `i`: a pointer and a length, which can be indexed and assigned to, but which does not own its memory and stops being valid once `A` is resized or destroyed
- `A.getNumRows()`, `A.getNumCols()`, `A.data()`

### A matrix and a vector

- `A * u` the matrix-vector product, giving a vector with one element per row of `A`
- `A.transposeMult(u)` the product of the transpose of `A` with `u`, giving a vector with one element per column of `A`

`transposeMult` exists so that back propagation, which needs the transpose, does not have to form it. Transposing would copy the whole matrix and would leave the inner loop walking down a column, across the stride, instead of along a row.

### A matrix and a scalar

- `A + s` adds `s` to every element
- `A * s` multiplies every element by `s`

### Two matrices

- `A += B` adds `B` into `A` in place

### Other operations

- `A.fill(value)`, `A.size()` (the number of rows)
- `Matrix<T>::rand(rows, cols, min, max)`
- `A.toString()`, and `operator<<` for streams

## The outer product

`outer(u, v)` is the matrix whose element `(i, j)` is `u[i] * v[j]`.

It is the one operation that is not carried out immediately. It returns a small object holding references to the two vectors, and the arithmetic happens when that object is assigned to a matrix or added into one:

```cpp
Matrix<float> M = outer(u, v);   // forms the matrix
N += outer(u, v);                // adds it into N without forming it
```

The reason is that back propagation accumulates an outer product into the weight gradient once per training sample. Forming it first would allocate and fill a matrix the size of all the weights every time, which measured at roughly a third of the training time. Because it keeps references rather than copies, it is meant to be used in the statement that creates it, and not stored.

## A note on the inner loops

The inner product uses `std::transform_reduce` with `std::execution::unseq` rather than a plain loop. Adding into a single running total makes each step wait for the one before it, and because floating point addition is not associative a compiler is not allowed to reorder that, so nothing vectorizes. The policy says the steps may be interleaved, which allows it. Measured over one epoch of the first layer: a plain loop 525 ms, hand written independent totals 305 ms, this 103 ms.

The loops that write each element once, such as adding in an outer product, are left as plain loops. There is nothing accumulated across steps for a compiler to be cautious about, so it vectorizes them unaided, and routing them through `std::transform` with the same policy measured five times *slower* with MSVC.

Note that this is `unseq` and not `par_unseq`. The unsequenced policy only permits vectorization, which fixes the order of the additions when the code is compiled and keeps results reproducible from one run to the next. A parallel policy hands the work to a thread pool, which lets the order depend on which thread finishes first, and made the same seed give different answers on different runs.
