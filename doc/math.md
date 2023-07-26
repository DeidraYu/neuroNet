# Vector and Matrix operations

The `math` namespace defines classes for vector and matrix operations. Below is a description of the operations that are currently implementd and must be implemented. Scalars, vectors, or matrices have a type. This type can be any arithmetic type such as `uint16_t`, `int`, `float`, `double`. In the description below the `int` type is used.

## The `Vector` class
In the following let `u` and `v` be of type `Vector` and let `s` be a scalar.

### Expressions with two vectors `u` and `v`

- `u + v` is an elementwise addition
- `u - v` is an elementwise subtraction, *tbd*
- `u * v` is an elementwise multiplication
- `u.dot(v)` is a vector dot product (inner product)

### Expressions with vectors `u` and a scalar `s`

- `u + s` is an vector-scalar addition
- `u - s` is an vector-scalar subtraction, *tbd*
- `u * s` is an vector-scalar multiplication

### Evaluating expressions

Let us define two integer vectors `u` and `v` and add them together in different ways.

```
// create and initialize two vectors
Vector<int> u{1, 2, 3};
Vector<int> v{4, 5, 6};

// add them together
auto w = u + v;
Vector<int> x = u + v;
Vector<int> y{u + v};

// print the results
std::cout << "w: " << x << std::endl;
std::cout << "x: " << x << std::endl;
std::cout << "y: " << x << std::endl;
```

When running this code we get:

```
w: [5, 7, 9]
x: [5, 7, 9]
y: [5, 7, 9]
```

All three are the same, aren't they? No, they are not. They seem the same because the print functionality just works for them, but there is a difference between the `auto` version and the `Vector<int>` versions. This simple math library makes used of Expression Templates. Instead directly evaluating the operations in the typical order, and expression tree is build. In this example this means that `u + v` is *not* a vector that contains the three resulting values, instead, it a tree with the addition operator at the root and the vectors `u` and `v` at the leave nodes. The type of this expression is `math::VectorVectorAddition<math::Vector<int>, math::Vector<int>>`. When using the `auto` keyword to define `w` this is also what the type of `w` is.

The two lines starting with `Vector<int>` construct a new vector and call the constructor of the `Vector<int>` class. This constructor enforces the evaluation of the expression at the right hand side. This delayed or lazy evaluation is done to optimize performance, in particular by avoiding the creation of temporaries.

The evaluation of the vector expressions is also done when using the `+=`, `-=`, and `*=` operators and when calling the `.dot()` method.

### Assignment
- `auto w = u + v;`,  `w` is a vector expression
- `Vector<int> x = u + v;`, `x` is a vector

### Compound assignment operators

With a vector expression
- `x += u`, `x` is incremented by the evaluated version of `u`
- `x -= u`, `x` is decremented by the evaluated version of `u`, *tbd*
- `x *= u`, `x` is element-wise multiplied by the evaluated version of `u` 

With a scalar
- `x += s`, `x` all elements of x are incremented by `s` 
- `x -= s`, `x` all elements of x are decremented by `s`, *tbd*
- `x *= s`, `x` all elements of x are multiplied by `s` 






Other operations have not yet been implemented


## The `Matrix` class

### Expression with two matrices `A` and `B`
- `A + B` *tbd*
- `A - B` *tbd*
- `A * B` *tbd*

### Expression with matrix `A` and scalar `s`
- `A + s` *tbd*
- `A - s` *tbd*
- `A * s` *tbd*

### Expression with matrix `A` and vector `u`
- `A * v` is a matrix-vector multiplication (actually this is a vector expression because it evaluates to a vector.)

Rationale of using the `*` operator:
The first rationale is that `*` is the most related to the matrix-vector product in mathematical notation. The second rationale is that in the implementation of this library, the matrix data is stored in a `Vector<Vector<int>>` in row-major order. This means that when `A` is a `Matrix<int>`, then `A[0]` is a `Vector<int>` that corresponds with the first row of `A`. So, in product `A * v` we multiply the first vector (the row with index 0) of `A` by `v`, then the second vector (the row with index 1) of `A` by `v` and so on. This exactly how the vector-scalar product is performed, that uses the `*` operator as well. The difference is that the vector-scalar product internally performs scalar-scalar multiplications, while the matrix-vector product internally uses the dot product.

### Compound assignment operators
- `+=` *tbd*
- `-=` *tbd*
- `*=` *tbd*

### Expression with two vectors `u` and `v`
This particular use case we need to compute the outer product of the vectors `u` and `v`. We will write `u.outer(v)` to specify the expression.
- `u.outer(v)` *tbd*

