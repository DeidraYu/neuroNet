# Initializing the weights

Back propagation tells us how to *change* the weights, but not what they should be before the first training sample arrives. That choice turns out to matter a great deal: with a badly chosen starting range the network barely learns at all, whatever the learning rate.

This page works out why, and derives the rule used by `Layer::initializeWB`. It is known as *Xavier* or *Glorot* initialization, after Xavier Glorot, who published it with Yoshua Bengio in 2010.

The argument is entirely about how large the numbers inside the network are, so it needs a few quantities from statistics. Those are defined first, from scratch, since everything afterwards leans on them.

## The statistics we need

Throughout, a *random variable* is simply a quantity whose value we do not know in advance, such as one weight drawn from a random range, or one pixel of an as yet unseen image.

### Expected value

The expected value $E[X]$ of a random variable $X$ is the value we would get by averaging $X$ over very many draws. For a variable taking values $x_1, x_2, \ldots$ with probabilities $p_1, p_2, \ldots$ it is

$$E[X] = \sum_i x_i\, p_i$$

The one property we use constantly is that expectation is **linear**, for any random variables at all:

$$E[aX + bY] = a\,E[X] + b\,E[Y]$$

In particular the expectation of a sum is the sum of the expectations. This needs no assumptions; it is true even when $X$ and $Y$ are related to each other.

### Variance

The variance of $X$ is the expected value of its squared distance from its own mean.

$$\mathrm{Var}(X) = E\!\left[\,\left(X - E[X]\right)^2\,\right]$$

In words: take how far $X$ falls from its own average, square that, and average the result. It measures how spread out $X$ is. A constant has variance $0$.

The squaring is there because the unsquared version is useless — the deviations cancel exactly, by the definition of the mean:

$$E\!\left[X - E[X]\right] = E[X] - E[X] = 0$$

Squaring makes every deviation positive so they cannot cancel. Taking absolute values would do that too, but squares can be expanded and rearranged algebraically and absolute values cannot, which is what makes the calculation below possible at all.

Expanding that definition, writing $\mu = E[X]$ and using linearity of expectation, gives a second form that is usually easier to compute with:

$$\mathrm{Var}(X) = E\!\left[(X - \mu)^2\right] = E\!\left[X^2 - 2\mu X + \mu^2\right] = E[X^2] - 2\mu^2 + \mu^2$$

so

$$\boxed{\;\mathrm{Var}(X) = E[X^2] - E[X]^2\;}$$

We will lean on one special case of this repeatedly. **If $E[X] = 0$, then $\mathrm{Var}(X) = E[X^2]$.** That is why it matters that the weights are drawn from a distribution whose mean is zero: it makes the variance and the mean square the same thing. Drawing uniformly from $[-a, a]$ gives that, because the distribution is symmetric about zero.

### Standard deviation

The variance is in the *square* of the units of $X$. To get back to something comparable with $X$ itself we take the square root:

$$\mathrm{sd}(X) = \sqrt{\mathrm{Var}(X)}$$

This is the number to reach for when asking "how big is this typically", because it is in the same units as the quantity itself.

### Independence

Two random variables are *independent* when knowing one tells you nothing about the other. A weight drawn from our random generator and a pixel of an input image are independent; two neighbouring pixels of the same image are certainly not.

Independence gives us the property that the expectation of a product splits into the product of the expectations:

$$E[XY] = E[X]\,E[Y]$$

which is what we need below to make the double sum collapse.

## How large is $z$?

A layer computes ${\bf z} = W{\bf x} + {\bf b}$, so each element of ${\bf z}$ is a sum over the $n$ inputs of the layer:

$$z_i = \sum_{j=1}^{n} w_{ij}x_j + b_i$$

Take the biases to be zero, which is how we initialize them, and drop the row index $i$ since every row behaves the same. Assume the weights $w_j$ are drawn independently of one another, with mean $0$, and independently of the inputs $x_j$.

**First the mean of $z$.** Expectation is linear, so the expectation of the sum is the sum of the expectations; and because each $w_j$ is independent of each $x_j$ the products split:

$$E[z] = \sum_{j=1}^{n} E[w_j x_j] = \sum_{j=1}^{n} E[w_j]\,E[x_j] = 0$$

since every $E[w_j] = 0$. So $z$ is centred on zero, and by the special case noted above its variance is simply $E[z^2]$.

**Now the variance.** Writing the square of the sum as a double sum over all pairs of terms:

$$\mathrm{Var}(z) = E[z^2]
= E\!\left[\left(\sum_{j} w_j x_j\right)\left(\sum_{k} w_k x_k\right)\right]
= \sum_{j}\sum_{k} E[w_j w_k x_j x_k]$$

Split that into the terms where the two indices differ and the terms where they are equal.

For $j \neq k$ the two weights are independent of each other and of the inputs, so the expectation splits into a product, two factors of which are zero:

$$E[w_j w_k x_j x_k] = E[w_j]\,E[w_k]\,E[x_j x_k] = 0 \cdot 0 \cdot E[x_j x_k] = 0$$

**Every cross term vanishes.** This is the step that does the work, and it happens purely because the weights are drawn independently and centred on zero. Note that nothing was assumed about the pixels: neighbouring pixels of an image are strongly related, and it does not matter, because one zero factor is enough.

Only the $n$ terms with $j = k$ survive. For those, the weight and the input are still independent of each other, so:

$$E[w_j^2 x_j^2] = E[w_j^2]\,E[x_j^2] = \mathrm{Var}(w)\cdot E[x^2]$$

using $E[w^2] = \mathrm{Var}(w)$, which holds because the weights have mean zero. There are $n$ such terms and each contributes the same amount, so:

$$\boxed{\;\mathrm{Var}(z) = n \cdot \mathrm{Var}(w) \cdot E[x^2]\;}$$

Two remarks. It is $E[x^2]$ and not $\mathrm{Var}(x)$ because we never assumed the inputs are centred on zero, and MNIST pixels certainly are not; the two are related by $E[x^2] = \mathrm{Var}(x) + E[x]^2$. And this result is *exact*, not an approximation for large $n$ — no appeal to the central limit theorem is involved.

The important part is the factor $n$. **The spread of ${\bf z}$ grows with the width of the layer.**

## The condition we want

Recall from [Trainer.md](Trainer.md) that the gradient of a single weight is

$$\dfrac{\partial C}{\partial W_{ij}} = v_i\, \sigma^\prime(z_i)\, x_j$$

a product of three factors, one of which is $\sigma^\prime(z_i)$. When that factor is near zero the whole gradient is too, whatever the other two are. The derivative of the sigmoid is largest at $z = 0$, where it is $0.25$, and falls away quickly on either side: $\sigma^\prime(10) \approx 4.5 \cdot 10^{-5}$. A layer whose $z$ values are large therefore receives almost no gradient at all. It is not broken, it is *saturated*: it sits far out on the flat tails of the sigmoid, where there is no slope to descend.

Keeping ${\bf z}$ small enough not to saturate would, for a single layer, be satisfied by simply bounding it to some fixed interval. What rules that out is depth: the output of one layer is the input of the next, so whatever factor a layer applies to the spread is applied again by every layer after it. If each layer multiplies the spread by $r$, then after $k$ layers it has been multiplied by $r^k$, which runs away to zero or to infinity for any $r$ other than $1$. Hitting a fixed target at the first layer says nothing about the tenth.

A way to achieve this is to ask that the spread of ${\bf z}$ stays of the same order as the spread of ${\bf x}$, whatever the width of the layer, so that a signal neither grows nor shrinks as it passes through. From $\mathrm{Var}(z) = n\,\mathrm{Var}(w)\,E[x^2]$, that asks for:

$$n_{\text{in}} \cdot \mathrm{Var}(w) = 1
\qquad\Longrightarrow\qquad
\mathrm{Var}(w) = \frac{1}{n_{\text{in}}}$$

That is the condition for the forward pass. The backward pass runs the other way: the gradient is propagated by ${\bf u} = W^{T}\pmb\gamma$, which sums over the $n_{\text{out}}$ outputs rather than the $n_{\text{in}}$ inputs. The same argument applied to that sum asks for:

$$\mathrm{Var}(w) = \frac{1}{n_{\text{out}}}$$

Both cannot hold at once unless the layer happens to be square. Glorot and Bengio proposed splitting the difference by averaging the two requirements:

$$\mathrm{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

## Turning the condition into a range

We draw uniformly from $[-a, a]$, which has variance $a^2/3$. Setting that equal to the target variance and solving for $a$:

$$\frac{a^2}{3} = \frac{2}{n_{\text{in}} + n_{\text{out}}}
\qquad\Longrightarrow\qquad
a^2 = \frac{6}{n_{\text{in}} + n_{\text{out}}}
\qquad\Longrightarrow\qquad
a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$

which is where the 6 comes from. It is not a tuned constant: it is $3 \times 2$, the 3 from the variance of a uniform distribution and the 2 from averaging the forward and backward conditions.

For our first layer, $n_{\text{in}} = 784$ and $n_{\text{out}} = 30$, giving $a = \sqrt{6/814} \approx 0.086$ rather than $2$. Repeating the variance calculation with that value gives $\mathrm{Var}(z) \approx 0.19$, so $\mathrm{sd}(z) \approx 0.44$, and

$$\sigma^\prime(0.44) \approx 0.24$$

against the maximum of $0.25$. The layer now starts on the steep part of the sigmoid, where there is a slope to descend.

## The biases

The biases are initialized to zero. They are not sums over $n$ terms, so the effect above does not apply to them, and starting at zero leaves each unit centred on the steep part of the sigmoid. Giving them random values would only shift units off centre for no benefit.

## Measured

Full MNIST, mini batch 10, seed 0, evaluated on the 10000 test images:

| configuration | 1 epoch | 5 epochs |
|---|---|---|
| $[-2, 2]$, learning rate 1.0 | 87.19% | 92.32% |
| $[-2, 2]$, learning rate 3.0 | 90.24% | 93.62% |
| scaled, learning rate 1.0 | 93.30% | 95.56% |
| scaled, learning rate 3.0 | **93.74%** | **95.73%** |

The error rate after a single epoch falls from 12.8% to 6.3%. A learning rate of 10, which the old initialization appeared to need, is worse than 3 for both.

## A note for later

The argument above assumed a sigmoid, which is roughly linear near the origin, so a signal passes through the activation with its spread more or less intact. ReLU instead discards the whole negative half of its input, which halves the variance getting through. Compensating for that asks for

$$\mathrm{Var}(w) = \frac{2}{n_{\text{in}}}$$

which is due to He and colleagues, and is what we will need when the sigmoid is replaced.
