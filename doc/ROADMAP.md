# Roadmap

This page lists the concepts we intend to add to neuroNet, why each one is needed, and in which order. It is deliberately coarse. Each step is refined into its own document and its own set of tests at the moment we pick it up, not before.

The destination is a small language model that we have written ourselves and can inspect, because the question that motivates this project is whether a network that predicts well has learned any real structure, or is only recalling. That question is easier to answer on a model whose every line we wrote than on one we downloaded.

## How to use this page

Status is tracked with checkboxes: `[ ]` not started, `[~]` in progress, `[x]` done.

Working agreements, carried over from the existing code:

- The theory is written down first, in its own document under `doc/`, and the code that follows uses the same names as the mathematics. If the code needs a concept the document does not mention, the document is not finished.
- Tests assert properties and tolerances, never exact floating point values or exact scores. The reasoning is written down in `test/NearComparison.hpp`.
- Every step builds and passes on Windows and on WSL before it is called done.
- Decisions about performance are made by measuring, not by expectation. Several plausible ideas have already been dropped after measurement said they were worthless.

## Where we are now

A fully connected network of arbitrary depth, trained by hand-derived back propagation, on MNIST, reaching about 94% after one epoch and 96% after five. Sigmoid activations, quadratic cost, Xavier initialization, samples shuffled every epoch. The mathematics is derived in `Trainer.md`, `Initialization.md` and `math.md`.

Samples are fed through the network one at a time. This is the single most important fact about the current code, because it decides what the next step has to be.

---

## Stage 1: foundation

### 1. Train on a batch of samples at once

- [ ] Carry a mini batch through the network as a matrix rather than one sample at a time.
- [ ] Flat, contiguous storage for `Matrix`, with explicit row stride.
- [ ] Our own matrix multiply, behind a narrow interface with one entry point.

**Why.** Feeding one sample at a time makes every product a matrix times a *vector*. That operation reads the whole weight matrix to do comparatively little arithmetic, so it is limited by memory bandwidth and there is nothing to optimize. Measurement confirmed it: flat storage instead of a vector of rows made no difference at all on this shape, 0.98 times, which is noise.

Batching turns those products into matrix times *matrix*, where each weight that is read is reused across every sample in the batch. Only then does layout matter, and it matters a great deal: on this machine a textbook `ijk` loop reached 2.6 GFLOP/s while the same arithmetic with the `j` and `k` loops exchanged reached 39.9, because the inner loop then walks memory contiguously and the compiler can vectorize it. A blocked version was slower than the plain reordered one at these sizes, which is a reminder to measure rather than assume.

So batching, flat storage and the matrix multiply are one step, not three: the latter two are pointless without the first.

We write the multiply ourselves rather than taking Eigen. The reordered loop is within a small factor of a tuned library, and keeping it as our own code means it stays readable next to the mathematics and can be ported to the GPU later. Eigen would be reasonable if we ever measure that the remaining factor matters, which is why it sits behind a narrow interface.

**Document.** `math.md` is updated where the storage and the multiply are described.

### 2. An automatic differentiation engine

- [ ] A tape of operations, each carrying its own backward rule, and a `backward()` that walks it in reverse.
- [ ] Decide the tensor shape the tape carries: two dimensional to begin with, looping over the batch where more is needed.

**Why.** `Network::backProp` is reverse mode automatic differentiation specialised by hand to one architecture. Every new architecture currently means deriving and coding the backward pass again. Nothing later in this roadmap is reasonable without this step: it is the difference between trying an idea in an afternoon and spending a week on algebra.

Worth stating clearly because it is easy to misread: this is not a move away from analytic derivatives. Each primitive operation still has a hand-derived analytic backward rule. The engine only composes them using the chain rule, and the result is exact, not a numerical approximation.

The tensor shape belongs to this step rather than to attention, because the tape is what carries it. Attention wants batch by sequence by feature, which is three dimensional, and it is far cheaper to decide that now than to retrofit it later.

**Document.** `Autograd.md` - the chain rule over a computation graph, forward versus reverse mode and why reverse mode is the right choice when there are many inputs and one scalar output, and how the existing derivation in `Trainer.md` is the special case.

---

## Stage 2: training that actually works

### 3. The remaining training basics

- [x] Shuffle the training data between epochs.
- [x] Initialize weights scaled to the size of the layer.
- [ ] ReLU as the hidden activation.
- [ ] Cross-entropy cost.
- [ ] The Adam optimiser.

**Why.** The first two are done and were worth about six points of accuracy after one epoch. The rest address the same family of problems.

ReLU does not saturate for positive inputs, so gradients survive depth, and its derivative is cheaper than the sigmoid's. Quadratic cost combined with a sigmoid output multiplies the gradient by the sigmoid derivative, which means the network learns slowest exactly when it is most confidently wrong; cross-entropy cancels that factor. Adam adapts the step size per parameter, which matters once the network is deep enough that one global learning rate suits no layer particularly well.

Switching to ReLU changes the initialization: it discards half its input, so it wants twice the variance. That variant is already noted at the end of `Initialization.md`.

**Document.** Extends `Trainer.md` with the cross-entropy derivation, and `Initialization.md` with the ReLU variant.

---

## Stage 3: from classification to language

### 4. Character-level language modelling

- [ ] Read a text corpus, build a vocabulary, produce context and next character pairs.
- [ ] Train the existing network to predict the next character.

**Why.** This changes the task from "which of ten digits is this" to "what comes next", which is the objective every language model is built on. Doing it with the network we already have keeps exactly one thing new at a time. It also gives an interpretable measure of progress: cross-entropy in bits per character, which can be compared against the entropy of English.

**Document.** `LanguageModel.md` - next-token prediction, why cross-entropy is the natural cost, and bits per character as a yardstick.

### 5. Embeddings

- [ ] A learned lookup table mapping each token to a vector, trained with everything else.

**Why.** A character is a discrete symbol with no notion of similarity, and a network needs vectors. Embeddings are also the first place where learned structure becomes directly inspectable: we can ask which characters ended up near each other and see whether the network discovered anything real, such as vowels clustering. That is a small rehearsal of the question this whole project is aimed at.

**Document.** Folded into `LanguageModel.md`.

---

## Stage 4: the transformer

### 6. Attention

- [ ] Queries, keys and values, scaled dot-product attention, causal masking.

**Why.** This is the mechanism that made language models work. The derivation is shorter than the back propagation derivation already in `Trainer.md`; the difficulty in transformers is bookkeeping, not mathematics. Causal masking is what makes next-token prediction honest, by preventing a position from attending to its own future.

**Document.** `Attention.md` - the form of the operation, why the dot products are scaled by the square root of the key dimension, and what the mask does.

### 7. The transformer block

- [ ] Multi-head attention, residual connections, layer normalisation, position-wise feed forward.

**Why.** Each part exists to fix a specific failure, and we will have met most of them already. Residual connections keep gradients alive through depth, which is the vanishing gradient problem we hit with the sigmoid at two layers. Layer normalisation enforces at every step what initialization only sets once, which is the point `Initialization.md` closes on. Multiple heads let one layer attend to several relations at once.

**Document.** `Transformer.md`.

---

## Stage 5: a model that runs

### 8. Train a small model on the CPU

- [ ] Stack the blocks, train on a few megabytes of text.
- [ ] Sample text from it.

**Why.** Everything before this is machinery. This is where it either produces language or does not, and the loss curve is a verifiable target: it goes down or it does not.

The CPU comes first deliberately. A wrong gradient is far easier to find when intermediate values can be printed without a kernel in the way, and the CPU version remains useful afterwards as the reference that GPU results are checked against.

---

## Stage 6: the GPU

### 9. Move the training to the GPU

- [ ] Keep weights, activations and gradients resident in video memory for the whole run.
- [ ] Elementwise operations as kernels over an index space, written so the same body serves both back ends.
- [ ] The matrix multiply swapped for cuBLAS behind the interface from step 1.

**Why.** At transformer scale with real batch sizes the arithmetic is large enough that the GPU wins clearly, which is not true of the network we have today. It is also a thing worth understanding in its own right.

The structural discipline matters more than the kernels. Operations written as "for each output index, compute this" port almost mechanically, because the loop becomes a thread index. The matrix multiply does not port: fast on a CPU means cache blocking and SIMD, fast on a GPU means shared memory tiling and tensor cores, and those are different algorithms rather than the same code. That is why it sits behind its own interface from the start.

**Document.** `GPU.md` - why data stays resident rather than crossing the bus, how to tell in advance whether an operation will be limited by arithmetic, by memory bandwidth, or by the cost of launching it, and what that implies for batch size.

---

## Stage 7: the payoff

### 10. Go looking for structure

- [ ] Inspect attention patterns; look for induction heads, the heads that find an earlier occurrence of the current token and copy what followed it.
- [ ] Test in-context learning: give the model a pattern it never saw in training and see whether it continues it without any weight being updated.

**Why.** This is the actual question. Induction heads are a concrete, documented circuit that appears during training at the same moment few-shot ability does. Finding one in a model we wrote ourselves turns an argument about whether these systems understand anything into a measurement.

---

## Deliberately not on this list

- **CNNs.** The right tool for MNIST, but a detour from the question driving the project.
- **A competitive matrix multiply.** We write our own to remove an obstacle and to keep the code portable, not to beat a tuned library. If the remaining factor ever measures as important, put a real library behind the interface.
- **Training anything large.** The aim is a model small enough to understand, running on hardware we own.
