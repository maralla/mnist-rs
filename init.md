mnist-rs
========

A pure Rust project for handwritten digit recognition.

Goal
----

* Uses neural network for trainning and inference
* Utilize the state-of-the-art technology in nerual network to implement the project
* No third-party requirements for neural network handling
* Uses the MNIST data set for trainning
* A command line app that accepts an image with a handwritten digit in it, and outputs the top k digits with its probability.
* Supports Linux and Mac. If on Mac, uses Metal for accelaration.


Code patterns
-------------

* Uses the latest Rust edition, that is 2024.
* Uses as fewer dependencies as possible.
* MIT licensed.
* Unit tests for every function.
* Only add comments if the underlying code is hard to understand or uses tricks.
* Uses empty line for clear code block orignization.
* Avoid functions with large amounts of lines.


Misc
----

* Iterate step by step, avoid creating a large portion of code in one shot.
* Do not create markdown files for summary.
* After a step of code change, run lints and tests until all problems are fixed, do not ignore issues.
