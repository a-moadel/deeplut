# DeepLut : DeepLUT: an End-to-End Library for Training Lookup Tables

Welcome to DeepLUT

Contents:
<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->
* [What is DeepLut?](#what-is-deeplut)
* [Why DeepLut is needed?](#why-deeplut-is-needed)
<!-- /TOC -->

## What is DeepLut?

![Deeplut Architecture](images/architecture.jpeg)

A Python library that aims to provide a flexible, extendible, lightening fast, and easy-to-use framework to train look-up tables (LUT) deep neural networks from scratch. Deeplut is focusing in helping researchers to:

* More rapid prototyping
* Ease of reproducing and comparing results obtained by other methods.
* Consistent integration with hardware synthesis tools. 
* Ease of extending the framework and innovating new ideas.

Deeplut is organized in the following modules

* **Initalizers**: These modules contain the implementation of initializations. It currently includes an implementation based on a learning and memorization paper.

* **Trainers**: These modules should contain a variety of differentiable lookup table representations. In one implementation, Lagrange interpolation is used. They serve as an abstraction layer for hardware lookup tables, which can be trained with back propagation and used as building blocks for higher layers.

* **Mask Builder**: These modules provide various implementations of how weights in normal neural network layers can be displayed in look-up tables. There are currently two implementations available. Expanded implementation based on modeling each weight as a standalone lookup table and randomly filling in the remaining inputs from a specific set of inputs determined by the layer implementation. In contrast, the Minimal implementation groups multiple weights into a single look-up table based on the current K.

* **Layers**: These modules contain various layer implementations and use the same naming convention as the Pytorch "nn" module. Currently, we have two implementations: linear and conv2d. The ultimate goal is to mimic all of Pytorch's main layers, which will provide a great deal of flexibility and the ability to implement more complex architectures.

* **Optimizers**: These modules provide various optimizer wrappers that can be used to wrap Pytorch optimizers and adapt them for binary and LUT training.

## Why DeepLut is needed?
We believe having a flexible, extendible, fast, and easy framework will help in advancing the research in this area. Frameworks enable innovation and make researchers focus on experimentation. Extendible will help in building an ecosystem between researchers to add plugins and implement innovative modules to benefit all other researchers.