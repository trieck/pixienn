# PixieNN Project Documentation

## Project Overview

Pixienn is a modern C++ reimplementation of the well-known darknet project, originally created by Joshua Redmon.
Motivated by a desire to gain a deeper understanding of Convolutional Neural Networks (CNNs) for image processing, the
project aims to provide a fresh perspective on darknet's functionalities while utilizing modern C++ practices.

## Background and Motivation

The inspiration behind Pixienn stems from the renowned darknet project. In the pursuit of better understanding CNNs for
image analysis, I embarked on the journey to reimplement darknet in modern C++. This endeavor serves as an educational
opportunity to delve into the intricacies of CNNs, with a specific focus on object detection. By undertaking a complete
reimplementation of darknet's features, the goal is to enhance my understanding of the underlying mechanics and,
concurrently, to structure and design the project in alignment with personal preferences.

## Implemented Features

As of the current development stage, Pixienn boasts successful implementation of inference functionality for several
object detector models. The supported models include YOLOv1-tiny, YOLOv3-tiny, and YOLOv3. Users can leverage Pixienn
for accurate and efficient object detection tasks.

## Ongoing Work: Training Implementation

The ongoing focus of Pixienn's development is the implementation of training functionality, with specific attention
directed towards YOLOv1. This phase involves integrating the training pipeline, experimenting with hyperparameters, and
incorporating data augmentation techniques. The objective is to provide users with a comprehensive solution for training
object detection models within the Pixienn framework.

Pixienn serves as a platform for learning, experimentation, and customization. The combination of modern C++ practices,
a focus on educational goals, and the reimagining of darknet's capabilities makes Pixienn a unique and valuable resource
for those interested in delving into the world of CNNs and object detection.

## Features

List the key features of Pixienn, both existing and planned. This can include inference capabilities, training
functionality, supported models, etc.

## Getting Started

### Installation

Provide step-by-step instructions for installing Pixienn, including any dependencies.

### Quick Start Guide

Offer a simple guide to running inference or training a model for new users.

## Usage

### Training

`pixienn-train` wiil frequently output statistics while training.

Some of the statistics it outputs are:

* **Avg. IoU** (Average Intersection over Union): A measure how well the predicted boxes overlap with the ground truth
  box.
  It's computed as the intersection area divided by the union area.
* **Pos. Cat.** (Positive Category): This represents the fraction of correctly predicted categories among the
  instances where a category is present. It focuses on the correct identification of the category when an object of that
  category is present
* **All Cat.** (All Category): This is a more general metric that considers all instances, both true positives and false
  positives, across all categories. It provides an overall assessment of the model's ability to predict categories,
  regardless of whether an object of that category is actually present.
* **Pos.Obj.** (Positive Object): Measures the objectness scores specifically for correctly identified objects.
* **Any Obj.** (Any Object):  Measures the objectness scores for all instances, including both correctly and incorrectly
  identified objects.
* **Count**: The count of detected objects or instances.
* **Loss**: The current value of the loss function, which the model is trying to minimize during training.
* **Avg. Loss** (Average Loss): The average value of the loss function.
* **LR** (Learning Rate): The current learning rate used for updating the model weights during optimization.
* **Batches seen**: The total number of batches processed during training.

## Folder Structure

Provide an overview of the project's directory structure. Explain the purpose of major directories and key files.

## Configuration

Document any configuration options available to users. Explain how to customize settings for both inference and
training.

## Dependencies

List and describe the external libraries or tools Pixienn relies on. Include version requirements if applicable.

## Examples

Include examples of how to use Pixienn for common tasks. This can be helpful for users to quickly understand how to
integrate Pixienn into their projects.

## Performance

If relevant, provide information about the expected performance of Pixienn, including hardware recommendations and
benchmark results.

## Contributing

Encourage contributions from the community. Explain how others can contribute to the project, whether it's through bug
reports, feature requests, or code contributions.

## License

Specify the project's license. This information is crucial for users and contributors to understand how they can use,
modify, and distribute Pixienn.

## Contact

Provide a way for users to get in touch with you, whether it's through an email address, a link to a discussion forum,
or any other preferred method.
