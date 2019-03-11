# Shuffle
This doc contains experimental results of shuffling.

I will test 4 different dataset, 6 different shuffle strategy, 2 different models, different layers.

No data augmentation; no dropout; no pre-training; same learning rate; same learning schedule; different dataset use different normalization for images

Dataset: CIFAR100, CIFAR10, STL10, SVHN
Shuffle strategy: shuff_all, shuff, rand, crand, cshuff, none
Models: VGG16, ResNet50
*The numbers in the following table are test accuracy in percent.

## TODO Lists

### TODO from Max Losch:
generalization, training curves, dropout, transfer learning, robustness, deconv, activation maximization 

searching keyword: permutation

paper: On the robustness of convolutional neural networks to internal architecture and weight perturbation; Understanding deep learning requires rethinking generalization; Rethinking generalization requires revisiting old ideas: statistical mechanics approaches and complex learning behavior

### TODO from FAN YUE:
check if the effective capacity of shuff_nets is sufficient for memorizing the entire data set by Randomization tests.

- if shuff is really a regularizer?
- how many layers should be shuffled or 1D?
- does the shuffled net have enough effective capacity to fit Randomized label or noisy images? i.e. how much capacity does shuff remove from the model?
- does the order of channels the only way to encode info or the main way? how many different ways are there for NNs to encode info? Justify.
- anaylize a small MLP to demonstrate the enormous capacity is from factorial
- design experiments or quantity to measure the effective capacity of S and C
- relation to over- / underfitting.
- the ability of fitting in terms of number of classes

### TODO from Yongqing: 
fix initial, test without shuff, conv with few fc, randomize the weights and only train the fc
paper keywords: permutation invariant mnist--Lecunn

### TODO from Marius: 
The Limitations of Adversarial Training and the Blind-Spot Attack - specify 10 test imgs, train 100 plain models, draw the bar-graph for each test img; train 100 badly shuffled (low test acc) models, draw the bar-graph; train 100 well shuffled (high test acc) models, draw the bar-graph

