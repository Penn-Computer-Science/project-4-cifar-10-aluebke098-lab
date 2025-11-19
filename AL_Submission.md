# CIFAR10 Assignment

![Accuracy_and_Loss_Graphs](model_5_graphs.png)
![Confusion_Matrix](model_5_matrix.png)
![Evaluation_and_Model_Summary](model_5_accuracy_and_summary.png)

## Reflection
Running the model on the MNIST dataset is a lot more accurate than using the CIFAR10 dataset, which is to be expected because CIFAR10 is a lot more complex than MNIST because it used RGB instead of grayscale, and the objects to identify are more complex as well. 
I actually didn't make too many changes to it, but I did end up changing it so that all of the Conv2D layers had 3x3 p0x kernels, instead of how the first used to have a 5x5 p0x kernel. Doing so did make it a little bit more accurate, but not by a super duper significant amount.
Um I think I would try to implement the image adjustment thingy to help reduce overfitting but still allow for better accuracy. After that I don't really know what to do to make it better, like is it just add more of the same layers kinda? idk

---

## Model_lol - just for funsies
I added this because I was just curious how it would do, because it got around a 92% accuracy and 30% loss with the MNIST dataset. 
It did significantly worse with the CIFAR10 dataset, which was to be expected, at only 36% accuracy and an amazing 183% loss.

![Accuracy_and_Loss_Graphs](model_lol_graph.png)
![Confusion_Matrix](model_lol_matrix.png)

---

# CIFAR100
 
![Accuracy_and_Loss_Graphs](cifar100_graphs.png)
![Confusion_Matrix](cifar100_matrix.png)

## Model_lol (on CIFAR100)
*it's amazing*

![Accuracy_and_Loss_Graphs](cifar100_lol_graphs.png)
![Confusion_Matrix](cifar100_lol_matrix.png)