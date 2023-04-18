## Training & Evaluation Introduction

### Training introduction

Multiple training strategies are supported by MindVideo.For main-stream neural-based models, automatic gradient descent is well equipped and set as default training strategy.  In addition, users who need an unusual training strategy can customize the Trainer.

To control the training method, we design a series of training parameters in config, and you can check the config file for more information.

### Evaluation introduction

The function of evaluation module is to implement commonly used evaluation protocols for recommender systems. Since different models can be compared under the same evaluation modules, MindVideo standardizes the evaluation of recommender systems.

#### Evaluation method

The evaluation method supported by MindVideo is as following. Among them,user can customize batch size,pretain model,dataset.
