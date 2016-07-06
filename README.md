
### v0.001  time: 2016/07/04 
- Split model into cifar10_model.py file, and change it into slim version.
- The original version of cifar10_eval.py will come to an NOTNAMEERROR. So I change the saver to restore tf.all_variables()

### v0.002  time: 2016/07/05
- Change the model according to the idea of AlexNet (3x3 filter, more deeper).
- Too small stddev (1e-4) of weight initialization leads to gradient vanish problem.
- Remove batch_normaliztion leads to gradient vanish/explore problem (deep network).

### v0.003  time: 2016/07/06
- Change fully-connected layer into average_pooling layer
