# AudioTexturesRISpec
PyTorch Implementation of "Sound texture synthesis using RI spectrograms" by Hugo Caracalla and Axel Roebel.

One minor difference:

* The optimizer for LBFGS in PyTorch is slightly different from scipy, so the number of iterations is counted differently.
* All hyperparameters may not match the original paper, but I was able to get similar synthesis quality.

Thanks to Dr. Roebel for sharing a link to the official Repo so I could figure out what was causing problems in my implementation.
  
