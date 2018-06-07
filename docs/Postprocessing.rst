#################################
Post-processing Optimised Weights
#################################

After optimal weights have been generated, it is often useful to do some post-processing.
In particular it is very likely that you want to use the optimisation to generate a 
**portfolio allocation**, i.e something that you could actually buy. However, it is not 
completely trivial to convert the continuous weights that are the output of any optimisation
into an actionable allocation. 

It is not as easy as multiplying the weights by the total allocation 


.. note::
    This does not support portfolios with short selling.

