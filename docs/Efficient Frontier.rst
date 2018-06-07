.. _efficient-frontier:

###############################
Efficient Frontier Optimisation
###############################


.. _L2-Regularisation:

L2 Regularisation
=================


.. caution::

    if you pass an unreasonable parameter into ``target_risk`` or ``target_return``, it will likely 
    fail silently and return weird weights. *Caveat emptor* applies!


.. note::

    I realise that most optimisation projects in python use `cvxopt` rather than `scipy.optimize`, 
    but the latter is far cleaner and much more readable. 
    If it transpires that performance differs by orders of magnitude, 
    I will definitely consider switching.
