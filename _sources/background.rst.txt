Background
==========================

Chimeric-Tools aims to rigourosly test ensemble algorithms, algorithms that map a set of individual forecasts into a single forecast, of the trajectory of an infectious agent by (i) generating simulated outbreaks that depend on past surveillance data, (ii) training a set of individual models to produce forecasts of this simulated outbreak, (iii) applying the proposed ensemble algorithm to aggregate these individual models, and (iv) scoring the ensemble algorithm. 
Past work has tested novel ensemble algorithms on empirical data which makes it difficult to extract statistical properties of the newly proposed ensemble algorithm and compare two ensembles unless they are trained on the same dataset and incorporate the same individual models into their ensemble.
The Chimeric-Tools package allows one to compare the performance of two or more different ensemble algorithms on the same surveillance data and on the saem set of individual models. 
Chimeric-tools offers infecitous disease modelers a testbed for ensemble algoruthms in an effort to more quickly improve ensemble algorithms that generate forecasts of a pathogen.  
