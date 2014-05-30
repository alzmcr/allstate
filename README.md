##Allstate Purchase Prediction Challenge

### Requirements
Python 2.7.5 with Scikit-Learn 0.14a1, Numpy 1.8, Pandas 0.12<br>
Windows 8, Intel i5-3230M @ 2.60Ghz, 16GB RAM<br>
Developed on a HP Envy 17 j100tx laptop<br>

### How to generate the solution
Type "python majorityvote_modelselection.py" in Python shell or
easily double click on Windows. Watch out on memory usage, even
though "should" be configured not to exceed 8 GB with the
default settings.

### Comments
Using the default setting, this will fit the model and creates 
the submission which will score 0.53705 in the private L. This 
is the setting which combined with Breakfast Pirate ABCEDF 
combination, scored 0.53715 in the private LB and .54535 in the 
public LB. On the above system configuration this will take 
approximately 3 hours. If you’re impatience, set N=10 and NS=7 
and will score 0.53710 in just 30 minutes! If you think is still
slow try setting N=8, NS=6, params=[(30,5,23)] and is going to
be even faster scoring as my best submission 0.53705 but lower
on the public LB. If still slow, get a better computer!!!

The script will perform the the following steps:

1. Prepare the data (load the files, transformation, clean and
   create the engineered features)
2. Fit the Random Forests
3. Make the prediction of the product G
4. Selected the best Random Forest given the train set accuracy
5. Do a majority vote using all the N model(s) and print the 
   score on the cross validation set
6. Do a majority vote using the NS selected model(s) and print 
   the score on the cross validation set

Then, if submit is set to False:<br>
a. Records the performance of the k-fold and loop<br>
b. Exit the loop and make the prediction on the test set, do 
   a majority vote using the selected models, fix the product
   accordingly with the state rule and create the submission file

### License
Please refer for LICENSE.txt file
