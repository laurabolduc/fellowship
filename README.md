This project used game camera images taken in series of 3 when motion was detected. 
The binary image preprocessing concatenates each series of 3 into 1 single image and resizes them to 450x200 pixels.
Then it labels each one either ANIMAL or NOANIMAL based on the CSV. 
Those are sorted into respective folders.
Like the binary image preprocessing, the multiclass image preprocessing concatenates each series of 3 into 1 single image and resizes them to 450x200 pixels.  
However, the multiclass classification labels and sorts the images using the CSV into folders labeled by animal. 
A seperate folder is created for each different animal species in the dataset unless there are less than 10 occurances in the data set. 
Those are grouped together in the folder SOMETHINGHERE. There is also a folder for NOANIMAL. 

After the images have been sorted, there are 4 different ways I tested training a CNN.
The first is with just the data as it is (training.py).
The second is with data augmentation which I founnd to be fairly affective, though it did increase the run time (training_data_aug.py). 
The third is with trasfer learning, which for the significant increase in run time I found to not be effective (training_transfer_learn.py).
The fourth is with both transfer learning and data augmentation which again because of the TL, increased the run time significantly with little reward (training_both.py).

Disclaimer: These files cannot be run from github as the images are stored on the supercomputer at St. Lawrence University and are too large to be stored on a local computer. 
