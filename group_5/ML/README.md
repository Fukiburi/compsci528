# How to use

Obviously this wouldn't work until the ESP32 C code is installed in this repo, but this is an explanation of how I was able to make this work on my own.

## 1 Data collection

Running record_imu.py with start_imu_data_collection uncommented will start a sequence (as long as the ESP32 is connected and flashed) where 20 4-second recordings are taken of the imu data with 2-second breaks in between. All data is then automatically saved in files named [gesture]_[recording_number].txt. This is just like the homework because I didn't change anything, I just wanted to port over the code to be modified later.

```sh
python3 ./src/ML/record_imu.py --PORT [detected-port-number]
```

The display functions could just be used for visual confirmation or analysis of recordings.

## 2 SVM training and prediction

svm_test.py can just be run as a script with different factors such as dimensionality reduction type and test partition.

```sh
python3 ./src/ML/svm_test.py
```


# Future work

This training pipeline will likely be enough to create a model that we can then pickle (save our trained model to a file to be used in another script) and use during our project runtime.

The bigest difference between what I've made here and project runtime using this model is interpreting a stream of data rather than pre-loaded inputs. Off the top of my head this may mean we need to:
 - use a different ML model type to handle streamed data
 - use SVM but divide streamed data into different windows to increase likelihood of capturing a full gesture (however, because I am reducing the dimensionality to a set number the time window shouldn't matter as long as we somehow capture the whole gesture)