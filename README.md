# Symbolic Pattern Forest (SPF)

This repository contains the code accompanying the paper, "[Linear Time Complexity Time Series Clustering with Symbolic Pattern Forest](https://www.ijcai.org/proceedings/2019/0406.pdf)" (Xiaosheng Li, Jessica Lin and Liang Zhao, IJCAI 2019). This paper proposes a time series clustering algorithm that has linear time complexity.

## To Compile the Code

Assume using a Linux system:

`g++ -O3 SPF.cpp libmetis.a -std=c++11 -o SPF`

Or can directly use the compiled file SPF included in the folder.

## To Run the Code

`./SPF [datasetname] [ensemble_size]`

\[datasetname\] is the name of the dataset to run, the user needs to place a folder named with the \[datasetname\] and the folder contains a training file \[datasetname\]_TRAIN and a testing file \[datasetname\]_TEST (The [UCR-Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/) format). \[ensemble_size\] is the ensemble size. Please see the FaceFour example contained in the directory.

## Example

`./SPF FaceFour 100`

Output:

```
dataset:FaceFour, ensemble size:100
rand index: 1
The running time is: 1.860000seconds
```

## Note

The code uses a char array buffer of size 1000000 to read each line of the input file, so if the time series to use is very long, the characters that each line the input file contains may surpass the limit. In this case the buffer limit (line 32 of SPF.cpp, MAX_PER_LINE) should be enlarged correspondingly.

## Citation

