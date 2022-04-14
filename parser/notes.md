# Training result

### v1.1:
> Result:
> loss: 6.3931 - output_content_loss: 3.4153 - output_title_loss: 2.9778 - output_content_accuracy: 0.0750 - output_title_accuracy: 0.0500
> val_loss: 19.3706 - val_output_content_loss: 9.7473 - val_output_title_loss: 9.6234 - val_output_content_accuracy: 0.0000e+00 - val_output_title_accuracy: 0.0000e+00
> No accuracy at all for every pridiction

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }
  dense_compressed_size = 16
  dense_compressed_activation = 'relu'
  hierarchy_rnn_size = 256
  dense_1_size = 128
  dense_2_dropout_ratio = 0.25
  dense_2_size = 64
  dense_content_dropout_ratio = 0.25
  dense_title_dropout_ratio = 0.45
  EPOCHS = 11
  BATCH_SIZE = 16
```

### v1.2 (change 'relu' -> 'sigmoid')
> Result:
> 1st - loss: 6.1697 - output_content_loss: 3.5206 - output_title_loss: 2.6491 - val_loss: 18.4702 - val_output_content_loss: 9.5693 - val_output_title_loss: 8.9008
> 2nd - loss: 6.3353 - output_content_loss: 3.4299 - output_title_loss: 2.9054 - val_loss: 20.1490 - val_output_content_loss: 9.8949 - val_output_title_loss: 10.2541

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }
  dense_compressed_size = 32
  dense_compressed_activation = 'sigmoid'
  hierarchy_rnn_size = 256
  dense_1_size = hierarchy_rnn_size * 3
  dense_2_dropout_ratio = 0.25
  dense_2_size = hierarchy_rnn_size
  dense_content_dropout_ratio = 0.25
  dense_title_dropout_ratio = 0.35
  EPOCHS = 11
  BATCH_SIZE = 16
```
### v1.3 (change layers'size)
> Result:
> loss: 5.9365 - output_content_loss: 3.3916 - output_title_loss: 2.5449 - val_loss: 19.0729 - val_output_content_loss: 10.0128 - val_output_title_loss: 9.0601

```python
  dense_compressed_size = 16
  dense_compressed_activation = 'relu'
  hierarchy_rnn_size = 512
  dense_1_size = hierarchy_rnn_size * 2
  dense_2_dropout_ratio = 0.25
  dense_2_size = hierarchy_rnn_size * 4
  dense_content_dropout_ratio = 0.55
  dense_title_dropout_ratio = 0.75
  EPOCHS = 51
  BATCH_SIZE = 8
```
### v1.4 (change dense_compressed_activation, change dense_1_size, dense_2_size)
> Result:
> loss: 5.9241 - output_content_loss: 3.4442 - output_title_loss: 2.4798 - val_loss: 19.1763 - val_output_content_loss: 9.9217 - val_output_title_loss: 9.2545

```python
  dense_compressed_size = 16
  dense_compressed_activation = 'sigmoid'
  hierarchy_rnn_size = 256
  dense_1_size = hierarchy_rnn_size * 4
  dense_2_dropout_ratio = 0.35
  dense_2_size = hierarchy_rnn_size * 2
  dense_content_dropout_ratio = 0.75
  dense_title_dropout_ratio = 0.75
  EPOCHS = 51
  BATCH_SIZE = 8
```
### v1.5 (train over many epochs)
> Result:
> loss: 6.0183 - output_content_loss: 3.4281 - output_title_loss: 2.5901 - val_loss: 20.7560 - val_output_content_loss: 10.8080 - val_output_title_loss: 9.9479

```python
  dense_compressed_size = 16
  dense_compressed_activation = 'sigmoid'
  hierarchy_rnn_size = 256
  dense_1_size = hierarchy_rnn_size * 4
  dense_2_dropout_ratio = 0.35
  dense_2_size = hierarchy_rnn_size * 2
  dense_content_dropout_ratio = 0.75
  dense_title_dropout_ratio = 0.75
  EPOCHS = 199
  BATCH_SIZE = 8
```

### v1.6 (big change)
> Result:
> loss: 5.9961 - output_content_loss: 3.2468 - output_title_loss: 2.7493 - val_loss: 26.0587 - val_output_content_loss: 13.2149 - val_output_title_loss: 12.8438

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }

  dense_compressed_size = 32
  dense_compressed_activation = 'relu'
  hierarchy_rnn_size = 512
  dense_1_size = 256
  dense_2_dropout_ratio = 0.15
  dense_2_size = 128
  dense_content_dropout_ratio = 0.45
  dense_title_dropout_ratio = 0.72
  EPOCHS = 101
  BATCH_SIZE = 16
```

### v1.7 (activation: 'sigmoid', increase dropout rates)
> Result:
> loss: 6.3247 - output_content_loss: 3.3778 - output_title_loss: 2.9469 - val_loss: 22.8633 - val_output_content_loss: 11.3375 - val_output_title_loss: 11.5258

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }

  dense_compressed_size = 32
  dense_compressed_activation = 'sigmoid'
  hierarchy_rnn_size = 512
  dense_1_size = 256
  dense_2_dropout_ratio = 0.35
  dense_2_size = 128
  dense_content_dropout_ratio = 0.75
  dense_title_dropout_ratio = 0.82
  EPOCHS = 121
  BATCH_SIZE = 16
```

### v1.8 (activation: 'relu', increase dense layers's size)
> Result:
> loss: 6.4213 - output_content_loss: 3.5228 - output_title_loss: 2.8985 - val_loss: 22.3720 - val_output_content_loss: 10.9530 - val_output_title_loss: 11.4191

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }

  dense_compressed_size = 64
  dense_compressed_activation = 'relu'
  hierarchy_rnn_size = 1028
  dense_1_size = 512
  dense_2_dropout_ratio = 0.45
  dense_2_size = 256
  dense_content_dropout_ratio = 0.75
  dense_title_dropout_ratio = 0.82
  EPOCHS = 121
  BATCH_SIZE = 16
```

### v1.9 (activation: 'tanh', increase dense_compressed_layer's size, decrease dense layers' size, few epochs)
> Result:
> loss: 6.2200 - output_content_loss: 3.3700 - output_title_loss: 2.8500 - val_loss: 22.1980 - val_output_content_loss: 10.8746 - val_output_title_loss: 11.3234

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }

  dense_compressed_size = 128
  dense_compressed_activation = 'tanh'
  hierarchy_rnn_size = 512
  dense_1_size = 256
  dense_2_dropout_ratio = 0.15
  dense_2_size = 256
  dense_content_dropout_ratio = 0.25
  dense_title_dropout_ratio = 0.35
  EPOCHS = 21
  BATCH_SIZE = 16
```

### v2.1:
> Result:
> loss: 0.2799 - output_content_loss: 0.2223 - output_title_loss: 0.0576 - output_content_accuracy: 0.9250 - output_title_accuracy: 1.0000
> val_loss: 61.3434 - val_output_content_loss: 28.4051 - val_output_title_loss: 32.9383 - val_output_content_accuracy: 0.0000e+00 - val_output_title_accuracy: 0.0000e+00
> Failed on validation

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }
  dense_compressed_size = 16
  dense_compressed_activation = 'relu'
  # hierarchy_rnn_size = 256  # Remove lstm layers
  dense_1_size = 128
  dense_2_dropout_ratio = 0.25
  dense_2_size = 64
  dense_content_dropout_ratio = 0.25
  dense_title_dropout_ratio = 0.45
  EPOCHS = 121
  BATCH_SIZE = 16
```

### v2.2(change 'relue' to 'sigmoid', increase denses' size, increase dropout rate):
> Result:
> loss: 2.9341 - output_content_loss: 1.6091 - output_title_loss: 1.3250 - output_content_accuracy: 0.4000 - output_title_accuracy: 0.4750
> val_loss: 31.4152 - val_output_content_loss: 15.5807 - val_output_title_loss: 15.8345 - val_output_content_accuracy: 0.0000e+00 - val_output_title_accuracy: 0.0000e+00

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }
  dense_compressed_size = 16
  dense_compressed_activation = 'sigmoid'
  # hierarchy_rnn_size = 256  # Remove lstm layers
  dense_1_size = 512
  dense_2_dropout_ratio = 0.35
  dense_2_size = 256
  dense_content_dropout_ratio = 0.45
  dense_title_dropout_ratio = 0.75
  EPOCHS = 79
  BATCH_SIZE = 16
```

### v2.3(increase denses' size):
> Result:
> loss: 2.9341 - output_content_loss: 1.6091 - output_title_loss: 1.3250 - output_content_accuracy: 0.4000 - output_title_accuracy: 0.4750
> val_loss: 31.4152 - val_output_content_loss: 15.5807 - val_output_title_loss: 15.8345 - val_output_content_accuracy: 0.0000e+00 - val_output_title_accuracy: 0.0000e+00

```python
  cfg = {
      'max_length': 1024,
      'batch_size': 16,
      'num_encoded_nodes': 201,
      'num_categories': 301,
      'hierarchical_encoded_depth': 11,
    }
  dense_compressed_size = 256
  dense_compressed_activation = 'sigmoid'
  # hierarchy_rnn_size = 256  # Remove lstm layers
  dense_1_size = 1024
  dense_2_dropout_ratio = 0.45
  dense_2_size = 512
  dense_content_dropout_ratio = 0.55
  dense_title_dropout_ratio = 0.85
  EPOCHS = 199
  BATCH_SIZE = 16
```

### v3.1 - Conv1D
> Parameters: 5,453,498
> Result:
> loss: 0.6674 - output_content_loss: 0.3795 - output_title_loss: 0.2880 - output_content_accuracy: 0.8819 - output_title_accuracy: 0.9167
> val_loss: 51.6385 - val_output_content_loss: 41.5361 - val_output_title_loss: 10.1025 - val_output_content_accuracy: 0.0469 - val_output_title_accuracy: 0.1406

### v3.2
> Parameters: 5,755,558
> Result (epoch 12/121):
> loss: 2.5641 - output_content_loss: 0.9419 - output_title_loss: 1.6223 - output_content_accuracy: 0.7308 - output_title_accuracy: 0.6346
> val_loss: 18.8432 - val_output_content_loss: 12.8224 - val_output_title_loss: 6.0208 - val_output_content_accuracy: 0.0167 - val_output_title_accuracy: 0.2583

### v3.3
> Parameters: 6,277,347
> Result: Epoch 13/121
> loss: 9.1223 - abstract_content_output_loss: 3.0903 - detail_content_output_loss: 3.3431 - title_output_loss: 2.6888
> abstract_content_output_accuracy: 0.2425 - detail_content_output_accuracy: 0.1725 - title_output_accuracy: 0.2825
> val_loss: 19.8328 - val_abstract_content_output_loss: 7.9952 - val_detail_content_output_loss: 6.6764 - val_title_output_loss: 5.1612
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.2031

### v3.3, improved: increase dropout rate
> Parameters: 6,277,347
> Result: Epoch 30/121
> loss: 7.7988 - abstract_content_output_loss: 2.5208 - detail_content_output_loss: 2.9796 - title_output_loss: 2.2984
> abstract_content_output_accuracy: 0.3325 - detail_content_output_accuracy: 0.2650 - title_output_accuracy: 0.3925
> val_loss: 20.8848 - val_abstract_content_output_loss: 8.5212 - val_detail_content_output_loss: 6.7609 - val_title_output_loss: 5.6028
> val_abstract_content_output_accuracy: 0.0391 - val_detail_content_output_accuracy: 0.0078 - val_title_output_accuracy: 0.1953

### v3.3, improved: increase max_length to 1024
> Parameters: 11,479,267
> Result: Epoch 30/121
> loss: 8.7269 - abstract_content_output_loss: 3.0854 - detail_content_output_loss: 3.1505 - title_output_loss: 2.4910
> abstract_content_output_accuracy: 0.1582 - detail_content_output_accuracy: 0.1607 - title_output_accuracy: 0.3036
> val_loss: 20.2684 - val_abstract_content_output_loss: 8.0208 - val_detail_content_output_loss: 6.3865 - val_title_output_loss: 5.8611
> val_abstract_content_output_accuracy: 0.0333 - val_detail_content_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.1833

### v3.3, changes in dropout rate, concatenated layer
> Parameters: 11,440,739
> Result: Epoch 79/121
> loss: 8.7936 - abstract_content_output_loss: 3.3576 - detail_content_output_loss: 3.0275 - title_output_loss: 2.4085
> abstract_content_output_accuracy: 0.1786 - detail_content_output_accuracy: 0.2219 - title_output_accuracy: 0.3750
> val_loss: 19.9692 - val_abstract_content_output_loss: 6.6064 - val_detail_content_output_loss: 7.3921 - val_title_output_loss: 5.9708
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.1917

### v3.3, fix bad vocab dict: use number that is not zero to represent oov (coz it makes amibigous with padding values)
> Parameters: 11,440,739
> Result: Epoch 72/121
> loss: 9.1629 - abstract_content_output_loss: 3.4851 - detail_content_output_loss: 3.1621 - title_output_loss: 2.5156
> abstract_content_output_accuracy: 0.1786 - detail_content_output_accuracy: 0.1939 - title_output_accuracy: 0.3189
> val_loss: 19.2142 - val_abstract_content_output_loss: 6.8776 - val_detail_content_output_loss: 6.4344 - val_title_output_loss: 5.9022
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.2000

### v3.3, fix bad learning_rate scheduler, set constant learning rate to 2e-4
> Parameters: 11,440,739
> Result: Epoch 56/121
> loss: 1.0063 - abstract_content_output_loss: 0.4891 - detail_content_output_loss: 0.3924 - title_output_loss: 0.1248
> abstract_content_output_accuracy: 0.8571 - detail_content_output_accuracy: 0.9031 - title_output_accuracy: 0.9898
> val_loss: 28.3955 - val_abstract_content_output_loss: 13.1539 - val_detail_content_output_loss: 8.2864 - val_title_output_loss: 6.9552
> val_abstract_content_output_accuracy: 0.0333 - val_detail_content_output_accuracy: 0.0333 - val_title_output_accuracy: 0.0333

### v3.3, Proof that show Learning Rate Schedule (case of InverseTimeDecay) gives bad performance
> Parameters: 11,440,739
> Result: Epoch 51/121
> loss: 12.1569 - abstract_content_output_loss: 4.4349 - detail_content_output_loss: 4.0980 - title_output_loss: 3.6241
> abstract_content_output_accuracy: 0.0663 - detail_content_output_accuracy: 0.0893 - title_output_accuracy: 0.0663
> val_loss: 17.5402 - val_abstract_content_output_loss: 5.9478 - val_detail_content_output_loss: 6.0303 - val_title_output_loss: 5.5621
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0250 - val_title_output_accuracy: 0.0000e+00

### v3.3, Increase dropout rate
> Parameters: 11,440,803
> Result: Epoch 51/51
> loss: 1.3658 - abstract_content_output_loss: 0.7186 - detail_content_output_loss: 0.4816 - title_output_loss: 0.1656
> abstract_content_output_accuracy: 0.8036 - detail_content_output_accuracy: 0.9005 - title_output_accuracy: 0.9847
> val_loss: 25.3659 - val_abstract_content_output_loss: 11.1590 - val_detail_content_output_loss: 7.3650 - val_title_output_loss: 6.8419
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0250 - val_title_output_accuracy: 0.0250

### v3.3, Increase dropout rate
> Parameters: 11,440,803
> Result: Epoch 51/51
> loss: 1.3539 - abstract_content_output_loss: 0.6595 - detail_content_output_loss: 0.5137 - title_output_loss: 0.1806
> abstract_content_output_accuracy: 0.8010 - detail_content_output_accuracy: 0.8852 - title_output_accuracy: 0.9796
> val_loss: 25.1427 - val_abstract_content_output_loss: 11.0590 - val_detail_content_output_loss: 7.8634 - val_title_output_loss: 6.2204
> val_abstract_content_output_accuracy: 0.0333 - val_detail_content_output_accuracy: 0.0250 - val_title_output_accuracy: 0.2083

### v3.3, Change dense_compressed's activation to sigmoid -> negative
> Parameters: 11,440,803
> Result: Epoch 51/51
> loss: 7.4920 - abstract_content_output_loss: 2.8584 - detail_content_output_loss: 2.5721 - title_output_loss: 2.0614
> abstract_content_output_accuracy: 0.2296 - detail_content_output_accuracy: 0.2423 - title_output_accuracy: 0.3878
> val_loss: 23.5684 - val_abstract_content_output_loss: 8.8083 - val_detail_content_output_loss: 7.9522 - val_title_output_loss: 6.8079
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.2083

### v3.4, Branching the convolution by various sizes of kernel
> Parameters: 21,855,899
> Result: Epoch 51/51
> loss: 2.2126 - abstract_content_output_loss: 0.9990 - detail_content_output_loss: 0.5555 - abstract_title_output_loss: 0.4889 - title_output_loss: 0.1692
> abstract_content_output_accuracy: 0.7066 - detail_content_output_accuracy: 0.8878 - abstract_title_output_accuracy: 0.8597 - title_output_accuracy: 0.9796
> val_loss: 33.8510 - val_abstract_content_output_loss: 11.1983 - val_detail_content_output_loss: 7.5611 - val_abstract_title_output_loss: 9.0011 - val_title_output_loss: 6.0905
> val_abstract_content_output_accuracy: 0.0167 - val_detail_content_output_accuracy: 0.0250 - val_abstract_title_output_accuracy: 0.0250 - val_title_output_accuracy: 0.1000

### v3.5, Branching the convolution by various sizes of kernel, merging after first conv
> Parameters: 11,452,528
> Result: Epoch 40/51
> loss: 4.2199 - abstract_content_output_loss: 1.6650 - detail_content_output_loss: 1.1192 - abstract_title_output_loss: 0.9074 - title_output_loss: 0.5283
> abstract_content_output_accuracy: 0.5383 - detail_content_output_accuracy: 0.7474 - abstract_title_output_accuracy: 0.7704 - title_output_accuracy: 0.9031
> val_loss: 30.8486 - val_abstract_content_output_loss: 9.1149 - val_detail_content_output_loss: 7.5375 - val_abstract_title_output_loss: 8.0857 - val_title_output_loss: 6.1105
> val_abstract_content_output_accuracy: 0.0167 - val_detail_content_output_accuracy: 0.0167 - val_abstract_title_output_accuracy: 0.1167 - val_title_output_accuracy: 0.1917

### v3.5, Branching the convolution by various sizes of kernel, using average instead of concatenating
> Parameters: 11,446,384
> Result: Epoch 51/51
> loss: 2.1163 - abstract_content_output_loss: 1.0478 - detail_content_output_loss: 0.5039 - abstract_title_output_loss: 0.4192 - title_output_loss: 0.1454
> abstract_content_output_accuracy: 0.6939 - detail_content_output_accuracy: 0.8724 - abstract_title_output_accuracy: 0.8903 - title_output_accuracy: 0.9770
> val_loss: 34.5822 - val_abstract_content_output_loss: 11.7685 - val_detail_content_output_loss: 7.8960 - val_abstract_title_output_loss: 8.7065 - val_title_output_loss: 6.2112
> val_abstract_content_output_accuracy: 0.0083 - val_detail_content_output_accuracy: 0.0167 - val_abstract_title_output_accuracy: 0.2000 - val_title_output_accuracy: 0.2083

### v3.5, More branching
> Parameters: 11,448,960
> Result: Epoch 51/51
> loss: 2.6326 - abstract_content_output_loss: 1.1378 - detail_content_output_loss: 0.6606 - abstract_title_output_loss: 0.5979 - title_output_loss: 0.2363
> abstract_content_output_accuracy: 0.6531 - detail_content_output_accuracy: 0.8546 - abstract_title_output_accuracy: 0.8291 - title_output_accuracy: 0.9694
> val_loss: 36.4105 - val_abstract_content_output_loss: 11.7939 - val_detail_content_output_loss: 7.8987 - val_abstract_title_output_loss: 10.2502 - val_title_output_loss: 6.4678
> val_abstract_content_output_accuracy: 0.0083 - val_detail_content_output_accuracy: 0.0083 - val_abstract_title_output_accuracy: 0.2000 - val_title_output_accuracy: 0.2333

### v3.5 (with more training data)
> Parameters: 11,448,960
> Result: Epoch 51/51
> loss: 2.1682 - abstract_content_output_loss: 0.9990 - detail_content_output_loss: 0.5440 - abstract_title_output_loss: 0.4362 - title_output_loss: 0.1890
> abstract_content_output_accuracy: 0.6983 - detail_content_output_accuracy: 0.8578 - abstract_title_output_accuracy: 0.8750 - title_output_accuracy: 0.9612
> val_loss: 38.6010 - val_abstract_content_output_loss: 13.2114 - val_detail_content_output_loss: 7.8676 - val_abstract_title_output_loss: 10.7953 - val_title_output_loss: 6.7267
> val_abstract_content_output_accuracy: 0.0167 - val_detail_content_output_accuracy: 0.0500 - val_abstract_title_output_accuracy: 0.1917 - val_title_output_accuracy: 0.4500

### v3.5 abstract outputs use only last dense layer
> Parameters: 11,448,960
> Result: Epoch 51/51
> loss: 3.4038 - abstract_content_output_loss: 1.6784 - detail_content_output_loss: 0.5252 - abstract_title_output_loss: 1.0344 - title_output_loss: 0.1658
> abstract_content_output_accuracy: 0.5237 - detail_content_output_accuracy: 0.8793 - abstract_title_output_accuracy: 0.7284 - title_output_accuracy: 0.9698
> val_loss: 34.9941 - val_abstract_content_output_loss: 10.6847 - val_detail_content_output_loss: 8.4120 - val_abstract_title_output_loss: 9.5560 - val_title_output_loss: 6.3413
> val_abstract_content_output_accuracy: 0.0333 - val_detail_content_output_accuracy: 0.0083 - val_abstract_title_output_accuracy: 0.2333 - val_title_output_accuracy: 0.2000

### v3.5 simplify embedding
> Parameters: 11,448,960
> Result: Epoch 51/51
> loss: 2.3265 - abstract_content_output_loss: 1.2813 - detail_content_output_loss: 0.3475 - abstract_title_output_loss: 0.6178 - title_output_loss: 0.0798
> abstract_content_output_accuracy: 0.6358 - detail_content_output_accuracy: 0.8879 - abstract_title_output_accuracy: 0.8276 - title_output_accuracy: 0.9806
> val_loss: 35.7847 - val_abstract_content_output_loss: 12.0912 - val_detail_content_output_loss: 7.9853 - val_abstract_title_output_loss: 9.7954 - val_title_output_loss: 5.9127
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0250 - val_abstract_title_output_accuracy: 0.3750 - val_title_output_accuracy: 0.4000

### v3.6 Big change in input layer, embeddding layer + Improve preprocess data (remove stop words) + Fix severe mistak in process dataset
> Parameters: 1,409,318
> Result: 51/51
> loss: 11.1824 - abstract_content_output_loss: 3.7726 - detail_content_output_loss: 2.3289 - abstract_title_output_loss: 3.3117 - title_output_loss: 1.7691
> abstract_content_output_accuracy: 0.1771 - detail_content_output_accuracy: 0.4354 - abstract_title_output_accuracy: 0.1813 - title_output_accuracy: 0.5667
> val_loss: 25.5393 - val_abstract_content_output_loss: 5.9934 - val_detail_content_output_loss: 7.2725 - val_abstract_title_output_loss: 5.5593 - val_title_output_loss: 6.7141
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0234 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0859

### v3.6 add conv(64, 11, 1)
> Parameters: 1,932,598
> Result: Epoch 51/51
> loss: 9.0119 - abstract_content_output_loss: 3.5294 - detail_content_output_loss: 1.5344 - abstract_title_output_loss: 2.9094 - title_output_loss: 1.0387
> abstract_content_output_accuracy: 0.1979 - detail_content_output_accuracy: 0.6896 - abstract_title_output_accuracy: 0.2771 - title_output_accuracy: 0.8271
> val_loss: 26.9909 - val_abstract_content_output_loss: 6.8721 - val_detail_content_output_loss: 7.5539 - val_abstract_title_output_loss: 5.8792 - val_title_output_loss: 6.6857
> val_abstract_content_output_accuracy: 0.0391 - val_detail_content_output_accuracy: 0.0234 - val_abstract_title_output_accuracy: 0.1562 - val_title_output_accuracy: 0.0312

### v3.6 Max length to 68000, activation to hard_sigmoid, Average layer to Maximum
> Parameters: 1,932,598
> Result: Epoch 51/51
> loss: 12.8813 - abstract_content_output_loss: 4.3829 - detail_content_output_loss: 2.7138 - abstract_title_output_loss: 3.6875 - title_output_loss: 2.0971
> abstract_content_output_accuracy: 0.0868 - detail_content_output_accuracy: 0.3542 - abstract_title_output_accuracy: 0.1076 - title_output_accuracy: 0.5278
> val_loss: 26.7691 - val_abstract_content_output_loss: 5.8129 - val_detail_content_output_loss: 6.6447 - val_abstract_title_output_loss: 6.5317 - val_title_output_loss: 7.7798
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0000e+00 - val_abstract_title_output_accuracy: 0.0156 - val_title_output_accuracy: 0.0000e+00

### v3.6 Switch to swish activation
> Parameters: 1,932,598
> Result: Epoch 51/51
> loss: 15.1035 - abstract_content_output_loss: 4.5929 - detail_content_output_loss: 3.5245 - abstract_title_output_loss: 4.0923 - title_output_loss: 2.8937
> abstract_content_output_accuracy: 0.0521 - detail_content_output_accuracy: 0.1042 - abstract_title_output_accuracy: 0.0521 - title_output_accuracy: 0.2222
> val_loss: 28.7354 - val_abstract_content_output_loss: 6.4965 - val_detail_content_output_loss: 7.1360 - val_abstract_title_output_loss: 7.3410 - val_title_output_loss: 7.7619
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0312 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0000e+00

### v3.6 Switch to elu activation
> Parameters: 1,932,598
> Result: Epoch 51/51
> loss: 14.2182 - abstract_content_output_loss: 4.5543 - detail_content_output_loss: 3.1077 - abstract_title_output_loss: 3.9717 - title_output_loss: 2.5845
> abstract_content_output_accuracy: 0.0312 - detail_content_output_accuracy: 0.1875 - abstract_title_output_accuracy: 0.0625 - title_output_accuracy: 0.2778
> val_loss: 28.3038 - val_abstract_content_output_loss: 6.4053 - val_detail_content_output_loss: 7.0855 - val_abstract_title_output_loss: 6.8094 - val_title_output_loss: 8.0037
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0000e+00 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0312

### v3.6 Switch to relu activation
> Parameters: 1,932,598
> Result: Epoch 51/51
> loss: 13.5569 - abstract_content_output_loss: 4.5131 - detail_content_output_loss: 2.9510 - abstract_title_output_loss: 3.8290 - title_output_loss: 2.2638
> abstract_content_output_accuracy: 0.0312 - detail_content_output_accuracy: 0.3299 - abstract_title_output_accuracy: 0.1215 - title_output_accuracy: 0.4653
> val_loss: 27.0275 - val_abstract_content_output_loss: 5.9476 - val_detail_content_output_loss: 6.6863 - val_abstract_title_output_loss: 6.8155 - val_title_output_loss: 7.5780
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0000e+00 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0000e+00

### v3.6 detail output: dense_1 + 2 + 3; abstract output: dense_4 + 5
> Parameters: 1,853,238
> Result: Epoch 51/51
> loss: 14.1310 - abstract_content_output_loss: 4.2415 - detail_content_output_loss: 3.4746 - abstract_title_output_loss: 3.5453 - title_output_loss: 2.8696
> abstract_content_output_accuracy: 0.0486 - detail_content_output_accuracy: 0.1250 - abstract_title_output_accuracy: 0.1354 - title_output_accuracy: 0.2014
> val_loss: 28.6284 - val_abstract_content_output_loss: 6.6041 - val_detail_content_output_loss: 6.8663 - val_abstract_title_output_loss: 7.5261 - val_title_output_loss: 7.6319
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0469 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0000e+00

### v3.6 detail output: dense_2 + 3 + 4; abstract output: dense_4 + 5
> Parameters: 1,628,790
> Result: Epoch 51/51
> loss: 12.7819 - abstract_content_output_loss: 4.0304 - detail_content_output_loss: 3.1723 - abstract_title_output_loss: 3.0871 - title_output_loss: 2.4922
> abstract_content_output_accuracy: 0.0729 - detail_content_output_accuracy: 0.2257 - abstract_title_output_accuracy: 0.2882 - title_output_accuracy: 0.3819
> val_loss: 28.4074 - val_abstract_content_output_loss: 6.4803 - val_detail_content_output_loss: 6.8092 - val_abstract_title_output_loss: 7.3041 - val_title_output_loss: 7.8138
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0156 - val_abstract_title_output_accuracy: 0.0312 - val_title_output_accuracy: 0.0312

### v3.6 dim_embeddings: 4, remove all dense_compressed layers
> Parameters: 1,612,058
> Result: Epoch 51/51
> loss: 13.8436 - abstract_content_output_loss: 4.1468 - detail_content_output_loss: 3.5128 - abstract_title_output_loss: 3.3324 - title_output_loss: 2.8516
> abstract_content_output_accuracy: 0.0729 - detail_content_output_accuracy: 0.1146 - abstract_title_output_accuracy: 0.1667 - title_output_accuracy: 0.2361
> val_loss: 27.8573 - val_abstract_content_output_loss: 6.1904 - val_detail_content_output_loss: 7.0042 - val_abstract_title_output_loss: 6.8455 - val_title_output_loss: 7.8172
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0000e+00

### v3.6 Using average to merging branches
> Parameters: 1,612,058
> Result: Epoch 51/51
> Report number: 20211206-022856
> loss: 15.2754 - abstract_content_output_loss: 4.3548 - detail_content_output_loss: 3.8856 - abstract_title_output_loss: 3.7769 - title_output_loss: 3.2582
> abstract_content_output_accuracy: 0.0486 - detail_content_output_accuracy: 0.0938 - abstract_title_output_accuracy: 0.1042 - title_output_accuracy: 0.1493
> val_loss: 28.2081 - val_abstract_content_output_loss: 6.3969 - val_detail_content_output_loss: 6.7247 - val_abstract_title_output_loss: 7.2237 - val_title_output_loss: 7.8628
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0156 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0000e+00

### v3.6 dim_embeddings: 8
> Parameters: 1,612,058
> Result: Epoch 51/51
> Report number: 20211206-040017
> loss: 15.6195 - abstract_content_output_loss: 4.4520 - detail_content_output_loss: 3.9764 - abstract_title_output_loss: 3.8444 - title_output_loss: 3.3467
> abstract_content_output_accuracy: 0.0417 - detail_content_output_accuracy: 0.0451 - abstract_title_output_accuracy: 0.0590 - title_output_accuracy: 0.1458
> val_loss: 29.0082 - val_abstract_content_output_loss: 6.4529 - val_detail_content_output_loss: 6.9376 - val_abstract_title_output_loss: 7.5205 - val_title_output_loss: 8.0972
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_abstract_title_output_accuracy: 0.0000e+00 - val_title_output_accuracy: 0.0000e+00

### v3.7.0.0 Network reconstruction: use one_hot instead of embedding, use 3 dense layer, kernel 6 conv use kernel of size 1, use 1 title output, set large dropout ratios
> Parameters: 763,438
> Result: Epoch 91/91
> Report number: 20211206-053730
> loss: 8.8821 - abstract_content_output_loss: 3.7287 - detail_content_output_loss: 2.7376 - title_output_loss: 2.4158
> abstract_content_output_accuracy: 0.1111 - detail_content_output_accuracy: 0.3194 - title_output_accuracy: 0.3299
> val_loss: 21.2924 - val_abstract_content_output_loss: 6.2629 - val_detail_content_output_loss: 6.9611 - val_title_output_loss: 8.0685
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0625 - val_title_output_accuracy: 0.0312

### v3.7.1.0 Output layer reconstruction: detail output layers concatenate all previous dense layers, abstract output layer merges detail content and detail title
> Parameters: 552,907
> Result: Epoch 91/91
> Report number: 20211206-065828
> loss: 15.7893 - abstract_content_output_loss: 5.8799 - detail_content_output_loss: 5.1341 - detail_title_output_loss: 4.7752
> abstract_content_output_accuracy: 0.0451 - detail_content_output_accuracy: 0.0347 - detail_title_output_accuracy: 0.0347
> val_loss: 17.4820 - val_abstract_content_output_loss: 6.0338 - val_detail_content_output_loss: 5.6816 - val_detail_title_output_loss: 5.7667
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.1 activation -> swish, dropout rate -> decrease, learning_rate -> 5e-4, convoluted_11 size -> 128, batch_size -> 16
> Parameters: 476,250
> Result: Epoch 131/131
> Report number: 20211206-075352
> loss: 12.7748 - abstract_content_output_loss: 4.4825 - detail_content_output_loss: 4.4086 - detail_title_output_loss: 3.8836
> abstract_content_output_accuracy: 0.0451 - detail_content_output_accuracy: 0.0417 - detail_title_output_accuracy: 0.0556
> val_loss: 19.5845 - val_abstract_content_output_loss: 6.1418 - val_detail_content_output_loss: 6.4505 - val_detail_title_output_loss: 6.9923
> val_abstract_content_output_accuracy: 0.0417 - val_detail_content_output_accuracy: 0.0417 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.2 activation -> relu, dropout rate -> increase, learning_rate -> 1e-3, optimizer -> Nadam
> Parameters: 476,250
> Result: Epoch 131/131
> Report number: 20211206-093330
> loss: ~ 13 (stagnated sine epoch ~ 100th)
> val_loss: ~ 19

### v3.7.1.3 activation -> elu, dropout rate -> decrease, learning_rate -> 2e-4, optimizer -> Adam
> Parameters: 476,250
> Result: Epoch 131/131
> Report number: 20211206-105738
> loss: 14.3035 - abstract_content_output_loss: 5.1131 - detail_content_output_loss: 4.7564 - detail_title_output_loss: 4.4340
> abstract_content_output_accuracy: 0.0483 - detail_content_output_accuracy: 0.0448 - detail_title_output_accuracy: 0.0483
> val_loss: 17.3071 - val_abstract_content_output_loss: 5.7623 - val_detail_content_output_loss: 5.7460 - val_detail_title_output_loss: 5.7987
> val_abstract_content_output_accuracy: 0.0172 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 activation -> elu, learning_rate -> 1e-4, optimizer -> Nadam
> Parameters: 465,594
> Result: Epoch 199/199
> Report number: 20211206-121028
> loss: 14.8351 - abstract_content_output_loss: 5.3061 - detail_content_output_loss: 4.9156 - detail_title_output_loss: 4.6134
> abstract_content_output_accuracy: 0.0345 - detail_content_output_accuracy: 0.0483 - detail_title_output_accuracy: 0.0552
> val_loss: 17.4488 - val_abstract_content_output_loss: 5.7916 - val_detail_content_output_loss: 5.6474 - val_detail_title_output_loss: 6.0099
> val_abstract_content_output_accuracy: 0.0345 - val_detail_content_output_accuracy: 0.0345 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 activation -> elu, learning_rate -> 1e-3, optimizer -> Nadam, dropout rate -> changed
> Parameters: 465,594
> Result: Epoch 199/199
> Report number: 20211206-132251
> loss: 12.6739 - abstract_content_output_loss: 4.4231 - detail_content_output_loss: 4.3994 - detail_title_output_loss: 3.8515
> abstract_content_output_accuracy: 0.0448 - detail_content_output_accuracy: 0.0448 - detail_title_output_accuracy: 0.0552
> val_loss: 19.9837 - val_abstract_content_output_loss: 6.2831 - val_detail_content_output_loss: 6.3990 - val_detail_title_output_loss: 7.3015
> val_abstract_content_output_accuracy: 0.0345 - val_detail_content_output_accuracy: 0.0345 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 activation -> swish, learning_rate -> 1e-3, optimizer -> RMSprop, dropout rate -> inscrease, merged_abstract_content -> Add
> Parameters: 465,594
> Result: Epoch 199/199
> Report number: 20211206-152255
> loss: 4.7660 - abstract_content_output_loss: 1.1754 - detail_content_output_loss: 1.7480 - detail_title_output_loss: 1.8426
> abstract_content_output_accuracy: 0.6586 - detail_content_output_accuracy: 0.4931 - detail_title_output_accuracy: 0.4379
> val_loss: 44.6219 - val_abstract_content_output_loss: 16.5707 - val_detail_content_output_loss: 14.9151 - val_detail_title_output_loss: 13.1361
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 dense_3 -> content_dense_3 vs title_dense_3, merged_abstract_content = content_dense_3 + title_dense_3
## (v3.7.2.0.0)
> Training samples: 286, validating samples: 38
> Parameters: 465,730
> Report number: 20211206-163412
> Result: Epoch 31/71
> loss: 10.1338 - abstract_content_output_loss: 3.2092 - detail_content_output_loss: 3.8394 - detail_title_output_loss: 3.0852
> abstract_content_output_accuracy: 0.1724 - detail_content_output_accuracy: 0.1000 - detail_title_output_accuracy: 0.1966
> val_loss: 23.5970 - val_abstract_content_output_loss: 7.2704 - val_detail_content_output_loss: 6.9511 - val_detail_title_output_loss: 9.3754
> val_abstract_content_output_accuracy: 0.0862 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0172
> Result: Epoch 71/71
> loss: 7.5381 - abstract_content_output_loss: 2.3574 - detail_content_output_loss: 2.7036 - detail_title_output_loss: 2.4771
> abstract_content_output_accuracy: 0.3621 - detail_content_output_accuracy: 0.2621 - detail_title_output_accuracy: 0.2379
> val_loss: 31.6807 - val_abstract_content_output_loss: 9.6061 - val_detail_content_output_loss: 10.4328 - val_detail_title_output_loss: 11.6419
> val_abstract_content_output_accuracy: 0.0517 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0345

### v3.7.1.4 More data
> Training samples: 349, validating samples: 42
> Parameters: 465,730
> Report number: 20211207-090145
> Result: Epoch 71/71
> loss: 7.6470 - abstract_content_output_loss: 2.4805 - detail_content_output_loss: 2.5913 - detail_title_output_loss: 2.5751
> abstract_content_output_accuracy: 0.3316 - detail_content_output_accuracy: 0.2944 - detail_title_output_accuracy: 0.2971
> val_loss: 31.1945 - val_abstract_content_output_loss: 10.9802 - val_detail_content_output_loss: 11.4512 - val_detail_title_output_loss: 8.7632
> val_abstract_content_output_accuracy: 0.0862 - val_detail_content_output_accuracy: 0.1034 - val_detail_title_output_accuracy: 0.1034

### v3.7.1.4 conv layers: Average -> Add, Add -> Maxium; dropout rate -> increase
## (v3.7.2.1.0)
> Training samples: 349, validating samples: 42
> Parameters: 555,336
> Report number: 20211207-101939
> Result: Epoch 71/71
> loss: 7.4365 - abstract_content_output_loss: 3.0620 - detail_content_output_loss: 2.2884 - detail_title_output_loss: 2.0861
> abstract_content_output_accuracy: 0.2149 - detail_content_output_accuracy: 0.3448 - detail_title_output_accuracy: 0.3899
> val_loss: 28.5857 - val_abstract_content_output_loss: 8.5881 - val_detail_content_output_loss: 10.8534 - val_detail_title_output_loss: 9.1441
> val_abstract_content_output_accuracy: 0.0172 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 conv layers: Average -> Add, Add -> Maxium; dropout rate -> increase
## (v3.7.1.1)
> Training samples: 349, validating samples: 42
> Parameters: 555,336
> Report number: 20211207-110253
> Result: Epoch 91/91
> loss: 8.9197 - abstract_content_output_loss: 3.0782 - detail_content_output_loss: 2.8887 - detail_title_output_loss: 2.9528
> abstract_content_output_accuracy: 0.1645 - detail_content_output_accuracy: 0.1883 - detail_title_output_accuracy: 0.1989
> val_loss: 27.0255 - val_abstract_content_output_loss: 8.6857 - val_detail_content_output_loss: 9.6552 - val_detail_title_output_loss: 8.6845
> val_abstract_content_output_accuracy: 0.0172 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 changes in conv kernel's cells, layer 1: conv1d(109, 5, 1), layer 2: conv1d(17, 3, 1), layer 3: conv1d(11, 3, 1), layer 4: conv1d(5,3, 1), layer 5: conv1d(3, 3, 1), layer 6: conv1d(1, 3, 1)
## (v3.7.2.0)
> Training samples: 349, validating samples: 42
> Parameters: 935,575
> Report number: 20211207-122554
> Result: Epoch 131/131
> loss: 7.4851 - abstract_content_output_loss: 2.5998 - detail_content_output_loss: 2.5344 - detail_title_output_loss: 2.3509
> abstract_content_output_accuracy: 0.2918 - detail_content_output_accuracy: 0.3024 - detail_title_output_accuracy: 0.3475
> val_loss: 26.5247 - val_abstract_content_output_loss: 8.0584 - val_detail_content_output_loss: 9.6723 - val_detail_title_output_loss: 8.7940
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0690

### v3.7.1.4 changes in conv kernel's size
## (v3.7.3.0)
> Training samples: 349, validating samples: 42
> Parameters: 935,575
> Report number: 20211207-142640
> Result: Epoch 131/131
> loss: 8.2360 - abstract_content_output_loss: 2.9393 - detail_content_output_loss: 2.6729 - detail_title_output_loss: 2.6238
> abstract_content_output_accuracy: 0.1751 - detail_content_output_accuracy: 0.2255 - detail_title_output_accuracy: 0.1936
> val_loss: 26.1106 - val_abstract_content_output_loss: 7.5663 - val_detail_content_output_loss: 9.1089 - val_detail_title_output_loss: 9.4355
> val_abstract_content_output_accuracy: 0.0345 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 make denses deeper
## (v3.8.0.0) <- (v3.7.0.0)
> Training samples: 349, validating samples: 42
> Parameters: 335,737
> Report number: 20211207-162004
> Result: Epoch 122/131
> loss: 9.9255 - abstract_content_output_loss: 3.8407 - detail_content_output_loss: 3.3737 - detail_title_output_loss: 2.7111
> abstract_content_output_accuracy: 0.0743 - detail_content_output_accuracy: 0.1326 - detail_title_output_accuracy: 0.2414
> val_loss: 47.4358 - val_abstract_content_output_loss: 13.6539 - val_detail_content_output_loss: 17.7406 - val_detail_title_output_loss: 16.0413
> val_abstract_content_output_accuracy: 0.0172 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0517

### v3.7.1.4 changes in size of output denses
## (v3.8.1.0)
> Training samples: 349, validating samples: 42
> Parameters: 322,151
> Report number: 20211208-052941
> Result: Epoch 131/131
> loss: 10.2047 - abstract_content_output_loss: 3.9305 - detail_content_output_loss: 3.4184 - detail_title_output_loss: 2.8558
> abstract_content_output_accuracy: 0.0663 - detail_content_output_accuracy: 0.1114 - detail_title_output_accuracy: 0.1883
> val_loss: 56.3562 - val_abstract_content_output_loss: 12.0744 - val_detail_content_output_loss: 18.3734 - val_detail_title_output_loss: 25.9085
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.1.4 Decrease dropout rate (v3)
## (v3.8.1.1)
> Training samples: 349, validating samples: 42
> Parameters: 322,151
> Report number: 20211208-063932
> Result: Epoch 131/131
> loss: 9.6304 - abstract_content_output_loss: 3.4139 - detail_content_output_loss: 3.1660 - detail_title_output_loss: 3.0504
> abstract_content_output_accuracy: 0.1114 - detail_content_output_accuracy: 0.1618 - detail_title_output_accuracy: 0.1645
> val_loss: 29.8590 - val_abstract_content_output_loss: 9.0238 - val_detail_content_output_loss: 9.9021 - val_detail_title_output_loss: 10.9332
> val_abstract_content_output_accuracy: 0.0172 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0345

### v3.7.1.4 Learning rate -> 1e-2, optimizer -> Adam
## (v3.8.1.2)
> Training samples: 349, validating samples: 42
> Parameters: 322,151
> Report number: 20211208-074531
> Result: Epoch 131/131
> loss: 13.2345 - abstract_content_output_loss: 4.5595 - detail_content_output_loss: 4.5587 - detail_title_output_loss: 4.1163
> abstract_content_output_accuracy: 0.0398 - detail_content_output_accuracy: 0.0398 - detail_title_output_accuracy: 0.0504
> val_loss: 19.9558 - val_abstract_content_output_loss: 6.6755 - val_detail_content_output_loss: 6.6200 - val_detail_title_output_loss: 6.6603
> val_abstract_content_output_accuracy: 0.0172 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0000e+00


### v3.7.1.4 Learning rate -> 1e-2, optimizer -> RMSprop
## (v3.8.1.3) <- (v3.8.1.1)
> Training samples: 349, validating samples: 42
> Parameters: 322,151
> Report number: 20211208-092102
> Result: Epoch 52/131 (stagnated since step 18th)
> loss: 13.2841 - abstract_content_output_loss: 4.5758 - detail_content_output_loss: 4.5759 - detail_title_output_loss: 4.1324
> abstract_content_output_accuracy: 0.0371 - detail_content_output_accuracy: 0.0371 - detail_title_output_accuracy: 0.0477
> val_loss: 21.9408 - val_abstract_content_output_loss: 7.3785 - val_detail_content_output_loss: 7.3090 - val_detail_title_output_loss: 7.2534
> val_abstract_content_output_accuracy: 0.0345 - val_detail_content_output_accuracy: 0.0345 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.2.0 (<- v3.7.1.4) Change conv kernel size, to fix hypothetic problem about losing much information when drastically change in size from layer 1 to layer 2
## (v3.8.2.0) <- (v3.8.1.3)
> Training samples: 349, validating samples: 42
> Parameters: 338,534
> Report number: 20211208-103803
> Result: Epoch 121/131
> loss: 9.0910 - abstract_content_output_loss: 3.4766 - detail_content_output_loss: 2.9956 - detail_title_output_loss: 2.6188
> abstract_content_output_accuracy: 0.0902 - detail_content_output_accuracy: 0.1645 - detail_title_output_accuracy: 0.2228
> val_loss: 55.9183 - val_abstract_content_output_loss: 16.6361 - val_detail_content_output_loss: 20.5124 - val_detail_title_output_loss: 18.7699
> val_abstract_content_output_accuracy: 0.0690 - val_detail_content_output_accuracy: 0.0172 - val_detail_title_output_accuracy: 0.0172

### v3.7.3.0 (<- v3.7.2.0) Make dense size larger to fix hypothetic problem of parameter distribution unequal drastically (information flow stacks at somewhere as bottle neck)
## (v3.8.3.0) <- (v3.8.2.0)
> Training samples: 349, validating samples: 42
> Parameters: 1,031,947
> Report number: 20211208-121837
> Result: Epoch 131/131
> loss: 1.9581 - abstract_content_output_loss: 1.0403 - detail_content_output_loss: 0.7450 - detail_title_output_loss: 0.1728
> abstract_content_output_accuracy: 0.7333 - detail_content_output_accuracy: 0.8250 - detail_title_output_accuracy: 0.9472
> val_loss: 36.6080 - val_abstract_content_output_loss: 11.3566 - val_detail_content_output_loss: 12.7264 - val_detail_title_output_loss: 12.5250
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.3.0 More data
## (v3.8.3.0)
> Training samples: 397, validating samples: 56
> Parameters: 1,031,947
> Report number: 20211208-154543
> Result: Epoch 32/131
> loss: 6.2249 - abstract_content_output_loss: 2.9938 - detail_content_output_loss: 2.4716 - detail_title_output_loss: 0.7595
> abstract_content_output_accuracy: 0.2353 - detail_content_output_accuracy: 0.3309 - detail_title_output_accuracy: 0.7819
> val_loss: 32.8379 - val_abstract_content_output_loss: 8.1948 - val_detail_content_output_loss: 10.4533 - val_detail_title_output_loss: 14.1898
> val_abstract_content_output_accuracy: 0.0972 - val_detail_content_output_accuracy: 0.0278 - val_detail_title_output_accuracy: 0.0417
> Result: Epoch 49/131
> loss: 4.7753 - abstract_content_output_loss: 2.3876 - detail_content_output_loss: 1.8693 - detail_title_output_loss: 0.5184
> abstract_content_output_accuracy: 0.3456 - detail_content_output_accuracy: 0.4828 - detail_title_output_accuracy: 0.8456
> val_loss: 36.1908 - val_abstract_content_output_loss: 8.7705 - val_detail_content_output_loss: 11.4030 - val_detail_title_output_loss: 16.0173
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0139 - val_detail_title_output_accuracy: 0.0417
> Result: Epoch 99/133
> loss: 2.4003 - abstract_content_output_loss: 1.2708 - detail_content_output_loss: 0.8795 - detail_title_output_loss: 0.2500
> abstract_content_output_accuracy: 0.6544 - detail_content_output_accuracy: 0.7721 - detail_title_output_accuracy: 0.9387
> val_loss: 31.3608 - val_abstract_content_output_loss: 8.8743 - val_detail_content_output_loss: 11.1822 - val_detail_title_output_loss: 11.3043
> val_abstract_content_output_accuracy: 0.1111 - val_detail_content_output_accuracy: 0.0556 - val_detail_title_output_accuracy: 0.0556

### v3.7.4.0 (<- 3.7.3.0)
## (v3.8.4.0) <- (3.8.3.0)
> Training samples: 397, validating samples: 56
> Parameters: 1,082,163
> Report number: 20211208-163640
> Result: Epoch 11/131
> loss: 12.9160 - abstract_content_output_loss: 4.6168 - detail_content_output_loss: 4.4764 - detail_title_output_loss: 3.8228
> abstract_content_output_accuracy: 0.0515 - detail_content_output_accuracy: 0.0490 - detail_title_output_accuracy: 0.1054
> val_loss: 16.9907 - val_abstract_content_output_loss: 5.6696 - val_detail_content_output_loss: 5.6543 - val_detail_title_output_loss: 5.6668
> val_abstract_content_output_accuracy: 0.0278 - val_detail_content_output_accuracy: 0.0694 - val_detail_title_output_accuracy: 0.0139
> Result: Epoch 32/131
> loss: 6.2736 - abstract_content_output_loss: 2.8917 - detail_content_output_loss: 2.1793 - detail_title_output_loss: 1.2026
> abstract_content_output_accuracy: 0.2623 - detail_content_output_accuracy: 0.3995 - detail_title_output_accuracy: 0.6446
> val_loss: 28.4263 - val_abstract_content_output_loss: 7.2329 - val_detail_content_output_loss: 9.2317 - val_detail_title_output_loss: 11.9616
> val_abstract_content_output_accuracy: 0.0556 - val_detail_content_output_accuracy: 0.0417 - val_detail_title_output_accuracy: 0.0000e+00
> Result: Epoch 49/131
> loss: 4.3123 - abstract_content_output_loss: 2.0403 - detail_content_output_loss: 1.6566 - detail_title_output_loss: 0.6153
> abstract_content_output_accuracy: 0.4289 - detail_content_output_accuracy: 0.5882 - detail_title_output_accuracy: 0.8211
> val_loss: 31.7256 - val_abstract_content_output_loss: 7.8815 - val_detail_content_output_loss: 10.1834 - val_detail_title_output_loss: 13.6607
> val_abstract_content_output_accuracy: 0.0278 - val_detail_content_output_accuracy: 0.0139 - val_detail_title_output_accuracy: 0.0833
> Result: Epoch 87/131
> loss: 2.5939 - abstract_content_output_loss: 1.1668 - detail_content_output_loss: 1.1056 - detail_title_output_loss: 0.3216
> abstract_content_output_accuracy: 0.6863 - detail_content_output_accuracy: 0.7059 - detail_title_output_accuracy: 0.9118
> val_loss: 30.3728 - val_abstract_content_output_loss: 8.7424 - val_detail_content_output_loss: 10.4335 - val_detail_title_output_loss: 11.1968
> val_abstract_content_output_accuracy: 0.0556 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0278

### v3.7.5.0 (<- 3.7.4.0) Implement embedding layer to reduce computation at conv layers (make input information being densed instead of sparse)
## (v3.8.5.0) <- (3.8.4.0)
> Training samples: 397, validating samples: 56
> Parameters: 829,040
> Report number: 20211209-051030
> Result: Epoch 131/131
> loss: 2.8986 - abstract_content_output_loss: 1.1003 - detail_content_output_loss: 1.1815 - detail_title_output_loss: 0.6168
> abstract_content_output_accuracy: 0.6851 - detail_content_output_accuracy: 0.6106 - detail_title_output_accuracy: 0.7981
> val_loss: 42.5652 - val_abstract_content_output_loss: 12.1679 - val_detail_content_output_loss: 15.1404 - val_detail_title_output_loss: 15.2569
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0156

### v3.7.6.0 (<- 3.7.5.0) layer 5 -> add conv(1, 1, 1) to reduce information abmount flows to dense block; batch_size -> 34, max_length -> 64000, chunk_length -> 4, learning_rate -> 5e-4, dropout rate -> decrease
## (v3.8.6.0) <- (3.8.5.0)
> Training samples: 399, validating samples: 56
> Parameters: 388,868
> Report number: 20211209-073059
> Result: Epoch 131/131
> loss: 2.3296 - abstract_content_output_loss: 0.9794 - detail_content_output_loss: 0.6428 - detail_title_output_loss: 0.7074
> abstract_content_output_accuracy: 0.7328 - detail_content_output_accuracy: 0.8211 - detail_title_output_accuracy: 0.8088
> val_loss: 80.0845 - val_abstract_content_output_loss: 26.1769 - val_detail_content_output_loss: 30.2908 - val_detail_title_output_loss: 23.6168
> val_abstract_content_output_accuracy: 0.0294 - val_detail_content_output_accuracy: 0.0147 - val_detail_title_output_accuracy: 0.0000e+00

### v3.7.6.0 More data
## (v3.8.6.0)
> Training samples: 439, validating samples: 56
> Parameters: 388,868
> Report number: 20211209-095015
> Result: Epoch 131/151
> loss: 2.6450 - abstract_content_output_loss: 1.1616 - detail_content_output_loss: 0.8091 - detail_title_output_loss: 0.6743
> abstract_content_output_accuracy: 0.6968 - detail_content_output_accuracy: 0.7783 - detail_title_output_accuracy: 0.7941
> val_loss: 55.4478 - val_abstract_content_output_loss: 14.4965 - val_detail_content_output_loss: 17.6610 - val_detail_title_output_loss: 23.2903
> val_abstract_content_output_accuracy: 0.0735 - val_detail_content_output_accuracy: 0.0588 - val_detail_title_output_accuracy: 0.0147
> Result: Epoch 151/151
> loss: 2.0448 - abstract_content_output_loss: 0.9465 - detail_content_output_loss: 0.5664 - detail_title_output_loss: 0.5319
> abstract_content_output_accuracy: 0.7104 - detail_content_output_accuracy: 0.8394 - detail_title_output_accuracy: 0.8303
> val_loss: 66.9744 - val_abstract_content_output_loss: 17.3547 - val_detail_content_output_loss: 21.2796 - val_detail_title_output_loss: 28.3401
> val_abstract_content_output_accuracy: 0.0441 - val_detail_content_output_accuracy: 0.0441 - val_detail_title_output_accuracy: 0.0147

### v3.7.6.0 More data, dropout rate -> inscrease (to improve val_loss)
## (v3.8.6.1)
> Training samples: 464, validating samples: 56
> Parameters: 388,868
> Report number: 20211209-124239
> Result: Epoch 171/171
> loss: 1.8251 - abstract_content_output_loss: 1.0015 - detail_content_output_loss: 0.4380 - detail_title_output_loss: 0.3856
> abstract_content_output_accuracy: 0.6912 - detail_content_output_accuracy: 0.8782 - detail_title_output_accuracy: 0.8803
> val_loss: 45.5095 - val_abstract_content_output_loss: 13.0424 - val_detail_content_output_loss: 15.3435 - val_detail_title_output_loss: 17.1236
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0588 - val_detail_title_output_accuracy: 0.0294

### 3.7.7.0 (<- v3.7.6.0) Significantly shrink conv block
## (v3.8.7.0) <- (v3.8.6.0)
> Training samples: 488, validating samples: 56
> Parameters: 73,920
> Report number: 20211209-163055
> Result: Epoch 73/171
> loss: 13.6372 - abstract_content_output_loss: 4.6797 - detail_content_output_loss: 4.7144 - detail_title_output_loss: 4.2431
> abstract_content_output_accuracy: 0.0284 - detail_content_output_accuracy: 0.0246 - detail_title_output_accuracy: 0.0208
> val_loss: 25.1843 - val_abstract_content_output_loss: 8.4570 - val_detail_content_output_loss: 8.1669 - val_detail_title_output_loss: 8.5604
> val_abstract_content_output_accuracy: 0.0208 - val_detail_content_output_accuracy: 0.0208 - val_detail_title_output_accuracy: 0.0208

### 3.8.0.0 (<- v3.7.6.0) conv blocks get reconstructed to improve way of encoding information
## (v3.9.0.0) <- (v3.8.6.0)
> Training samples: 514, validating samples: 56
> Parameters: 65,781
> Report number: 20211210-090114
> Result: Epoch 171/171
> loss: 10.3752 - abstract_content_output_loss: 3.8303 - detail_content_output_loss: 3.4670 - detail_title_output_loss: 3.0780
> abstract_content_output_accuracy: 0.0731 - detail_content_output_accuracy: 0.1154 - detail_title_output_accuracy: 0.1654
> val_loss: 36.5944 - val_abstract_content_output_loss: 8.8790 - val_detail_content_output_loss: 12.5039 - val_detail_title_output_loss: 15.2115
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0000e+00

### 3.8.1.0 (<- v3.8.0.0) Build conv block larger (more number of cells); dropout rate -> increase; learning rate -> increase, 1e-3; optimizer -> Adam
## (v3.9.1.0) <- (v3.9.0.0)
> Training samples: 514, validating samples: 56
> Parameters: 80,437
> Report number: 20211210-110557
> Result: Epoch 349/349
> loss: 7.5261 - abstract_content_output_loss: 2.6191 - detail_content_output_loss: 2.3639 - detail_title_output_loss: 2.5432
> abstract_content_output_accuracy: 0.2500 - detail_content_output_accuracy: 0.2962 - detail_title_output_accuracy: 0.2577
> val_loss: 50.0029 - val_abstract_content_output_loss: 16.5547 - val_detail_content_output_loss: 17.7147 - val_detail_title_output_loss: 15.7335
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0250 - val_detail_title_output_accuracy: 0.0000e+00

### 3.8.2.0 (<- v3.8.1.0) Shrink conv block 4 by reducing number of cells; learning rate -> 2e-3; Use AveragePooling1D at conv block 3
## (v3.9.2.0)
> Training samples: 514, validating samples: 56
> Parameters: 72,865
> Report number: 20211210-142738/
> Result: Epoch 131/131 (Best one, has prospect of convergence)
> loss: 9.8005 - abstract_content_output_loss: 3.3864 - detail_content_output_loss: 3.3837 - detail_title_output_loss: 3.0304
> abstract_content_output_accuracy: 0.1154 - detail_content_output_accuracy: 0.1192 - detail_title_output_accuracy: 0.1865
> val_loss: 23.7365 - val_abstract_content_output_loss: 7.4585 - val_detail_content_output_loss: 7.6677 - val_detail_title_output_loss: 8.6104
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0375 - val_detail_title_output_accuracy: 0.0250

### 3.8.3.0 (<- v3.8.2.0) Change conv stripe size to 1; Use pooling layer
## (v3.9.3.0)
> Training samples: 514, validating samples: 56
> Parameters: 169,169
> Report number: 20211211-060753
> Result: Epoch 131/131
> loss: 9.7328 - abstract_content_output_loss: 3.3103 - detail_content_output_loss: 3.2393 - detail_title_output_loss: 3.1832
> abstract_content_output_accuracy: 0.1288 - detail_content_output_accuracy: 0.1327 - detail_title_output_accuracy: 0.1442
> val_loss: 30.3566 - val_abstract_content_output_loss: 8.6805 - val_detail_content_output_loss: 11.5058 - val_detail_title_output_loss: 10.1704
> val_abstract_content_output_accuracy: 0.0375 - val_detail_content_output_accuracy: 0.0250 - val_detail_title_output_accuracy: 0.0750

### 3.8.4.0 (<- v3.8.3.0) Change conv blocks: use AveragePooling1D alongside with conv(cells, 3, 3) or conv(cells, 5, 5); Branching conv(3, 3) for detect node number.
## (v3.9.4.0)
## Reduce size of dense blocks (trying to avoid overffing by shrink decoder block); learning rate -> 1e-3
> Training samples: 514, validating samples: 56
> Parameters: 66,984
> Report number: 20211211-090745
> Result: Epoch 201/201
> loss: 10.5075 - abstract_content_output_loss: 3.7168 - detail_content_output_loss: 3.8301 - detail_title_output_loss: 2.9606
> abstract_content_output_accuracy: 0.0654 - detail_content_output_accuracy: 0.0558 - detail_title_output_accuracy: 0.1846
> val_loss: 30.4394 - val_abstract_content_output_loss: 9.4955 - val_detail_content_output_loss: 9.3765 - val_detail_title_output_loss: 11.5674
> val_abstract_content_output_accuracy: 0.0375 - val_detail_content_output_accuracy: 0.0250 - val_detail_title_output_accuracy: 0.0000e+00

### 3.8.4.1 (<- v3.8.4.0) Dropout rate -> decrease; learning_rate -> 2e-3, optimizer -> RMSprop
## (v3.9.4.1) <- (v3.9.4.0)
## Reduce size of dense blocks (trying to avoid overffing by shrink decoder block); learning rate -> 1e-3
> Training samples: 514, validating samples: 56
> Parameters: 66,984
> Report number: 20211211-113326
> Result: Epoch 231/231
> loss: 10.2232 - abstract_content_output_loss: 3.1629 - detail_content_output_loss: 4.1033 - detail_title_output_loss: 2.9569
> abstract_content_output_accuracy: 0.0962 - detail_content_output_accuracy: 0.0385 - detail_title_output_accuracy: 0.1673
> val_loss: 41.0041 - val_abstract_content_output_loss: 14.8152 - val_detail_content_output_loss: 8.2507 - val_detail_title_output_loss: 17.9382
> val_abstract_content_output_accuracy: 0.0375 - val_detail_content_output_accuracy: 0.0125 - val_detail_title_output_accuracy: 0.0000e+00

### 3.8.5.0 (<- v3.8.4.1) Branching conv layers by different levels of depth
## (v3.9.5.0) <- (v3.9.4.1)
> Training samples: 514, validating samples: 56
> Parameters: 74,124
> Report number: 20211211-141914
> Result: Epoch 157/231
> loss: 9.8821 - abstract_content_output_loss: 3.3540 - detail_content_output_loss: 3.4375 - detail_title_output_loss: 3.0906
> abstract_content_output_accuracy: 0.1077 - detail_content_output_accuracy: 0.1038 - detail_title_output_accuracy: 0.1538
> val_loss: 41.3869 - val_abstract_content_output_loss: 11.1519 - val_detail_content_output_loss: 14.0227 - val_detail_title_output_loss: 16.2123
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0250 - val_detail_title_output_accuracy: 0.0000e+00

### 3.8.5.1 (<- v3.8.5.0) increse dense size
## (v3.9.5.1) <- (v3.9.5.0)
> Training samples: 514, validating samples: 56
> Parameters: 147,731
> Report number: 20211211-160306
> Result: Epoch 85/231
> loss: 12.2536 - abstract_content_output_loss: 4.3059 - detail_content_output_loss: 4.2340 - detail_title_output_loss: 3.7138
> abstract_content_output_accuracy: 0.0442 - detail_content_output_accuracy: 0.0442 - detail_title_output_accuracy: 0.0923
> val_loss: 22.3937 - val_abstract_content_output_loss: 6.8310 - val_detail_content_output_loss: 7.0486 - val_detail_title_output_loss: 8.5141
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0500 - val_detail_title_output_accuracy: 0.0125

### 3.8.6.0 (<- v3.8.5.1) use one_hot (sparse embeddding)
## (v3.9.6.0) <- (v3.9.5.1)
> Training samples: 514, validating samples: 56
> Parameters: 169,355
> Report number: 20211211-165737
> Result: Epoch 47/231
> loss: 9.1816 - abstract_content_output_loss: 3.4923 - detail_content_output_loss: 3.0621 - detail_title_output_loss: 2.6271
> abstract_content_output_accuracy: 0.1038 - detail_content_output_accuracy: 0.1538 - detail_title_output_accuracy: 0.2250
> val_loss: 28.3941 - val_abstract_content_output_loss: 8.2509 - val_detail_content_output_loss: 9.7828 - val_detail_title_output_loss: 10.3604 - val_abstract_content_output_accuracy: 0.0125 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0125

### 3.8.6.1 (<- v3.8.6.0) merged_conv layers -> replace Maximum by Add; conv layers -> reduce cell number; dense decoder layer -> exclude dense_1 from being concatenated
## (v3.9.6.1) <- (v3.9.6.0)
> Training samples: 514, validating samples: 56
> Parameters: 144,619
> Report number: 20211212-004820
> Result: Epoch 249/249
> loss: 2.8876 - abstract_content_output_loss: 0.9314 - detail_content_output_loss: 0.9314 - detail_title_output_loss: 1.0248
> abstract_content_output_accuracy: 0.7000 - detail_content_output_accuracy: 0.6865 - detail_title_output_accuracy: 0.6654
> val_loss: 57.9011 - val_abstract_content_output_loss: 18.4852 - val_detail_content_output_loss: 20.8452 - val_detail_title_output_loss: 18.5707
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0250

### 3.8.7.0 (<- v3.8.6.1) Branching 1 more deep layer, using Add layer to merge previous branches; Include dense_1 to detail_content_dense layer and detail_title_dense layers; Replace title_dense_4 by dense_5.
## (v3.9.7.0) <- (v3.9.6.1)
> Training samples: 514, validating samples: 56
> Parameters: 156,772
> Report number: 20211212-041316
> Result: Epoch 249/249
> loss: 4.7490 - abstract_content_output_loss: 1.9232 - detail_content_output_loss: 1.4894 - detail_title_output_loss: 1.3364
> abstract_content_output_accuracy: 0.4038 - detail_content_output_accuracy: 0.4981 - detail_title_output_accuracy: 0.5712
> val_loss: 56.3419 - val_abstract_content_output_loss: 16.4657 - val_detail_content_output_loss: 18.3647 - val_detail_title_output_loss: 21.5115
> val_abstract_content_output_accuracy: 0.0125 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0750

### 3.8.7.1 (<- v3.8.7.0) Replace Add layers by Maximum layers
## (v3.9.7.1) <- (v3.9.7.0)
> Training samples: 514, validating samples: 56
> Parameters: 156,772
> Report number: 20211212-082117
> Result: Epoch 189/249
> loss: 5.5031 - abstract_content_output_loss: 2.2673 - detail_content_output_loss: 1.7235 - detail_title_output_loss: 1.5123
> abstract_content_output_accuracy: 0.3712 - detail_content_output_accuracy: 0.4327 - detail_title_output_accuracy: 0.4904
> val_loss: 56.0571 - val_abstract_content_output_loss: 17.0040 - val_detail_content_output_loss: 18.8842 - val_detail_title_output_loss: 20.1690
> val_abstract_content_output_accuracy: 0.0250 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0250

### 3.8.7.1 (<- v3.8.7.0) Exclude some pooling layers; Branching from first conv layer
## (v3.9.7.2) <- (v3.9.7.1)
> Training samples: 514, validating samples: 56
> Parameters: 341,260
> Report number: 20211212-121220
> Result: Epoch 249/249
> loss: 3.5299 - abstract_content_output_loss: 1.6634 - detail_content_output_loss: 1.0656 - detail_title_output_loss: 0.8009
> abstract_content_output_accuracy: 0.4712 - detail_content_output_accuracy: 0.6519 - detail_title_output_accuracy: 0.7596
> val_loss: 58.1701 - val_abstract_content_output_loss: 16.3095 - val_detail_content_output_loss: 20.3992 - val_detail_title_output_loss: 21.4614
> val_abstract_content_output_accuracy: 0.0125 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.1000

### 3.9.7.2 More data
> Training samples: 629, validating samples: 83
> Parameters: 341,260
> Report number: 20211212-154618
> Result: Epoch 94/249
> loss: 8.3744 - abstract_content_output_loss: 3.4120 - detail_content_output_loss: 2.7415 - detail_title_output_loss: 2.2210
> abstract_content_output_accuracy: 0.1250 - detail_content_output_accuracy: 0.2188 - detail_title_output_accuracy: 0.3250
> val_loss: 32.5411 - val_abstract_content_output_loss: 6.8176 - val_detail_content_output_loss: 9.0830 - val_detail_title_output_loss: 16.6405
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0625

### 3.9.2.0 More data
> Training samples: 629, validating samples: 83
> Parameters: 72,865
> Report number: 20211213-015759
> Result: Epoch 249/249
> loss: 6.7605 - abstract_content_output_loss: 2.3339 - detail_content_output_loss: 2.1019 - detail_title_output_loss: 2.3247
> abstract_content_output_accuracy: 0.3234 - detail_content_output_accuracy: 0.3703 - detail_title_output_accuracy: 0.2969
> val_loss: 61.2451 - val_abstract_content_output_loss: 15.3819 - val_detail_content_output_loss: 17.3366 - val_detail_title_output_loss: 28.5265
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0234 - val_detail_title_output_accuracy: 0.0469

### v3.9.8.0 (<- v3.9.7.0) Many miscellaneous changes. Remove processing text. Add more items in char dictionary. Change node encoding format.
## Return back to char embedding +  dense_compressed idea. Increase conv cell number. Change conv block structure.
> Training samples: 629, validating samples: 83
> Parameters: 525,380
> Report number: 20211213-082413
> Result: Epoch 249/249
> loss: 5.3306 - abstract_content_output_loss: 2.0432 - detail_content_output_loss: 1.7618 - detail_title_output_loss: 1.5256
> abstract_content_output_accuracy: 0.3500 - detail_content_output_accuracy: 0.4187 - detail_title_output_accuracy: 0.5047
> val_loss: 49.0776 - val_abstract_content_output_loss: 13.9459 - val_detail_content_output_loss: 15.8194 - val_detail_title_output_loss: 19.3122
> val_abstract_content_output_accuracy: 0.0703 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0625

## v3.8.6.1 More data
> Training samples: 629, validating samples: 83
> Parameters: 396,092
> Report number: 20211213-144914
> Result: Epoch 69/249
> loss: 5.5685 - abstract_content_output_loss: 2.9211 - detail_content_output_loss: 1.4218 - detail_title_output_loss: 1.2256
> abstract_content_output_accuracy: 0.2651 - detail_content_output_accuracy: 0.5968 - detail_title_output_accuracy: 0.6556
> val_loss: 38.1898 - val_abstract_content_output_loss: 7.9703 - val_detail_content_output_loss: 13.2156 - val_detail_title_output_loss: 17.0039
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0111 - val_detail_title_output_accuracy: 0.0333

### v3.9.9.0 (<- v3.9.8.0) Change the distribution of conv cells number by descent volumne.
## Not v3.9.9.0 by mistake. It still v3.8.6.1 with more data
> Training samples: 707, validating samples: 93
> Parameters: 392,508
> Report number: 20211214-011406
> Result: Epoch 249/249
> loss: 1.9504 - abstract_content_output_loss: 1.1486 - detail_content_output_loss: 0.4428 - detail_title_output_loss: 0.3590
> abstract_content_output_accuracy: 0.6639 - detail_content_output_accuracy: 0.8792 - detail_title_output_accuracy: 0.8931
> val_loss: 49.0310 - val_abstract_content_output_loss: 12.6322 - val_detail_content_output_loss: 16.7740 - val_detail_title_output_loss: 19.6248
> val_abstract_content_output_accuracy: 0.0104 - val_detail_content_output_accuracy: 0.0208 - val_detail_title_output_accuracy: 0.0104

### v3.9.9.0 (<- v3.9.8.0) Branching more at block 3, using Maximum, Add, Multiply layers to merge; Coordinate different cell size scheme; Deeper depth at embedding block;
## Change the distribution of conv cells number by descent volumne.
## learning_rate -> 5e-4, batch_size -> 32.
> Training samples: 707, validating samples: 93
> Parameters: 73,418
> Report number: 20211214-051510
> Result: Epoch 249/249
> loss: 11.8318 - abstract_content_output_loss: 4.2944 - detail_content_output_loss: 4.0929 - detail_title_output_loss: 3.4444
> abstract_content_output_accuracy: 0.0404 - detail_content_output_accuracy: 0.0430 - detail_title_output_accuracy: 0.1289
> val_loss: 22.0561 - val_abstract_content_output_loss: 5.6360 - val_detail_content_output_loss: 6.1318 - val_detail_title_output_loss: 10.2883
> val_abstract_content_output_accuracy: 0.0547 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0469

### v3.9.9.1 (<- v3.9.9.0) Reduce dropout rates
> Training samples: 707, validating samples: 127
> Parameters: 73,418
> Report number: 20211214-095359
> Result: Epoch 249/249
> loss: 11.3387 - abstract_content_output_loss: 4.2585 - detail_content_output_loss: 3.8061 - detail_title_output_loss: 3.2742
> abstract_content_output_accuracy: 0.0365 - detail_content_output_accuracy: 0.0599 - detail_title_output_accuracy: 0.1471
> val_loss: 24.6177 - val_abstract_content_output_loss: 5.5029 - val_detail_content_output_loss: 8.1878 - val_detail_title_output_loss: 10.9270
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0078 - val_detail_title_output_accuracy: 0.0234

### v3.9.9.2 (<- v3.9.9.1) Reduce dropout rates; change denses size; make embedding deeper, smalller; learning_rate -> 1e-3
> Training samples: 738, validating samples: 127
> Parameters: 198,195
> Report number: 20211215-012014
> Result: Epoch 200/249
> loss: 3.3099 - abstract_content_output_loss: 2.4154 - detail_content_output_loss: 0.4592 - detail_title_output_loss: 0.4352
> abstract_content_output_accuracy: 0.2943 - detail_content_output_accuracy: 0.8516 - detail_title_output_accuracy: 0.8789
> val_loss: 67.8841 - val_abstract_content_output_loss: 9.4932 - val_detail_content_output_loss: 29.3543 - val_detail_title_output_loss: 29.0366
> val_abstract_content_output_accuracy: 1.0234 - val_detail_content_output_accuracy: 0.0547 - val_detail_title_output_accuracy: 0.0234

### v3.9.10.0 (<- v3.9.9.2) Add dense before output layers, use 'sigmoid' activation
> Training samples: 738, validating samples: 127
> Parameters: 88,071
> Report number: 20211215-040625
> Result: Epoch 249/249
> loss: 9.6705 - abstract_content_output_loss: 2.3341 - detail_content_output_loss: 3.8851 - detail_title_output_loss: 3.4513
> abstract_content_output_accuracy: 0.2943 - detail_content_output_accuracy: 0.1133 - detail_title_output_accuracy: 0.1966
> val_loss: 26.7458 - val_abstract_content_output_loss: 16.0778 - val_detail_content_output_loss: 5.1422 - val_detail_title_output_loss: 5.5257
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0625

### v3.9.2.0 (More data)
> Training samples: 738, validating samples: 127
> Parameters: 81,625 (bigger number caused by bigger max_length)
> Report number: 20211215-085827
> Result: Epoch 249/249
> loss: 8.0125 - abstract_content_output_loss: 2.8685 - detail_content_output_loss: 2.6993 - detail_title_output_loss: 2.4447
> abstract_content_output_accuracy: 0.2132 - detail_content_output_accuracy: 0.2092 - detail_title_output_accuracy: 0.2816
> val_loss: 37.1729 - val_abstract_content_output_loss: 10.3903 - val_detail_content_output_loss: 12.3307 - val_detail_title_output_loss: 14.4520
> val_abstract_content_output_accuracy: 0.0063 - val_detail_content_output_accuracy: 0.0063 - val_detail_title_output_accuracy: 0.0500

### v3.9.10.1 (<- v3.9.10.0) output layers: detail -> 0.8, abstract -> 1.0
> Training samples: 738, validating samples: 127
> Parameters: 88,071
> Report number: 20211215-121308
> Result: Epoch 299/299
> loss: 8.3716 - abstract_content_output_loss: 3.0748 - detail_content_output_loss: 2.9238 - detail_title_output_loss: 2.3731
> abstract_content_output_accuracy: 0.1797 - detail_content_output_accuracy: 0.2448 - detail_title_output_accuracy: 0.3776
> val_loss: 22.0527 - val_abstract_content_output_loss: 10.6599 - val_detail_content_output_loss: 5.4089 - val_detail_title_output_loss: 5.9840
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0391 - val_detail_title_output_accuracy: 0.0547

### v3.9.11.0 (<- v3.9.10.1) Remove 1 block of depth; reduce conv cell number; increase stripe size; use one_hot
> Training samples: 738, validating samples: 127
> Parameters: 144,961
> Report number: 20211215-155629
> Result: Epoch 71/299
> loss: 10.8417 - abstract_content_output_loss: 3.0324 - detail_content_output_loss: 4.1380 - detail_title_output_loss: 3.6713
> abstract_content_output_accuracy: 0.1615 - detail_content_output_accuracy: 0.1042 - detail_title_output_accuracy: 0.1719
> val_loss: 19.6940 - val_abstract_content_output_loss: 8.9098 - val_detail_content_output_loss: 5.2934 - val_detail_title_output_loss: 5.4908
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0547 - val_detail_title_output_accuracy: 0.0703

### v3.9.11.1 (<- v3.9.11.0) Remove embedding to use one_hot; Remove conv block 3; In conv block 1 apply pooling; Decrease conv cell number; Decrease dropout rate, apply droput for abstract output.
> Training samples: 738, validating samples: 127
> Parameters: 166,125
> Report number: 20211216-015203
> Result: Epoch 299/299
> loss: 5.2298 - abstract_content_output_loss: 1.3711 - detail_content_output_loss: 1.4479 - detail_title_output_loss: 2.4109
> abstract_content_output_accuracy: 0.5911 - detail_content_output_accuracy: 0.6966 - detail_title_output_accuracy: 0.3880
> val_loss: 41.4272 - val_abstract_content_output_loss: 27.9963 - val_detail_content_output_loss: 7.0420 - val_detail_title_output_loss: 6.3889
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0078 - val_detail_title_output_accuracy: 0.0234

### v3.9.11.2 (<- v3.9.11.1) Apply L2 regularization and layer normalization
> Training samples: 807, validating samples: 127
> Parameters: 166,573
> Report number: 20211216-063931
> Result: Epoch 103/299
> loss: 2.1514 - abstract_content_output_loss: 1.7104 - detail_content_output_loss: 0.1270 - detail_title_output_loss: 0.1352
> abstract_content_output_accuracy: 0.6238 - detail_content_output_accuracy: 0.9976 - detail_title_output_accuracy: 0.9928
> val_loss: 24.4497 - val_abstract_content_output_loss: 7.3796 - val_detail_content_output_loss: 8.6891 - val_detail_title_output_loss: 8.2031
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0156

### v3.9.11.3 (<- v3.9.11.2) Improve regularization and normalization; Increase dropout rate
> Training samples: 807, validating samples: 127
> Parameters: 136,116
> Report number: 20211216-111155
> Result: Epoch 299/299
> loss: 2.9492 - abstract_content_output_loss: 1.9526 - detail_content_output_loss: 0.3506 - detail_title_output_loss: 0.3207
> abstract_content_output_accuracy: 0.5325 - detail_content_output_accuracy: 0.9387 - detail_title_output_accuracy: 0.9399
> val_loss: 22.6623 - val_abstract_content_output_loss: 6.0281 - val_detail_content_output_loss: 8.1942 - val_detail_title_output_loss: 8.1143
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0234 - val_detail_title_output_accuracy: 0.0391

### v3.9.11.3 (More data)
> Training samples: 932, validating samples: 127
> Parameters: 136,116
> Report number: 20211217-005328
> Result: Epoch 199/391
> loss: 3.8375 - abstract_content_output_loss: 2.4605 - detail_content_output_loss: 0.5103 - detail_title_output_loss: 0.5127
> abstract_content_output_accuracy: 0.4469 - detail_content_output_accuracy: 0.9312 - detail_title_output_accuracy: 0.9187
> val_loss: 22.3236 - val_abstract_content_output_loss: 6.3733 - val_detail_content_output_loss: 8.2806 - val_detail_title_output_loss: 7.3160
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0391

### v3.9.11.4 (<- v3.9.11.3) Restructure embedding block
> Training samples: 932, validating samples: 127
> Parameters: 134,750
> Report number: 20211217-040713
> Result: Epoch 191/391
> loss: 4.0083 - abstract_content_output_loss: 2.5549 - detail_content_output_loss: 0.5907 - detail_title_output_loss: 0.5087
> abstract_content_output_accuracy: 0.3979 - detail_content_output_accuracy: 0.8948 - detail_title_output_accuracy: 0.9052
> val_loss: 21.1669 - val_abstract_content_output_loss: 5.7790 - val_detail_content_output_loss: 7.3622 - val_detail_title_output_loss: 7.6707
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0391 - val_detail_title_output_accuracy: 0.0312

### v3.9.11.5 (<- v3.9.11.4) Increase dropout rate
> Training samples: 932, validating samples: 127
> Parameters: 134,750
> Report number:
> Result: Epoch 391/391
> loss: 3.2079 - abstract_content_output_loss: 1.9605 - detail_content_output_loss: 0.4350 - detail_title_output_loss: 0.4449
> abstract_content_output_accuracy: 0.5104 - detail_content_output_accuracy: 0.8875 - detail_title_output_accuracy: 0.9021
> val_loss: 25.9485 - val_abstract_content_output_loss: 7.1653 - val_detail_content_output_loss: 9.2670 - val_detail_title_output_loss: 9.1492
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0078 - val_detail_title_output_accuracy: 0.0312

### v3.9.11.5 Use pretrained embedding layer v1.01
> Training samples: 1030, validating samples: 127
> Parameters: 134,750
> Report number: 20211219-104847
> Result: Epoch 379/391
> loss: 2.1189 - abstract_content_output_loss: 1.3917 - detail_content_output_loss: 0.1503 - detail_title_output_loss: 0.1453
> abstract_content_output_accuracy: 0.6259 - detail_content_output_accuracy: 0.9651 - detail_title_output_accuracy: 0.9651
> val_loss: 33.4204 - val_abstract_content_output_loss: 9.1510 - val_detail_content_output_loss: 11.8838 - val_detail_title_output_loss: 11.9543
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0078 - val_detail_title_output_accuracy: 0.0078

### v3.9.11.5 Use pretrained embedding layer v1.01; Drastically increase dropout rate.
> Training samples: 1030, validating samples: 127
> Parameters: 134,750
> Report number: 20211220-080045
> Result: Epoch 320/391
> loss: 15.1833 - abstract_content_output_loss: 5.0355 - detail_content_output_loss: 5.0465 - detail_title_output_loss: 5.0419
> abstract_content_output_accuracy: 0.0175 - detail_content_output_accuracy: 0.0221 - detail_title_output_accuracy: 0.0165 - val_loss: 18.2022
> val_abstract_content_output_loss: 6.0885 - val_detail_content_output_loss: 6.0812 - val_detail_title_output_loss: 5.9770
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0078

### v3.9.11.3 Train and apply pretrained embedding layer v1.02
> Training samples: 1037, validating samples: 127
> Parameters: 136,116 (135,667)
> Report number: 20211221-040907
> Result: Epoch 203/391
> loss: 3.0999 - abstract_content_output_loss: 2.1309 - detail_content_output_loss: 0.2657 - detail_title_output_loss: 0.2672
> abstract_content_output_accuracy: 0.5028 - detail_content_output_accuracy: 0.9596 - detail_title_output_accuracy: 0.9550
> val_loss: 24.6659 - val_abstract_content_output_loss: 6.7953 - val_detail_content_output_loss: 8.8399 - val_detail_title_output_loss: 8.5939
> val_abstract_content_output_accuracy: 0.0078 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0078

### v3.9.11.3 Train and apply pretrained embedding layer v01x02u01; Drastically increase droput rate
> Training samples: 1037, validating samples: 127
> Parameters: 136,116 (135,667)
> Report number: 20211221-152752
> Result: Epoch 65/391
> loss: 10.4068 - abstract_content_output_loss: 4.5768 - detail_content_output_loss: 2.7264 - detail_title_output_loss: 2.6398
> abstract_content_output_accuracy: 0.0643 - detail_content_output_accuracy: 0.4540 - detail_title_output_accuracy: 0.4706
> val_loss: 17.3236 - val_abstract_content_output_loss: 5.3053 - val_detail_content_output_loss: 5.7285 - val_detail_title_output_loss: 5.8268
> val_abstract_content_output_accuracy: 0.0234 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0156

### v3.10.0.0 (<- v3.9.11.3, keep branching structure, keep decoder) Change the way information flows, from axis along sequence to the depth of conv cells
## learning_rate -> 1e-3; optimizer -> RMSprop; activation -> swish; CharEmbeddingV01x02; Dropout rate -> small values;
> Training samples: 1037, validating samples: 127
> Parameters: 54,004
> Report number: 20211223-041057
> Result: Epoch 73/391
> loss: 13.8289 - abstract_content_output_loss: 4.7160 - detail_content_output_loss: 4.5316 - detail_title_output_loss: 4.5289
> abstract_content_output_accuracy: 0.0257 - detail_content_output_accuracy: 0.0432 - detail_title_output_accuracy: 0.0368
> val_loss: 16.7997 - val_abstract_content_output_loss: 5.5040 - val_detail_content_output_loss: 5.6397 - val_detail_title_output_loss: 5.6030
> val_abstract_content_output_accuracy: 0.0547 - val_detail_content_output_accuracy: 0.0078 - val_detail_title_output_accuracy: 0.0312
> Result: Epoch 106/391
> loss: 12.3911 - abstract_content_output_loss: 4.3429 - detail_content_output_loss: 3.9907 - detail_title_output_loss: 3.9716
> abstract_content_output_accuracy: 0.0478 - detail_content_output_accuracy: 0.0699 - detail_title_output_accuracy: 0.0735
> val_loss: 16.3013 - val_abstract_content_output_loss: 5.3606 - val_detail_content_output_loss: 5.4317 - val_detail_title_output_loss: 5.4223
> val_abstract_content_output_accuracy: 0.0859 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0312
> Result: Epoch 132/391
> loss: 11.4397 - abstract_content_output_loss: 4.0479 - detail_content_output_loss: 3.6557 - detail_title_output_loss: 3.6222
> abstract_content_output_accuracy: 0.0671 - detail_content_output_accuracy: 0.1048 - detail_title_output_accuracy: 0.1222
> val_loss: 15.9839 - val_abstract_content_output_loss: 5.2511 - val_detail_content_output_loss: 5.3295 - val_detail_title_output_loss: 5.2888
> val_abstract_content_output_accuracy: 0.0547 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0391
> Result: Epoch 348/391
> loss: 7.5468 - abstract_content_output_loss: 2.9948 - detail_content_output_loss: 2.1511 - detail_title_output_loss: 2.1343
> abstract_content_output_accuracy: 0.2298 - detail_content_output_accuracy: 0.4017 - detail_title_output_accuracy: 0.4145
> val_loss: 18.9223 - val_abstract_content_output_loss: 5.4820 - val_detail_content_output_loss: 6.4678 - val_detail_title_output_loss: 6.7055
> val_abstract_content_output_accuracy: 0.0234 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0234

### v3.10.0.0 (<- v3.10.0.0) Use pretrained embedding layer v1.04 and increase dropout rate
> Training samples: 1037, validating samples: 127
> Parameters: 53,789 (53,555)
> Report number: 20211225-151553
> Result: Epoch 55/391
> loss: 13.2032 - abstract_content_output_loss: 4.5886 - detail_content_output_loss: 4.2292 - detail_title_output_loss: 4.2127
> abstract_content_output_accuracy: 0.0358 - detail_content_output_accuracy: 0.0570 - detail_title_output_accuracy: 0.0634
> val_loss: 16.7573 - val_abstract_content_output_loss: 5.3956 - val_detail_content_output_loss: 5.6007 - val_detail_title_output_loss: 5.5885
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0000e+00

### v3.10.0.1 (<- v3.10.0.0) Restructure decoder block: ouput title dense connects to dense 2, abtract content output connect to dense 5; Use BatchNormalization; Increase dropout rate.
> Training samples: 1037, validating samples: 127
> Parameters: 35,612 (35,102)
> Report number: 20211226-001122, 20211226-013738
> Result: Epoch 370/391
> loss: 8.9879 - abstract_content_output_loss: 3.5247 - detail_content_output_loss: 2.5901 - detail_title_output_loss: 2.5929
> abstract_content_output_accuracy: 0.1608 - detail_content_output_accuracy: 0.3667 - detail_title_output_accuracy: 0.3483
> val_loss: 18.8845 - val_abstract_content_output_loss: 5.8704 - val_detail_content_output_loss: 6.4669 - val_detail_title_output_loss: 6.2671
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0234 - val_detail_title_output_accuracy: 0.0078

### v3.10.0.1 Use pretrained embedding layer v2.01
> Training samples: 1037, validating samples: 127
> Parameters: 36,351 (35,158)
> Report number: 20211228-114900
> Result: Epoch 501/501
> loss: 10.5716 - abstract_content_output_loss: 3.8189 - detail_content_output_loss: 3.1956 - detail_title_output_loss: 3.3222
> abstract_content_output_accuracy: 0.1112 - detail_content_output_accuracy: 0.2068 - detail_title_output_accuracy: 0.2040
> val_loss: 17.2139 - val_abstract_content_output_loss: 5.4767 - val_detail_content_output_loss: 5.7979 - val_detail_title_output_loss: 5.7046
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0312
> Remarks: Very slow convergence (probably because of high drop out rate); Frequently reach the val_loss ~15 which is the best record; Accuracy once reached ~7%

### v3.10.0.2 (<-v3.10.0.1) Increase conv cell number; Decrease dense decoder cell number; Dropout rate -> decrease
> Training samples: 1037, validating samples: 127
> Parameters: 35,933 (34,768)
> Report number: 20211228-155824
> Result: Epoch 501/501
> loss: 10.0313 - abstract_content_output_loss: 3.8440 - detail_content_output_loss: 2.9958 - detail_title_output_loss: 2.9818
> abstract_content_output_accuracy: 0.1186 - detail_content_output_accuracy: 0.2417 - detail_title_output_accuracy: 0.2546
> val_loss: 16.7158 - val_abstract_content_output_loss: 5.3345 - val_detail_content_output_loss: 5.8113 - val_detail_title_output_loss: 5.3600
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0156

### v3.11.0.0 (<-v3.10.0.0) Increase conv cell number; Decrease dense decoder cell number; Dropout rate -> decrease; Restructure output block.
> Best performance ever, val_loss steadily stay at ~15 many epochs, val_loss gets new record of ~14
> Training samples: 1138, validating samples: 127
> Parameters: 50,195 (48,914)
> Report number: 20211230-145501
> Result: Epoch 701/701
> loss: 8.3666 - abstract_content_output_loss: 2.3540 - detail_content_output_loss: 2.3180 - detail_title_output_loss: 3.5376
> abstract_content_output_accuracy: 0.3863 - detail_content_output_accuracy: 0.3906 - detail_title_output_accuracy: 0.1710
> val_loss: 21.7750 - val_abstract_content_output_loss: 7.8568 - val_detail_content_output_loss: 7.8754 - val_detail_title_output_loss: 5.8858
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0312

### v3.12.0.0 (<-v3.11.0.0) Restructuure conv blocks -> change the way applying pooling layers
> Training samples: 1153, validating samples: 127
> Parameters: 151,473 (149,357)
> Report number: 20220101-052233
> Result: Epoch 701/701
> loss: 7.9766 - abstract_content_output_loss: 2.2153 - detail_content_output_loss: 2.3535 - detail_title_output_loss: 3.0345
> abstract_content_output_accuracy: 0.3931 - detail_content_output_accuracy: 0.3660 - detail_title_output_accuracy: 0.1916
> val_loss: 35.0385 - val_abstract_content_output_loss: 12.8472 - val_detail_content_output_loss: 13.6964 - val_detail_title_output_loss: 8.1217
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0312

### v3.13.0.0 (<-v3.12.0.0) Restructuure conv blocks
## Training performace is very good
> Training samples: 1153, validating samples: 127
> Parameters: 141,513 (139,505)
> Report number: 20220101-110454
> Result: Epoch 190/701
> loss: 3.4945 - abstract_content_output_loss: 1.0579 - detail_content_output_loss: 0.9635 - detail_title_output_loss: 1.3156
> abstract_content_output_accuracy: 0.7352 - detail_content_output_accuracy: 0.7878 - detail_title_output_accuracy: 0.6612
> val_loss: 22.3971 - val_abstract_content_output_loss: 7.5569 - val_detail_content_output_loss: 7.4498 - val_detail_title_output_loss: 7.2329
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0000e+00

### v3.14.0.0 (<-v3.13.0.0)
## val_loss reached 14; val_acc was steady around 0.0469, reaech new record value of 0.0938 two times.
## Training performace is very good
> Training samples: 1153, validating samples: 127
> Parameters: 75,272 (73,274)
> Report number: 20220101-171407
> Result: Epoch 153/701
> loss: 8.3734 - abstract_content_output_loss: 2.6445 - detail_content_output_loss: 2.5101 - detail_title_output_loss: 2.8854
> abstract_content_output_accuracy: 0.2780 - detail_content_output_accuracy: 0.3133 - detail_title_output_accuracy: 0.2442
> val_loss: 20.8168 - val_abstract_content_output_loss: 7.0585 - val_detail_content_output_loss: 7.0228 - val_detail_title_output_loss: 6.4018
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0000e+00

### v3.14.0.0 (More data)
## Performance on validation is worse than training with less data. One possible hypothesis is that the model complexity became unmatched with the complexity of data.
## There was stagnation at loss 4.6 and accuracy 0.6
> Training samples: 1215, validating samples: 127
> Parameters: 75,272 (73,274)
> Report number: 20220102-051617
> Result: Epoch 701/701
> loss: 4.5750 - abstract_content_output_loss: 1.3075 - detail_content_output_loss: 1.0335 - detail_title_output_loss: 1.6925
> abstract_content_output_accuracy: 0.6036 - detail_content_output_accuracy: 0.6891 - detail_title_output_accuracy: 0.4918
> val_loss: 39.0792 - val_abstract_content_output_loss: 12.3784 - val_detail_content_output_loss: 14.6292 - val_detail_title_output_loss: 11.5302
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0000e+00

### v3.14.1.0 (<-v3.14.0.0) Resstructure ouput block; Inscrease cell number of output block layers
> Training samples: 1215, validating samples: 127
> Parameters: 97,978 (95,828)
> Report number: 20220102-071050
> Result: Epoch 701/701
> loss: 2.8791 - abstract_content_output_loss: 0.7425 - detail_content_output_loss: 0.3765 - detail_title_output_loss: 1.1658
> abstract_content_output_accuracy: 0.7714 - detail_content_output_accuracy: 0.8816 - detail_title_output_accuracy: 0.6497
> val_loss: 40.0217 - val_abstract_content_output_loss: 13.2383 - val_detail_content_output_loss: 15.3074 - val_detail_title_output_loss: 10.8815
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0469

### v3.14.2.0 (<-v3.14.1.0) Use LayerNormalization instead of BatchNormalization
> Training samples: 1215, validating samples: 127
> Parameters: 97,808 (95,920)
> Report number: 20220102-085000
> Result: Epoch 701/701
> loss: 8.2994 - abstract_content_output_loss: 2.5914 - detail_content_output_loss: 2.4158 - detail_title_output_loss: 2.9383
> abstract_content_output_accuracy: 0.3076 - detail_content_output_accuracy: 0.3495 - detail_title_output_accuracy: 0.2163
> val_loss: 20.9619 - val_abstract_content_output_loss: 7.2880 - val_detail_content_output_loss: 7.1596 - val_detail_title_output_loss: 6.1601
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0156

### v3.15.0.0 (<-v3.14.2.0) Significantly shink the network both conv encoder block and dense decoder block; Restructure output block -> content_concatenated, title_concatenated connect to (dense_2, dense_3), abstract_content connect to dense_3_norm
> Training samples: 1335, validating samples: 127
> Parameters: 38,574 (36,686)
> Report number: 20220103-104648; 20220103-125713;
> Result: Epoch 701/701
> loss: 8.7783 - abstract_content_output_loss: 2.6072 - detail_content_output_loss: 2.2931 - detail_title_output_loss: 3.4884
> abstract_content_output_accuracy: 0.3043 - detail_content_output_accuracy: 0.3638 - detail_title_output_accuracy: 0.1272
> val_loss: 21.5940 - val_abstract_content_output_loss: 7.5203 - val_detail_content_output_loss: 7.7662 - val_detail_title_output_loss: 5.9176
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0000e+00
> Result: Epoch 701/701 (re01)
> loss: 8.5114 - abstract_content_output_loss: 2.4704 - detail_content_output_loss: 2.1020 - detail_title_output_loss: 3.5147
> abstract_content_output_accuracy: 0.3638 - detail_content_output_accuracy: 0.4375 - detail_title_output_accuracy: 0.1287
> val_loss: 22.3742 - val_abstract_content_output_loss: 7.7371 - val_detail_content_output_loss: 8.0809 - val_detail_title_output_loss: 6.1318
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0312

### v3.15.1.0 (<-v3.15.0.0) Seperate content output branch and title output branch
## Achievement on val_acc -> new record of 0.0938 (epoch number: 285/701, 320/701, 377/701), stayed at values 0.0781, 0.0625, 0.0469 most of epochs and was steady; val_loss way staying at 15 in long span.
> Training samples: 1335, validating samples: 127
> Parameters: 51,702 (49,814)
> Report number: 20220103-144239
> Result: Epoch 701/701
> loss: 9.5415 - abstract_content_output_loss: 3.0957 - detail_content_output_loss: 2.7893 - detail_title_output_loss: 3.2135
> abstract_content_output_accuracy: 0.1935 - detail_content_output_accuracy: 0.2567 - detail_title_output_accuracy: 0.1920
> val_loss: 19.5669 - val_abstract_content_output_loss: 6.4673 - val_detail_content_output_loss: 6.8927 - val_detail_title_output_loss: 5.7631
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0000e+00

### v3.16.0.0 (<-v3.15.1.0) Use new training of Embedding Layer v04; Remove abstract_content_output;
## (NEW DATASET LABEL)
> Training samples: 1334, validating samples: 127
> Parameters: 52,765 (50,877)
> Report number: 20220106-110030
> Result: Epoch 701/701
> loss: 7.0542 - detail_content_output_loss: 3.3994 - detail_title_output_loss: 3.3887 - detail_content_output_accuracy: 0.1972
> detail_title_output_accuracy: 0.1808 - val_loss: 10.7354 - val_detail_content_output_loss: 5.4413 - val_detail_title_output_loss: 5.0285
> val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0156

### v3.17.0.0 (v3.16.0.0): Separate into three output blocks: detail_content_output, detail_title_output, abstract_content_output. Increase both dense size and dropout rate; Use LayerNormalization after every dense layer;
## (NEW DATASET LABEL) (Embedding Layer v04x02u03)
## (BEST PERFORMANCE) -> Loss stayed steadily at 13 one third training time; acc reached 0.0938;
> Training samples: 1334, validating samples: 127
> Parameters: 91,607 (89,719)
> Report number: 20220107-035647
> Result: Epoch 107/701
> loss: 12.4123 - abstract_content_output_loss: 4.2589 - detail_content_output_loss: 4.0130 - detail_title_output_loss: 3.9854
> abstract_content_output_accuracy: 0.0766 - detail_content_output_accuracy: 0.1101 - detail_title_output_accuracy: 0.1109
> val_loss: 13.9069 - val_abstract_content_output_loss: 4.6239 - val_detail_content_output_loss: 4.6035 - val_detail_title_output_loss: 4.5243
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 233/701
> loss: 11.4570 - abstract_content_output_loss: 3.9839 - detail_content_output_loss: 3.6455 - detail_title_output_loss: 3.6101
> abstract_content_output_accuracy: 0.1243 - detail_content_output_accuracy: 0.1882 - detail_title_output_accuracy: 0.1845
> val_loss: 13.0866 - val_abstract_content_output_loss: 4.3589 - val_detail_content_output_loss: 4.2454 - val_detail_title_output_loss: 4.2658
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0312
> Result: Epoch 492/701
> loss: 10.7900 - abstract_content_output_loss: 3.6864 - detail_content_output_loss: 3.4056 - detail_title_output_loss: 3.4502
> abstract_content_output_accuracy: 0.1615 - detail_content_output_accuracy: 0.2173 - detail_title_output_accuracy: 0.2225
> val_loss: 13.5534 - val_abstract_content_output_loss: 4.4609 - val_detail_content_output_loss: 4.5237 - val_detail_title_output_loss: 4.3217
> val_abstract_content_output_accuracy: 0.0938 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0312
> Result: Epoch 653/701
> Loss: 10.6675 - abstract_content_output_loss: 3.6517 - detail_content_output_loss: 3.3641 - detail_title_output_loss: 3.4017
> abstract_content_output_accuracy: 0.1592 - detail_content_output_accuracy: 0.2344 - detail_title_output_accuracy: 0.2359
> val_loss: 13.6804 - val_abstract_content_output_loss: 4.4892 - val_detail_content_output_loss: 4.5033 - val_detail_title_output_loss: 4.4384
> val_abstract_content_output_accuracy: 0.0938 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0469
> Result: Epoch 701/701
> loss: 10.5365 - abstract_content_output_loss: 3.6394 - detail_content_output_loss: 3.2963 - detail_title_output_loss: 3.3495
> abstract_content_output_accuracy: 0.1615 - detail_content_output_accuracy: 0.2485 - detail_title_output_accuracy: 0.2366
> val_loss: 13.6277 - val_abstract_content_output_loss: 4.4516 - val_detail_content_output_loss: 4.5003 - val_detail_title_output_loss: 4.4244
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0469

### v3.17.1.0 (v3.17.0.0): Increase both dense size and dropout rate; Especially make abstract_content_output block bigger than others.
> Training samples: 1334, validating samples: 127
> Parameters: 118,579 (116,691)
> Report number: 20220107-081755
> Result: Epoch 347/701
> loss: 11.5332 - abstract_content_output_loss: 4.0138 - detail_content_output_loss: 3.6506 - detail_title_output_loss: 3.6721
> abstract_content_output_accuracy: 0.0908 - detail_content_output_accuracy: 0.1905 - detail_title_output_accuracy: 0.1749
> val_loss: 13.6164 - val_abstract_content_output_loss: 4.4274 - val_detail_content_output_loss: 4.6000 - val_detail_title_output_loss: 4.3923
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 504/701
> loss: 11.1457 - abstract_content_output_loss: 3.8894 - detail_content_output_loss: 3.5093 - detail_title_output_loss: 3.5334
> abstract_content_output_accuracy: 0.1250 - detail_content_output_accuracy: 0.2173 - detail_title_output_accuracy: 0.2083
> val_loss: 13.4382 - val_abstract_content_output_loss: 4.4293 - val_detail_content_output_loss: 4.4910 - val_detail_title_output_loss: 4.3044
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.1406
> Result: Epoch 588/701
> loss: 10.9798 - abstract_content_output_loss: 3.8080 - detail_content_output_loss: 3.4692 - detail_title_output_loss: 3.4815
> abstract_content_output_accuracy: 0.1168 - detail_content_output_accuracy: 0.2225 - detail_title_output_accuracy: 0.2225
> val_loss: 13.3989 - val_abstract_content_output_loss: 4.3581 - val_detail_content_output_loss: 4.4960 - val_detail_title_output_loss: 4.3237
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.1406
> Result: Epoch 666/701
> loss: 10.9538 - abstract_content_output_loss: 3.7642 - detail_content_output_loss: 3.4934 - detail_title_output_loss: 3.4701
> abstract_content_output_accuracy: 0.1473 - detail_content_output_accuracy: 0.2307 - detail_title_output_accuracy: 0.2292
> val_loss: 13.5087 - val_abstract_content_output_loss: 4.3989 - val_detail_content_output_loss: 4.5919 - val_detail_title_output_loss: 4.2915
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.1406
> Result: Epoch 686/701
> loss: 10.8511 - abstract_content_output_loss: 3.7456 - detail_content_output_loss: 3.3974 - detail_title_output_loss: 3.4809
> abstract_content_output_accuracy: 0.1577 - detail_content_output_accuracy: 0.2210 - detail_title_output_accuracy: 0.2225
> val_loss: 13.3624 - val_abstract_content_output_loss: 4.3032 - val_detail_content_output_loss: 4.5304 - val_detail_title_output_loss: 4.3015
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.1406
> Result: Epoch 688/701
> loss: 10.8571 - abstract_content_output_loss: 3.7596 - detail_content_output_loss: 3.4096 - detail_title_output_loss: 3.4607
> abstract_content_output_accuracy: 0.1421 - detail_content_output_accuracy: 0.2217 - detail_title_output_accuracy: 0.2098
> val_loss: 13.4803 - val_abstract_content_output_loss: 4.3455 - val_detail_content_output_loss: 4.5635 - val_detail_title_output_loss: 4.3442
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.1406
> Result: Epoch 701/701
> loss: 10.8175 - abstract_content_output_loss: 3.7512 - detail_content_output_loss: 3.3836 - detail_title_output_loss: 3.4561
> abstract_content_output_accuracy: 0.1473 - detail_content_output_accuracy: 0.2381 - detail_title_output_accuracy: 0.2039
> val_loss: 13.5101 - val_abstract_content_output_loss: 4.3661 - val_detail_content_output_loss: 4.5595 - val_detail_title_output_loss: 4.3578
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0938

### v3.17.1.0 (Retrain): Apply Embedding Layer v04x04u12, Remove root node from truth distribution of article node
> Training samples: 1334, validating samples: 127
> Parameters: 118,579 (116,691)
> Report number: 20220110-032245
> Result: Epoch 701/701
> loss: 11.0102 - abstract_content_output_loss: 3.7773 - detail_content_output_loss: 3.4883 - detail_title_output_loss: 3.4983
> abstract_content_output_accuracy: 0.1295 - detail_content_output_accuracy: 0.2254 - detail_title_output_accuracy: 0.1853
> val_loss: 14.4837 - val_abstract_content_output_loss: 4.8909 - val_detail_content_output_loss: 4.7403 - val_detail_title_output_loss: 4.6058
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0938

### v3.17.1.0 (Retrain): Apply Embedding Layer v04x04u12, Reverting 'Remove root node from truth distribution of article node'
## Compare to the former, assign probability to root node gives benefit to abstract acc but disadvantage to both detail acc and loss.
> Training samples: 1334, validating samples: 127
> Parameters: 118,579 (116,691)
> Report number: 20220110-050204
> Result: Epoch 701/701
> loss: 11.1163 - abstract_content_output_loss: 3.8144 - detail_content_output_loss: 3.4777 - detail_title_output_loss: 3.5504
> abstract_content_output_accuracy: 0.1421 - detail_content_output_accuracy: 0.2344 - detail_title_output_accuracy: 0.2068
> val_loss: 15.3626 - val_abstract_content_output_loss: 4.9507 - val_detail_content_output_loss: 5.1356 - val_detail_title_output_loss: 5.0025
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0469

### v3.17.2.0 (v3.17.0.0): v04x04u12, Increase cell number for dense (encoder) block and conv block; Add 1 more layer at block 2; REmove 1 layer at block 3,4;
## Best accuracy: 17%
> Training samples: 1334, validating samples: 127
> Parameters: 148,267 (146,379)
> Report number: 20220110-160103
> Result: Epoch 327/701
> loss: 12.8797 - abstract_content_output_loss: 4.3818 - detail_content_output_loss: 4.1653 - detail_title_output_loss: 4.1727
> abstract_content_output_accuracy: 0.0603 - detail_content_output_accuracy: 0.1116 - detail_title_output_accuracy: 0.1019
> val_loss: 14.0360 - val_abstract_content_output_loss: 4.7083 - val_detail_content_output_loss: 4.5595 - val_detail_title_output_loss: 4.6080
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.1094 - val_detail_title_output_accuracy: 0.0312
> Result: Epoch 409/701
> loss: 12.5693 - abstract_content_output_loss: 4.2719 - detail_content_output_loss: 4.0774 - detail_title_output_loss: 4.0606
> abstract_content_output_accuracy: 0.0662 - detail_content_output_accuracy: 0.1004 - detail_title_output_accuracy: 0.1094
> val_loss: 14.1778 - val_abstract_content_output_loss: 4.8173 - val_detail_content_output_loss: 4.5666 - val_detail_title_output_loss: 4.6344
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.1250 - val_detail_title_output_accuracy: 0.0469
> Result: Epoch 414/701
> loss: 12.5980 - abstract_content_output_loss: 4.2856 - detail_content_output_loss: 4.0602 - detail_title_output_loss: 4.0923
> abstract_content_output_accuracy: 0.0841 - detail_content_output_accuracy: 0.1213 - detail_title_output_accuracy: 0.1213
> val_loss: 13.7435 - val_abstract_content_output_loss: 4.6155 - val_detail_content_output_loss: 4.4827 - val_detail_title_output_loss: 4.4853
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.1406 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 414/701
> loss: 12.6219 - abstract_content_output_loss: 4.2944 - detail_content_output_loss: 4.0849 - detail_title_output_loss: 4.0834
> abstract_content_output_accuracy: 0.0722 - detail_content_output_accuracy: 0.1086 - detail_title_output_accuracy: 0.0997
> val_loss: 13.9543 - val_abstract_content_output_loss: 4.6482 - val_detail_content_output_loss: 4.5434 - val_detail_title_output_loss: 4.6034
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.1406 - val_detail_title_output_accuracy: 0.0781
> Result: Epoch 414/701
> loss: 12.6004 - abstract_content_output_loss: 4.2959 - detail_content_output_loss: 4.0581 - detail_title_output_loss: 4.0872
> abstract_content_output_accuracy: 0.0588 - detail_content_output_accuracy: 0.1220 - detail_title_output_accuracy: 0.1012
> val_loss: 13.7670 - val_abstract_content_output_loss: 4.6095 - val_detail_content_output_loss: 4.4878 - val_detail_title_output_loss: 4.5105
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.1406 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 617/701
> loss: 12.3025 - abstract_content_output_loss: 4.1965 - detail_content_output_loss: 3.9414 - detail_title_output_loss: 3.9924
> abstract_content_output_accuracy: 0.0789 - detail_content_output_accuracy: 0.1406 - detail_title_output_accuracy: 0.1190
> val_loss: 13.6989 - val_abstract_content_output_loss: 4.6022 - val_detail_content_output_loss: 4.4839 - val_detail_title_output_loss: 4.44
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.1719 - val_detail_title_output_accuracy: 0.0469
> Result: Epoch 652/701
> loss: 12.1828 - abstract_content_output_loss: 4.1791 - detail_content_output_loss: 3.8980 - detail_title_output_loss: 3.9312
> abstract_content_output_accuracy: 0.0938 - detail_content_output_accuracy: 0.1443 - detail_title_output_accuracy: 0.1481
> val_loss: 14.1242 - val_abstract_content_output_loss: 4.7506 - val_detail_content_output_loss: 4.6170 - val_detail_title_output_loss: 4.5824
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.1562 - val_detail_title_output_accuracy: 0.0469
> Result: Epoch 701/701
> loss: 12.1659 - abstract_content_output_loss: 4.1778 - detail_content_output_loss: 3.8775 - detail_title_output_loss: 3.9324
> abstract_content_output_accuracy: 0.0796 - detail_content_output_accuracy: 0.1518 - detail_title_output_accuracy: 0.1280
> val_loss: 13.9532 - val_abstract_content_output_loss: 4.6617 - val_detail_content_output_loss: 4.5844 - val_detail_title_output_loss: 4.5286
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.0625

### v3.17.2.0 (More data, Embedding v04x04u16)
> Training samples: 1408, validating samples: 127
> Parameters: 148,267 (146,379)
> Report number: 20220112-155457, 20220113-005905
> Result: Epoch 701/701
> loss: 12.3148 - abstract_content_output_loss: 4.2343 - detail_content_output_loss: 3.9442 - detail_title_output_loss: 3.9561
> abstract_content_output_accuracy: 0.0746 - detail_content_output_accuracy: 0.1214 - detail_title_output_accuracy: 0.1300
> val_loss: 14.1705 - val_abstract_content_output_loss: 4.7572 - val_detail_content_output_loss: 4.6122 - val_detail_title_output_loss: 4.6216
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0781
> Result: Epoch 1022/1024
> loss: 11.6282 - abstract_content_output_loss: 3.9972 - detail_content_output_loss: 3.6908 - detail_title_output_loss: 3.7296
> abstract_content_output_accuracy: 0.0930 - detail_content_output_accuracy: 0.1534 - detail_title_output_accuracy: 0.1619
> val_loss: 14.4900 - val_abstract_content_output_loss: 4.8234 - val_detail_content_output_loss: 4.7749 - val_detail_title_output_loss: 4.6816
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.1094 - val_detail_title_output_accuracy: 0.0781

### v3.17.3.0 (Embedding v05x05u08)
> Training samples: 1408, validating samples: 127
> Parameters: 147,587 (145,995)
> Report number: 20220115-083205, 20220115-095125, 20220115-112244
> Result: Epoch 701/701
> loss: 12.6979 - abstract_content_output_loss: 4.3487 - detail_content_output_loss: 4.0811 - detail_title_output_loss: 4.0899
> abstract_content_output_accuracy: 0.0547 - detail_content_output_accuracy: 0.1136 - detail_title_output_accuracy: 0.1143
> val_loss: 15.2268 - val_abstract_content_output_loss: 5.0927 - val_detail_content_output_loss: 5.0945 - val_detail_title_output_loss: 4.8615
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0781
> Result: Epoch 701/701
> loss: 11.9414 - abstract_content_output_loss: 4.1112 - detail_content_output_loss: 3.7926 - detail_title_output_loss: 3.8372
> abstract_content_output_accuracy: 0.0930 - detail_content_output_accuracy: 0.1584 - detail_title_output_accuracy: 0.1357
> val_loss: 15.0436 - val_abstract_content_output_loss: 5.0642 - val_detail_content_output_loss: 4.9959 - val_detail_title_output_loss: 4.7834
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0469
> Result: Epoch 701/701
> loss: 11.1335 - abstract_content_output_loss: 3.8620 - detail_content_output_loss: 3.5086 - detail_title_output_loss: 3.6407
> abstract_content_output_accuracy: 0.1236 - detail_content_output_accuracy: 0.2067 - detail_title_output_accuracy: 0.1847
> val_loss: 16.0133 - val_abstract_content_output_loss: 5.4452 - val_detail_content_output_loss: 5.4633 - val_detail_title_output_loss: 4.9827
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0312

### v3.17.4.0 (<- v3.17.2.0): Use one_hot instead of char embedding.
## Good performance on val_loss but it seemed that val_acc did not follow the val_loss
> Training samples: 1408, validating samples: 127
> Parameters: 155,595
> Report number: 20220115-140044/
> Result: Epoch 701/701
> loss: 11.9401 - abstract_content_output_loss: 4.0513 - detail_content_output_loss: 3.8444 - detail_title_output_loss: 3.8825
> abstract_content_output_accuracy: 0.0760 - detail_content_output_accuracy: 0.1257 - detail_title_output_accuracy: 0.1186
> val_loss: 13.6964 - val_abstract_content_output_loss: 4.3853 - val_detail_content_output_loss: 4.6112 - val_detail_title_output_loss: 4.5378
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0625

### v3.17.5.0 (<- v3.17.2.0): Char Embedding v41117x06u03
> Training samples: 1408, validating samples: 127
> Parameters: 150,551 (147,339)
> Report number: 20220116-114121
> Result: Epoch 501/501
> loss: 11.9637 - abstract_content_output_loss: 4.0751 - detail_content_output_loss: 3.8207 - detail_title_output_loss: 3.8562
> abstract_content_output_accuracy: 0.0859 - detail_content_output_accuracy: 0.1257 - detail_title_output_accuracy: 0.1271
> val_loss: 14.0232 - val_abstract_content_output_loss: 4.5521 - val_detail_content_output_loss: 4.7134 - val_detail_title_output_loss: 4.5456
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 501/501 (learning_rate = 1e-4, RMSprop)
> loss: 11.5820 - abstract_content_output_loss: 3.9705 - detail_content_output_loss: 3.7359 - detail_title_output_loss: 3.7742
> abstract_content_output_accuracy: 0.1016 - detail_content_output_accuracy: 0.1484 - detail_title_output_accuracy: 0.1491
> val_loss: 13.6461 - val_abstract_content_output_loss: 4.5123 - val_detail_content_output_loss: 4.5956 - val_detail_title_output_loss: 4.4369
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0938

### v3.17.6.0 (<- v3.17.2.0): Char Embedding v4117x07u04
> Training samples: 1408, validating samples: 127
> Parameters: 148,831 (146,379)
> Report number: 20220117-100921
> Result: Epoch 701/701
> loss: 11.9525 - abstract_content_output_loss: 4.0953 - detail_content_output_loss: 3.7470 - detail_title_output_loss: 3.8262
> abstract_content_output_accuracy: 0.0675 - detail_content_output_accuracy: 0.1371 - detail_title_output_accuracy: 0.1371
> val_loss: 14.1227 - val_abstract_content_output_loss: 4.6983 - val_detail_content_output_loss: 4.6650 - val_detail_title_output_loss: 4.4760
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 101/101
> loss: 11.5271 - abstract_content_output_loss: 3.9380 - detail_content_output_loss: 3.6378 - detail_title_output_loss: 3.7433
> abstract_content_output_accuracy: 0.0909 - detail_content_output_accuracy: 0.1541 - detail_title_output_accuracy: 0.1435
> val_loss: 14.0144 - val_abstract_content_output_loss: 4.6890 - val_detail_content_output_loss: 4.6415 - val_detail_title_output_loss: 4.4761
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 701/701 (learning_rate = 5e-4)
> loss: 11.2550 - abstract_content_output_loss: 3.8826 - detail_content_output_loss: 3.5134 - detail_title_output_loss: 3.6544
> abstract_content_output_accuracy: 0.0994 - detail_content_output_accuracy: 0.1626 - detail_title_output_accuracy: 0.1697
> val_loss: 14.3106 - val_abstract_content_output_loss: 4.7796 - val_detail_content_output_loss: 4.7440 - val_detail_title_output_loss: 4.5828
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.1250
> Result: Epoch 701/701 (RMSprop, learning_rate = 2e-4)
> loss: 10.9496 - abstract_content_output_loss: 3.7585 - detail_content_output_loss: 3.4289 - detail_title_output_loss: 3.6219
> abstract_content_output_accuracy: 0.1286 - detail_content_output_accuracy: 0.2088 - detail_title_output_accuracy: 0.1740
> val_loss: 14.5775 - val_abstract_content_output_loss: 4.9671 - val_detail_content_output_loss: 4.8680 - val_detail_title_output_loss: 4.6023
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 701/701 (Nadam, learning_rate = 2e-4)
> loss: 10.4943 - abstract_content_output_loss: 3.5758 - detail_content_output_loss: 3.2991 - detail_title_output_loss: 3.4684
> abstract_content_output_accuracy: 0.1506 - detail_content_output_accuracy: 0.2251 - detail_title_output_accuracy: 0.1832
> val_loss: 15.6093 - val_abstract_content_output_loss: 5.1108 - val_detail_content_output_loss: 5.3554 - val_detail_title_output_loss: 4.9921
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0625

### v3.13.1.0 (<-v3.13.0.0) change embedding to v4117x07u04
## Bad acc performance
> Training samples: 1408, validating samples: 127
> Parameters: 144,491 (141,919)
> Report number: 20220118-081742
> Result: Epoch 175/701
> loss: 8.6631 - abstract_content_output_loss: 2.7316 - detail_content_output_loss: 2.6938 - detail_title_output_loss: 3.1300
> abstract_content_output_accuracy: 0.4276 - detail_content_output_accuracy: 0.4545 - detail_title_output_accuracy: 0.2969
> val_loss: 15.7007 - val_abstract_content_output_loss: 5.3220 - val_detail_content_output_loss: 5.3158 - val_detail_title_output_loss: 4.9550
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0156

### v3.10.1.0 (<- v3.10.0.1) change embedding to v4117x07u04
> Training samples: 1408, validating samples: 127
> Parameters: 38,999 (36,271)
> Report number: (...)
> Result: Epoch 701/701
> loss: 11.6457 - abstract_content_output_loss: 4.0434 - detail_content_output_loss: 3.6943 - detail_title_output_loss: 3.7331
> abstract_content_output_accuracy: 0.1072 - detail_content_output_accuracy: 0.1896 - detail_title_output_accuracy: 0.1811
> val_loss: 14.3007 - val_abstract_content_output_loss: 4.6431 - val_detail_content_output_loss: 4.7367 - val_detail_title_output_loss: 4.7461
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0625

### v3.18.0.0 (<- v3.10.1.0, v3.17.6.0): embedding v4117x07u04, conv from v3.10.1.0 <- conv activations, dense from 3.17.6.0 <- dropout rates halve
## Best performance
> Training samples: 1408, validating samples: 127
> Parameters: 58,952 (56,500)
> Report number: 
> Result: Epoch 45/701 << Very Fast >>
> loss: 13.7700 - abstract_content_output_loss: 4.6045 - detail_content_output_loss: 4.5397 - detail_title_output_loss: 4.5463
> abstract_content_output_accuracy: 0.0405 - detail_content_output_accuracy: 0.0447 - detail_title_output_accuracy: 0.0455
> val_loss: 13.9516 - val_abstract_content_output_loss: 4.6621 - val_detail_content_output_loss: 4.6079 - val_detail_title_output_loss: 4.6020
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 372/701
> loss: 11.5799 - abstract_content_output_loss: 3.9288 - detail_content_output_loss: 3.7332 - detail_title_output_loss: 3.7640
> abstract_content_output_accuracy: 0.1108 - detail_content_output_accuracy: 0.1641 - detail_title_output_accuracy: 0.1612
> val_loss: 13.3917 - val_abstract_content_output_loss: 4.3975 - val_detail_content_output_loss: 4.3752 - val_detail_title_output_loss: 4.4648
> val_abstract_content_output_accuracy: 0.1094 - val_detail_content_output_accuracy: 0.1406 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 420/701
> loss: 11.4588 - abstract_content_output_loss: 3.8707 - detail_content_output_loss: 3.7045 - detail_title_output_loss: 3.7267
> abstract_content_output_accuracy: 0.1172 - detail_content_output_accuracy: 0.1619 - detail_title_output_accuracy: 0.1754 - val_loss: 13.5343
> val_abstract_content_output_loss: 4.3922 - val_detail_content_output_loss: 4.5512 - val_detail_title_output_loss: 4.4343
> val_abstract_content_output_accuracy: 0.1250 - val_detail_content_output_accuracy: 0.1094 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 701/701
> loss: 11.0391 - abstract_content_output_loss: 3.7389 - detail_content_output_loss: 3.5232 - detail_title_output_loss: 3.6136
> abstract_content_output_accuracy: 0.1399 - detail_content_output_accuracy: 0.1939 - detail_title_output_accuracy: 0.1974
> val_loss: 14.2197 - val_abstract_content_output_loss: 4.6230 - val_detail_content_output_loss: 4.7506 - val_detail_title_output_loss: 4.6829
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.1094

### v3.18.1.0 (<- v3.18.0.0): embedding v4117x08u09, increase dropout rates
> Training samples: 1408, validating samples: 127
> Parameters: 58,952 (56,500)
> Report number: 20220119-122825
> Result: Epoch 88/701
> loss: 13.5583 - abstract_content_output_loss: 4.5427 - detail_content_output_loss: 4.4594 - detail_title_output_loss: 4.4701
> abstract_content_output_accuracy: 0.0476 - detail_content_output_accuracy: 0.0547 - detail_title_output_accuracy: 0.0561
> val_loss: 13.9896 - val_abstract_content_output_loss: 4.6619 - val_detail_content_output_loss: 4.6496 - val_detail_title_output_loss: 4.5916
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0469
> Result: Epoch 701/701 (...)
> Result: Epoch 101/101
> 11.6666 - abstract_content_output_loss: 3.9610 - detail_content_output_loss: 3.6854 - detail_title_output_loss: 3.8341
> abstract_content_output_accuracy: 0.1115 - detail_content_output_accuracy: 0.1719 - detail_title_output_accuracy: 0.1584
> val_loss: 13.8144 - val_abstract_content_output_loss: 4.6195 - val_detail_content_output_loss: 4.6081 - val_detail_title_output_loss: 4.4009
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 101/101
> loss: 10.9567 - abstract_content_output_loss: 3.6935 - detail_content_output_loss: 3.4982 - detail_title_output_loss: 3.6114
> abstract_content_output_accuracy: 0.1307 - detail_content_output_accuracy: 0.2053 - detail_title_output_accuracy: 0.1790
> val_loss: 15.0712 - val_abstract_content_output_loss: 5.0443 - val_detail_content_output_loss: 5.1478 - val_detail_title_output_loss: 4.7254
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0781

### v3.19.0.0 (<- v3.18.0.0): embedding v4117x09u03, remove pooling layer at block 1
> Training samples: 1408, validating samples: 127
> Parameters: 58,952 (56,500)
> Report number: 20220120-142736
> Result: Epoch 3/701
> loss: 18.5907 - abstract_content_output_loss: 5.5752 - detail_content_output_loss: 5.6361 - detail_title_output_loss: 5.5769
> abstract_content_output_accuracy: 0.0178 - detail_content_output_accuracy: 0.0220 - detail_title_output_accuracy: 0.0135
> val_loss: 17.8771 - val_abstract_content_output_loss: 5.4646 - val_detail_content_output_loss: 5.4853 - val_detail_title_output_loss: 5.3794
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0000e+00
> Result: Epoch 52/701 (val_loss ~ 13)
> loss: 13.7553 - abstract_content_output_loss: 4.6067 - detail_content_output_loss: 4.5328 - detail_title_output_loss: 4.5442
> abstract_content_output_accuracy: 0.0369 - detail_content_output_accuracy: 0.0526 - detail_title_output_accuracy: 0.0511
> val_loss: 13.9629 - val_abstract_content_output_loss: 4.6518 - val_detail_content_output_loss: 4.6690 - val_detail_title_output_loss: 4.5703
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0781
> Result: Epoch 125/701
> loss: 13.0218 - abstract_content_output_loss: 4.4015 - detail_content_output_loss: 4.2633 - detail_title_output_loss: 4.2564
> abstract_content_output_accuracy: 0.0391 - detail_content_output_accuracy: 0.0753 - detail_title_output_accuracy: 0.0781
> val_loss: 13.9317 - val_abstract_content_output_loss: 4.7032 - val_detail_content_output_loss: 4.6053 - val_detail_title_output_loss: 4.5229
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 362/701
> loss: 12.3129 - abstract_content_output_loss: 4.1788 - detail_content_output_loss: 3.9610 - detail_title_output_loss: 4.0232
> abstract_content_output_accuracy: 0.0739 - detail_content_output_accuracy: 0.1229 - detail_title_output_accuracy: 0.1222
> val_loss: 13.8091 - val_abstract_content_output_loss: 4.6047 - val_detail_content_output_loss: 4.6049 - val_detail_title_output_loss: 4.4499
> val_abstract_content_output_accuracy: 0.1250 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 551/701
> loss: 11.9553 - abstract_content_output_loss: 4.0697 - detail_content_output_loss: 3.8002 - detail_title_output_loss: 3.9184
> abstract_content_output_accuracy: 0.0888 - detail_content_output_accuracy: 0.1477 - detail_title_output_accuracy: 0.1271
> val_loss: 14.0732 - val_abstract_content_output_loss: 4.6669 - val_detail_content_output_loss: 4.6229 - val_detail_title_output_loss: 4.6161
> val_abstract_content_output_accuracy: 0.0156 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 612/701
> loss: 11.8722 - abstract_content_output_loss: 4.0142 - detail_content_output_loss: 3.8116 - detail_title_output_loss: 3.8774
> abstract_content_output_accuracy: 0.0945 - detail_content_output_accuracy: 0.1463 - detail_title_output_accuracy: 0.1392
> val_loss: 14.3498 - val_abstract_content_output_loss: 4.7551 - val_detail_content_output_loss: 4.8077 - val_detail_title_output_loss: 4.6178
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 656/701
> loss: 11.8837 - abstract_content_output_loss: 4.0262 - detail_content_output_loss: 3.8028 - detail_title_output_loss: 3.8852
> abstract_content_output_accuracy: 0.0994 - detail_content_output_accuracy: 0.1399 - detail_title_output_accuracy: 0.1392
> val_loss: 14.3035 - val_abstract_content_output_loss: 4.7053 - val_detail_content_output_loss: 4.8160 - val_detail_title_output_loss: 4.6134
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.1875
> Result: Epoch 701/701
> loss: 11.7663 - abstract_content_output_loss: 3.9965 - detail_content_output_loss: 3.7581 - detail_title_output_loss: 3.8440
> abstract_content_output_accuracy: 0.1044 - detail_content_output_accuracy: 0.1626 - detail_title_output_accuracy: 0.1399
> val_loss: 14.2108 - val_abstract_content_output_loss: 4.7090 - val_detail_content_output_loss: 4.7337 - val_detail_title_output_loss: 4.6012
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0938

### v3.20.0.0 (<- v3.19.0.0): embedding v411x10u02; learning_rate = 5e-4;
> Training samples: 1408, validating samples: 127
> Parameters: 59,557 (56,812)
> Report number: 20220122-053311
> Result: Epoch 701/701
> loss: 11.8482 - abstract_content_output_loss: 4.0433 - detail_content_output_loss: 3.8229 - detail_title_output_loss: 3.8628
> abstract_content_output_accuracy: 0.1030 - detail_content_output_accuracy: 0.1499 - detail_title_output_accuracy: 0.1335
> val_loss: 14.2573 - val_abstract_content_output_loss: 4.8531 - val_detail_content_output_loss: 4.7225 - val_detail_title_output_loss: 4.5629
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0781

### v3.17.7.0 (<- v3.17.2.0): Char Embedding v411x10u02
> Training samples: 1408, validating samples: 127
> Parameters: 149,508 (146,763)
> Report number: 20220122-072952
> Result: Epoch 701/701
> loss: 11.9187 - abstract_content_output_loss: 4.0258 - detail_content_output_loss: 3.7681 - detail_title_output_loss: 3.8297
> abstract_content_output_accuracy: 0.0852 - detail_content_output_accuracy: 0.1278 - detail_title_output_accuracy: 0.1293
> val_loss: 15.1417 - val_abstract_content_output_loss: 4.9724 - val_detail_content_output_loss: 5.0384 - val_detail_title_output_loss: 4.8364
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0781

### v3.15.2.0 (<-v3.15.1.0): Char Embedding v411x10u02
> Training samples: 1408, validating samples: 127
> Parameters: 59,570 (56,825)
> Report number: 20220122-085724
> Result: Epoch 701/701
> loss: 10.7479 - abstract_content_output_loss: 3.5367 - detail_content_output_loss: 3.3099 - detail_title_output_loss: 3.5642
> abstract_content_output_accuracy: 0.1790 - detail_content_output_accuracy: 0.2116 - detail_title_output_accuracy: 0.1747
> val_loss: 18.4132 - val_abstract_content_output_loss: 6.1442 - val_detail_content_output_loss: 6.3855 - val_detail_title_output_loss: 5.5468
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0000e+00

### v3.10.2.0 (<- v3.10.1.0) change embedding to v411x10u02
> Training samples: 1408, validating samples: 127
> Parameters: 39,604 (36,583)
> Report number: 
> Result: Epoch 0/701

### v3.21.0.0 (<- v3.19.0.0): embedding v411x10u02; set tripe at block 1; Change some pooling layers to MaxPool1D
> Training samples: 1408, validating samples: 127
> Parameters: 53,307 (50,562)
> Report number: 20220122-135913
> Result: Epoch 60/701
> loss: 14.9969 - abstract_content_output_loss: 4.9978 - detail_content_output_loss: 4.9651 - detail_title_output_loss: 4.9789
> abstract_content_output_accuracy: 0.0256 - detail_content_output_accuracy: 0.0256 - detail_title_output_accuracy: 0.0227
> val_loss: 15.4127 - val_abstract_content_output_loss: 5.1746 - val_detail_content_output_loss: 5.1464 - val_detail_title_output_loss: 5.0356
> val_abstract_content_output_accuracy: 0.0938 - val_detail_content_output_accuracy: 0.0000e+00 - val_detail_title_output_accuracy: 0.0469
> Result: Epoch 701/701
> loss: 14.0501 - abstract_content_output_loss: 4.6935 - detail_content_output_loss: 4.5936 - detail_title_output_loss: 4.6569
> abstract_content_output_accuracy: 0.0348 - detail_content_output_accuracy: 0.0355 - detail_title_output_accuracy: 0.0440
> val_loss: 15.8568 - val_abstract_content_output_loss: 5.3898 - val_detail_content_output_loss: 5.2827 - val_detail_title_output_loss: 5.0788
> val_abstract_content_output_accuracy: 0.0000e+00 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.0156

### v3.22.0.0 (<- v3.21.0.0): embedding v5x10u03; changes in conv block; sharing detail_content_dense and detail_title_dense;
## Baseline 1
> Training samples: 1408, validating samples: 127
> Parameters: 112,237 (110,483) (5s 224ms/step)
> Report number: 20220123-045355, 20220123-060528
> Result: Epoch 68/701
> loss: 13.7446 - abstract_content_output_loss: 4.6052 - detail_content_output_loss: 4.5187 - detail_title_output_loss: 4.5189
> abstract_content_output_accuracy: 0.0312 - detail_content_output_accuracy: 0.0476 - detail_title_output_accuracy: 0.0476
> val_loss: 13.9431 - val_abstract_content_output_loss: 4.6981 - val_detail_content_output_loss: 4.6011 - val_detail_title_output_loss: 4.5417
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0156 - val_detail_title_output_accuracy: 0.0312
> Result: Epoch 238/701
> loss: 12.5669 - abstract_content_output_loss: 4.2446 - detail_content_output_loss: 4.0732 - detail_title_output_loss: 4.0880
> abstract_content_output_accuracy: 0.0859 - detail_content_output_accuracy: 0.0959 - detail_title_output_accuracy: 0.0916
> val_loss: 13.5791 - val_abstract_content_output_loss: 4.5973 - val_detail_content_output_loss: 4.4695 - val_detail_title_output_loss: 4.3515
> val_abstract_content_output_accuracy: 0.0938 - val_detail_content_output_accuracy: 0.0781 - val_detail_title_output_accuracy: 0.0781
> Result: Epoch 264/701
> loss: 12.5673 - abstract_content_output_loss: 4.2337 - detail_content_output_loss: 4.0725 - detail_title_output_loss: 4.0948
> abstract_content_output_accuracy: 0.0746 - detail_content_output_accuracy: 0.1072 - detail_title_output_accuracy: 0.0881
> val_loss: 13.9401 - val_abstract_content_output_loss: 4.6633 - val_detail_content_output_loss: 4.6171 - val_detail_title_output_loss: 4.4935
> val_abstract_content_output_accuracy: 0.0938 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 331/701
> loss: 12.4810 - abstract_content_output_loss: 4.2136 - detail_content_output_loss: 4.0242 - detail_title_output_loss: 4.0721
> abstract_content_output_accuracy: 0.0724 - detail_content_output_accuracy: 0.1172 - detail_title_output_accuracy: 0.1009
> val_loss: 13.8977 - val_abstract_content_output_loss: 4.6584 - val_detail_content_output_loss: 4.6006 - val_detail_title_output_loss: 4.4681
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.0781
> Result: Epoch 428/701
> loss: 12.2927 - abstract_content_output_loss: 4.1900 - detail_content_output_loss: 3.9457 - detail_title_output_loss: 3.9805
> abstract_content_output_accuracy: 0.0661 - detail_content_output_accuracy: 0.1058 - detail_title_output_accuracy: 0.1001
> val_loss: 14.4781 - val_abstract_content_output_loss: 4.8375 - val_detail_content_output_loss: 4.8065 - val_detail_title_output_loss: 4.6577
> val_abstract_content_output_accuracy: 0.1250 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 701/701
> loss: 12.0426 - abstract_content_output_loss: 4.0843 - detail_content_output_loss: 3.8561 - detail_title_output_loss: 3.9114
> abstract_content_output_accuracy: 0.0803 - detail_content_output_accuracy: 0.1364 - detail_title_output_accuracy: 0.1186
> val_loss: 14.3189 - val_abstract_content_output_loss: 4.7367 - val_detail_content_output_loss: 4.7106 - val_detail_title_output_loss: 4.6816
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 701/701 (r01)
> loss: 11.3047 - abstract_content_output_loss: 3.8516 - detail_content_output_loss: 3.5414 - detail_title_output_loss: 3.6237
> abstract_content_output_accuracy: 0.1257 - detail_content_output_accuracy: 0.1740 - detail_title_output_accuracy: 0.1697
> val_loss: 16.6815 - val_abstract_content_output_loss: 5.4402 - val_detail_content_output_loss: 5.5794 - val_detail_title_output_loss: 5.3747
> val_abstract_content_output_accuracy: 0.0312 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0625

### v3.23.0.0 (<- v3.22.0.0): use one_hot, batch_size -> 32;
## Baseline 2
> Training samples: 1408, validating samples: 127
> Parameters: 137,601 (367ms/step)
> Report number: 20220123-071751
> Result: Epoch 17/701
> loss: 14.9372 - abstract_content_output_loss: 4.9653 - detail_content_output_loss: 4.9560 - detail_title_output_loss: 4.9636
> abstract_content_output_accuracy: 0.0163 - detail_content_output_accuracy: 0.0277 - detail_title_output_accuracy: 0.0270
> val_loss: 14.6020 - val_abstract_content_output_loss: 4.8730 - val_detail_content_output_loss: 4.8395 - val_detail_title_output_loss: 4.8383
> val_abstract_content_output_accuracy: 0.1016 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 21/701
> loss: 14.5486 - abstract_content_output_loss: 4.8486 - detail_content_output_loss: 4.8161 - detail_title_output_loss: 4.8236
> abstract_content_output_accuracy: 0.0234 - detail_content_output_accuracy: 0.0312 - detail_title_output_accuracy: 0.0298
> val_loss: 14.3045 - val_abstract_content_output_loss: 4.7812 - val_detail_content_output_loss: 4.7249 - val_detail_title_output_loss: 4.7397
> val_abstract_content_output_accuracy: 0.0703 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.0547
> Result: Epoch 27/701
> loss: 14.1577 - abstract_content_output_loss: 4.7154 - detail_content_output_loss: 4.6794 - detail_title_output_loss: 4.6915
> abstract_content_output_accuracy: 0.0277 - detail_content_output_accuracy: 0.0391 - detail_title_output_accuracy: 0.0355
> val_loss: 13.9993 - val_abstract_content_output_loss: 4.6700 - val_detail_content_output_loss: 4.6214 - val_detail_title_output_loss: 4.6358
> val_abstract_content_output_accuracy: 0.0703 - val_detail_content_output_accuracy: 0.0781 - val_detail_title_output_accuracy: 0.0312
> Result: Epoch 117/701
> loss: 12.6990 - abstract_content_output_loss: 4.2600 - detail_content_output_loss: 4.1179 - detail_title_output_loss: 4.1754
> abstract_content_output_accuracy: 0.0497 - detail_content_output_accuracy: 0.0803 - detail_title_output_accuracy: 0.0866
> val_loss: 13.5463 - val_abstract_content_output_loss: 4.4764 - val_detail_content_output_loss: 4.4824 - val_detail_title_output_loss: 4.4396
> val_abstract_content_output_accuracy: 0.0859 - val_detail_content_output_accuracy: 0.0703 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 171/701
> loss: 12.4719 - abstract_content_output_loss: 4.2098 - detail_content_output_loss: 4.0110 - detail_title_output_loss: 4.0911
> abstract_content_output_accuracy: 0.0604 - detail_content_output_accuracy: 0.0803 - detail_title_output_accuracy: 0.0909
> val_loss: 13.5356 - val_abstract_content_output_loss: 4.4365 - val_detail_content_output_loss: 4.4787 - val_detail_title_output_loss: 4.4590
> val_abstract_content_output_accuracy: 0.1250 - val_detail_content_output_accuracy: 0.0938 - val_detail_title_output_accuracy: 0.1094
> Result: Epoch 207/701
> loss: 12.3358 - abstract_content_output_loss: 4.1920 - detail_content_output_loss: 3.9470 - detail_title_output_loss: 4.0283
> abstract_content_output_accuracy: 0.0611 - detail_content_output_accuracy: 0.0959 - detail_title_output_accuracy: 0.0966
> val_loss: 13.5366 - val_abstract_content_output_loss: 4.4629 - val_detail_content_output_loss: 4.4612 - val_detail_title_output_loss: 4.4429
> val_abstract_content_output_accuracy: 0.0938 - val_detail_content_output_accuracy: 0.0859 - val_detail_title_output_accuracy: 0.1406
> Result: Epoch 307/701
> loss: 11.9920 - abstract_content_output_loss: 4.0615 - detail_content_output_loss: 3.8186 - detail_title_output_loss: 3.9237
> abstract_content_output_accuracy: 0.0781 - detail_content_output_accuracy: 0.1207 - detail_title_output_accuracy: 0.1115
> val_loss: 13.0403 - val_abstract_content_output_loss: 4.2926 - val_detail_content_output_loss: 4.2384 - val_detail_title_output_loss: 4.3217
> val_abstract_content_output_accuracy: 0.0859 - val_detail_content_output_accuracy: 0.0781 - val_detail_title_output_accuracy: 0.0859
> Result: Epoch 396/701
> loss: 11.8805 - abstract_content_output_loss: 3.9839 - detail_content_output_loss: 3.7897 - detail_title_output_loss: 3.9176
> abstract_content_output_accuracy: 0.0952 - detail_content_output_accuracy: 0.1286 - detail_title_output_accuracy: 0.1080
> val_loss: 13.2473 - val_abstract_content_output_loss: 4.3490 - val_detail_content_output_loss: 4.3295 - val_detail_title_output_loss: 4.3788
> val_abstract_content_output_accuracy: 0.1328 - val_detail_content_output_accuracy: 0.0703 - val_detail_title_output_accuracy: 0.0859
> oss: 11.6934 - abstract_content_output_loss: 3.9192 - detail_content_output_loss: 3.7176 - detail_title_output_loss: 3.8830
> abstract_content_output_accuracy: 0.1044 - detail_content_output_accuracy: 0.1314 - detail_title_output_accuracy: 0.1257
> val_loss: 13.3566 - val_abstract_content_output_loss: 4.3863 - val_detail_content_output_loss: 4.3751 - val_detail_title_output_loss: 4.4227
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0547 - val_detail_title_output_accuracy: 0.0703

### v3.24.0.0 (<- v3.22.0.0): mix embedding and one hot; batch_size -> 32;
## Best performance record acc
> Training samples: 1408, validating samples: 127
> Parameters: 101,543 (99,789) (184ms/step)
> Report number: 20220123-104913
> Result: Epoch 33/701
> loss: 14.1899 - abstract_content_output_loss: 4.7239 - detail_content_output_loss: 4.6869 - detail_title_output_loss: 4.6953
> abstract_content_output_accuracy: 0.0327 - detail_content_output_accuracy: 0.0291 - detail_title_output_accuracy: 0.0355
> val_loss: 13.9729 - val_abstract_content_output_loss: 4.6513 - val_detail_content_output_loss: 4.6382 - val_detail_title_output_loss: 4.5986
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0234 - val_detail_title_output_accuracy: 0.0391
> Result: Epoch 172/701
> loss: 12.8904 - abstract_content_output_loss: 4.3565 - detail_content_output_loss: 4.1780 - detail_title_output_loss: 4.1883
> abstract_content_output_accuracy: 0.0653 - detail_content_output_accuracy: 0.0945 - detail_title_output_accuracy: 0.0788
> val_loss: 13.2054 - val_abstract_content_output_loss: 4.4049 - val_detail_content_output_loss: 4.3204 - val_detail_title_output_loss: 4.3119
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.1250 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 296/701
> loss: 12.5960 - abstract_content_output_loss: 4.2778 - detail_content_output_loss: 4.0378 - detail_title_output_loss: 4.1048
> abstract_content_output_accuracy: 0.0575 - detail_content_output_accuracy: 0.0838 - detail_title_output_accuracy: 0.0895
> val_loss: 13.2669 - val_abstract_content_output_loss: 4.4497 - val_detail_content_output_loss: 4.3289 - val_detail_title_output_loss: 4.3122
> val_abstract_content_output_accuracy: 0.1016 - val_detail_content_output_accuracy: 0.1484 - val_detail_title_output_accuracy: 0.1016
> Result: Epoch 574/701
> loss: 12.3505 - abstract_content_output_loss: 4.1636 - detail_content_output_loss: 3.9731 - detail_title_output_loss: 4.0294
> abstract_content_output_accuracy: 0.0810 - detail_content_output_accuracy: 0.1023 - detail_title_output_accuracy: 0.0994
> val_loss: 13.6390 - val_abstract_content_output_loss: 4.5639 - val_detail_content_output_loss: 4.4506 - val_detail_title_output_loss: 4.4406
> val_abstract_content_output_accuracy: 0.1328 - val_detail_content_output_accuracy: 0.0703 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 579/701
> loss: 12.2067 - abstract_content_output_loss: 4.1108 - detail_content_output_loss: 3.9166 - detail_title_output_loss: 3.9964
> abstract_content_output_accuracy: 0.0760 - detail_content_output_accuracy: 0.1065 - detail_title_output_accuracy: 0.1080
> val_loss: 13.6463 - val_abstract_content_output_loss: 4.5630 - val_detail_content_output_loss: 4.4630 - val_detail_title_output_loss: 4.4384
> val_abstract_content_output_accuracy: 0.1250 - val_detail_content_output_accuracy: 0.1250 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 580/701
> loss: 12.2831 - abstract_content_output_loss: 4.1461 - detail_content_output_loss: 3.9493 - detail_title_output_loss: 4.0050
> abstract_content_output_accuracy: 0.0859 - detail_content_output_accuracy: 0.1051 - detail_title_output_accuracy: 0.0994
> val_loss: 13.4231 - val_abstract_content_output_loss: 4.4585 - val_detail_content_output_loss: 4.3945 - val_detail_title_output_loss: 4.3873
> val_abstract_content_output_accuracy: 0.1016 - val_detail_content_output_accuracy: 0.1406 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 701/701
> loss: 12.1471 - abstract_content_output_loss: 4.0722 - detail_content_output_loss: 3.9085 - detail_title_output_loss: 3.9829
> abstract_content_output_accuracy: 0.0795 - detail_content_output_accuracy: 0.1087 - detail_title_output_accuracy: 0.0994
> val_loss: 13.4520 - val_abstract_content_output_loss: 4.4660 - val_detail_content_output_loss: 4.4323 - val_detail_title_output_loss: 4.3720
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0859 - val_detail_title_output_accuracy: 0.0781
### v3x24x00x00r01: batch_size -> 32; learning_rate -> 5e-4; RMSprop;
> Report number: 20220123-130251
> Result: Epoch 264/701
> loss: 11.3905 - abstract_content_output_loss: 3.8585 - detail_content_output_loss: 3.6401 - detail_title_output_loss: 3.7761
> abstract_content_output_accuracy: 0.1264 - detail_content_output_accuracy: 0.1506 - detail_title_output_accuracy: 0.1364
> val_loss: 13.8411 - val_abstract_content_output_loss: 4.5940 - val_detail_content_output_loss: 4.6247 - val_detail_title_output_loss: 4.5067
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 276/701
> loss: 11.4226 - abstract_content_output_loss: 3.8556 - detail_content_output_loss: 3.6551 - detail_title_output_loss: 3.7951
> abstract_content_output_accuracy: 0.1087 - detail_content_output_accuracy: 0.1591 - detail_title_output_accuracy: 0.1293
> val_loss: 13.8854 - val_abstract_content_output_loss: 4.5606 - val_detail_content_output_loss: 4.6621 - val_detail_title_output_loss: 4.5461
> val_abstract_content_output_accuracy: 0.0469 - val_detail_content_output_accuracy: 0.0781 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 278/701
> loss: 11.2992 - abstract_content_output_loss: 3.8292 - detail_content_output_loss: 3.6139 - detail_title_output_loss: 3.7397
> abstract_content_output_accuracy: 0.1257 - detail_content_output_accuracy: 0.1676 - detail_title_output_accuracy: 0.1520
> val_loss: 13.9379 - val_abstract_content_output_loss: 4.6076 - val_detail_content_output_loss: 4.6664 - val_detail_title_output_loss: 4.5475
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 448/701
> loss: 11.2959 - abstract_content_output_loss: 3.8366 - detail_content_output_loss: 3.5958 - detail_title_output_loss: 3.7454
> abstract_content_output_accuracy: 0.1151 - detail_content_output_accuracy: 0.1626 - detail_title_output_accuracy: 0.1357
> val_loss: 13.7813 - val_abstract_content_output_loss: 4.5759 - val_detail_content_output_loss: 4.5999 - val_detail_title_output_loss: 4.4876
> val_abstract_content_output_accuracy: 0.0938 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 571/701
> loss: 11.2213 - abstract_content_output_loss: 3.8178 - detail_content_output_loss: 3.5601 - detail_title_output_loss: 3.7262
> abstract_content_output_accuracy: 0.1264 - detail_content_output_accuracy: 0.1634 - detail_title_output_accuracy: 0.1413
> val_loss: 13.9175 - val_abstract_content_output_loss: 4.5540 - val_detail_content_output_loss: 4.7010 - val_detail_title_output_loss: 4.5453
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 607/701
> loss: 11.2434 - abstract_content_output_loss: 3.7941 - detail_content_output_loss: 3.5922 - detail_title_output_loss: 3.7395
> abstract_content_output_accuracy: 0.1236 - detail_content_output_accuracy: 0.1619 - detail_title_output_accuracy: 0.1406
> val_loss: 13.9313 - val_abstract_content_output_loss: 4.5954 - val_detail_content_output_loss: 4.6680 - val_detail_title_output_loss: 4.5507
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0312 - val_detail_title_output_accuracy: 0.1562
> Result: Epoch 701/701
> loss: 11.2779 - abstract_content_output_loss: 3.8255 - detail_content_output_loss: 3.5933 - detail_title_output_loss: 3.7417
> abstract_content_output_accuracy: 0.1250 - detail_content_output_accuracy: 0.1626 - detail_title_output_accuracy: 0.1463
> val_loss: 13.8465 - val_abstract_content_output_loss: 4.5730 - val_detail_content_output_loss: 4.6402 - val_detail_title_output_loss: 4.5160
> val_abstract_content_output_accuracy: 0.0781 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0938
### v3x24x00x00r01: batch_size -> 64; learning_rate -> 5e-4, Nadam;
> Report number: 20220124-085012
> Result: Epoch 701/701
> abstract_content_output_loss: 3.7098 - detail_content_output_loss: 3.3945 - detail_title_output_loss: 3.5748
> abstract_content_output_accuracy: 0.1314 - detail_content_output_accuracy: 0.1946 - detail_title_output_accuracy: 0.1825
> val_loss: 15.1288 - val_abstract_content_output_loss: 4.9601 - val_detail_content_output_loss: 5.0897 - val_detail_title_output_loss: 4.8959
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0469 - val_detail_title_output_accuracy: 0.0625
### v3x24x00x00r02: batch_size -> 43; learning_rate -> 5e-4, Nadam;
> Report number: 20220124-103144
> Result: Epoch 701/701
> loss: 10.7245 - abstract_content_output_loss: 3.6286 - detail_content_output_loss: 3.3983 - detail_title_output_loss: 3.4892
> abstract_content_output_accuracy: 0.1388 - detail_content_output_accuracy: 0.1875 - detail_title_output_accuracy: 0.1818
> val_loss: 13.7577 - val_abstract_content_output_loss: 4.4577 - val_detail_content_output_loss: 4.5243 - val_detail_title_output_loss: 4.5673
> val_abstract_content_output_accuracy: 0.0543 - val_detail_content_output_accuracy: 0.0620 - val_detail_title_output_accuracy: 0.0853
### v3x24x00x00r03: batch_size -> 32; learning_rate -> 5e-4, Nadam; fine_tuning -> 0.06;
> Report number: 20220124-134825
> Result: Epoch 55/701
> loss: 10.7423 - abstract_content_output_loss: 3.6698 - detail_content_output_loss: 3.4233 - detail_title_output_loss: 3.4451
> abstract_content_output_accuracy: 0.1428 - detail_content_output_accuracy: 0.2060 - detail_title_output_accuracy: 0.2109
> val_loss: 13.5144 - val_abstract_content_output_loss: 4.3867 - val_detail_content_output_loss: 4.4454 - val_detail_title_output_loss: 4.4777
> val_abstract_content_output_accuracy: 0.1641 - val_detail_content_output_accuracy: 0.0859 - val_detail_title_output_accuracy: 0.0703
> Result: Epoch 88/701
> loss: 10.6750 - abstract_content_output_loss: 3.6210 - detail_content_output_loss: 3.4253 - detail_title_output_loss: 3.4187
> abstract_content_output_accuracy: 0.1484 - detail_content_output_accuracy: 0.1868 - detail_title_output_accuracy: 0.1868
> val_loss: 13.4859 - val_abstract_content_output_loss: 4.3792 - val_detail_content_output_loss: 4.4319 - val_detail_title_output_loss: 4.4649
> val_abstract_content_output_accuracy: 0.1719 - val_detail_content_output_accuracy: 0.1016 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 125/701
> loss: 10.5443 - abstract_content_output_loss: 3.5777 - detail_content_output_loss: 3.3366 - detail_title_output_loss: 3.4153
> abstract_content_output_accuracy: 0.1591 - detail_content_output_accuracy: 0.2159 - detail_title_output_accuracy: 0.1911
> val_loss: 13.6600 - val_abstract_content_output_loss: 4.3669 - val_detail_content_output_loss: 4.5255 - val_detail_title_output_loss: 4.5530
> val_abstract_content_output_accuracy: 0.1719 - val_detail_content_output_accuracy: 0.0781 - val_detail_title_output_accuracy: 0.0703
> Result: Epoch 126/701
> loss: 10.6184 - abstract_content_output_loss: 3.6158 - detail_content_output_loss: 3.3706 - detail_title_output_loss: 3.4172
> abstract_content_output_accuracy: 0.1534 - detail_content_output_accuracy: 0.2102 - detail_title_output_accuracy: 0.2024
> val_loss: 13.5599 - val_abstract_content_output_loss: 4.3274 - val_detail_content_output_loss: 4.4923 - val_detail_title_output_loss: 4.5255
> val_abstract_content_output_accuracy: 0.1797 - val_detail_content_output_accuracy: 0.0781 - val_detail_title_output_accuracy: 0.0625
> Result: Epoch 682/701
> loss: 10.1271 - abstract_content_output_loss: 3.3899 - detail_content_output_loss: 3.1394 - detail_title_output_loss: 3.3629
> abstract_content_output_accuracy: 0.1825 - detail_content_output_accuracy: 0.2571 - detail_title_output_accuracy: 0.2188
> val_loss: 14.2012 - val_abstract_content_output_loss: 4.7001 - val_detail_content_output_loss: 4.6136 - val_detail_title_output_loss: 4.6535
> val_abstract_content_output_accuracy: 0.1562 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0938
> Result: Epoch 701/701
> loss: 10.1348 - abstract_content_output_loss: 3.3550 - detail_content_output_loss: 3.1667 - detail_title_output_loss: 3.3834
> abstract_content_output_accuracy: 0.1974 - detail_content_output_accuracy: 0.2479 - detail_title_output_accuracy: 0.2180
> val_loss: 14.6341 - val_abstract_content_output_loss: 4.9354 - val_detail_content_output_loss: 4.7469 - val_detail_title_output_loss: 4.7220
> val_abstract_content_output_accuracy: 0.1328 - val_detail_content_output_accuracy: 0.1016 - val_detail_title_output_accuracy: 0.1094
### v3x24x00x00r04: batch_size -> 64; learning_rate -> 5e-4, Nadam; fine_tuning -> 0;
> Report number: 20220124-160745
> Result: Epoch 701/701
> loss: 10.4158 - abstract_content_output_loss: 3.5057 - detail_content_output_loss: 3.2980 - detail_title_output_loss: 3.4149
> abstract_content_output_accuracy: 0.1548 - detail_content_output_accuracy: 0.2195 - detail_title_output_accuracy: 0.2209
> val_loss: 13.4738 - val_abstract_content_output_loss: 4.4827 - val_detail_content_output_loss: 4.1943 - val_detail_title_output_loss: 4.5997
> val_abstract_content_output_accuracy: 0.0625 - val_detail_content_output_accuracy: 0.0625 - val_detail_title_output_accuracy: 0.0469
### v3x24x00x00r05: batch_size -> 43; learning_rate -> 5e-4, Nadam; fine_tuning -> 0.06;
> Report number: 20220125-010546
> Result: Epoch 701/701
> loss: 10.4325 - abstract_content_output_loss: 3.5486 - detail_content_output_loss: 3.3459 - detail_title_output_loss: 3.3523
> abstract_content_output_accuracy: 0.1903 - detail_content_output_accuracy: 0.2255 - detail_title_output_accuracy: 0.2072
> val_loss: 13.2060 - val_abstract_content_output_loss: 4.2286 - val_detail_content_output_loss: 4.1961 - val_detail_title_output_loss: 4.5952
> val_abstract_content_output_accuracy: 0.0853 - val_detail_content_output_accuracy: 0.1240 - val_detail_title_output_accuracy: 0.0853
> Result: Epoch 41/701
> loss: 10.1677 - abstract_content_output_loss: 3.4627 - detail_content_output_loss: 3.2342 - detail_title_output_loss: 3.2746
> abstract_content_output_accuracy: 0.1875 - detail_content_output_accuracy: 0.2361 - detail_title_output_accuracy: 0.2396
> val_loss: 12.9827 - val_abstract_content_output_loss: 4.1370 - val_detail_content_output_loss: 4.1588 - val_detail_title_output_loss: 4.4911
> val_abstract_content_output_accuracy: 0.0930 - val_detail_content_output_accuracy: 0.1163 - val_detail_title_output_accuracy: 0.0465
> Result: Epoch 79/701
> loss: 10.0893 - abstract_content_output_loss: 3.4440 - detail_content_output_loss: 3.2031 - detail_title_output_loss: 3.2429
> abstract_content_output_accuracy: 0.1853 - detail_content_output_accuracy: 0.2375 - detail_title_output_accuracy: 0.2255
> val_loss: 12.9379 - val_abstract_content_output_loss: 4.0703 - val_detail_content_output_loss: 4.1516 - val_detail_title_output_loss: 4.5163
> val_abstract_content_output_accuracy: 0.0620 - val_detail_content_output_accuracy: 0.0853 - val_detail_title_output_accuracy: 0.0775
> Result: Epoch 93/701
> loss: 9.9839 - abstract_content_output_loss: 3.3878 - detail_content_output_loss: 3.1788 - detail_title_output_loss: 3.2152
> abstract_content_output_accuracy: 0.1994 - detail_content_output_accuracy: 0.2354 - detail_title_output_accuracy: 0.2340
> val_loss: 13.2855 - val_abstract_content_output_loss: 4.1869 - val_detail_content_output_loss: 4.2517 - val_detail_title_output_loss: 4.6448
> val_abstract_content_output_accuracy: 0.0853 - val_detail_content_output_accuracy: 0.0775 - val_detail_title_output_accuracy: 0.0388
> Result: Epoch 701/701
> loss: 9.8531 - abstract_content_output_loss: 3.2644 - detail_content_output_loss: 3.1065 - detail_title_output_loss: 3.2703
> abstract_content_output_accuracy: 0.2163 - detail_content_output_accuracy: 0.2685 - detail_title_output_accuracy: 0.2276
> val_loss: 13.9714 - val_abstract_content_output_loss: 4.6399 - val_detail_content_output_loss: 4.3793 - val_detail_title_output_loss: 4.7395
> val_abstract_content_output_accuracy: 0.0930 - val_detail_content_output_accuracy: 0.0853 - val_detail_title_output_accuracy: 0.0000e+00
### v3x24x00x00r06: batch_size -> 61; learning_rate -> 5e-4, Nadam; fine_tuning -> 0;
> Report number: 20220125-025057
> Result: Epoch 701/701
> loss: 10.2312 - abstract_content_output_loss: 3.4528 - detail_content_output_loss: 3.2384 - detail_title_output_loss: 3.3420
> abstract_content_output_accuracy: 0.1783 - detail_content_output_accuracy: 0.2343 - detail_title_output_accuracy: 0.2309
> val_loss: 14.5835 - val_abstract_content_output_loss: 4.8570 - val_detail_content_output_loss: 4.8354 - val_detail_title_output_loss: 4.6930
> val_abstract_content_output_accuracy: 0.0820 - val_detail_content_output_accuracy: 0.0765 - val_detail_title_output_accuracy: 0.0492
### v3x24x00x00r07: batch_size -> 43; learning_rate -> 5e-4, Nadam; fine_tuning -> 0;
> Report number:
> Result: Epoch 677/701
> loss: 10.1429 - abstract_content_output_loss: 3.4246 - detail_content_output_loss: 3.1589 - detail_title_output_loss: 3.3427
> abstract_content_output_accuracy: 0.1917 - detail_content_output_accuracy: 0.2431 - detail_title_output_accuracy: 0.2213
> val_loss: 14.4330 - val_abstract_content_output_loss: 4.6662 - val_detail_content_output_loss: 4.6167 - val_detail_title_output_loss: 4.9334
> val_abstract_content_output_accuracy: 0.1008 - val_detail_content_output_accuracy: 0.1085 - val_detail_title_output_accuracy: 0.0233
> Result: Epoch 701/701
> loss: 10.2850 - abstract_content_output_loss: 3.4537 - detail_content_output_loss: 3.2202 - detail_title_output_loss: 3.3941
> abstract_content_output_accuracy: 0.1790 - detail_content_output_accuracy: 0.2255 - detail_title_output_accuracy: 0.2220
> val_loss: 14.1777 - val_abstract_content_output_loss: 4.5505 - val_detail_content_output_loss: 4.5346 - val_detail_title_output_loss: 4.8758
> val_abstract_content_output_accuracy: 0.1163 - val_detail_content_output_accuracy: 0.1085 - val_detail_title_output_accuracy: 0.0388
### v3x24x00x00r08: batch_size -> 59; learning_rate -> 5e-4, Nadam; fine_tuning -> 0;
> Report number: 20220412-152907
> Result: Epoch 201/201
> loss: 10.2765 - abstract_content_output_loss: 3.5028 - detail_content_output_loss: 3.2540 - detail_title_output_loss: 3.3242
> abstract_content_output_accuracy: 0.1864 - detail_content_output_accuracy: 0.2366 - detail_title_output_accuracy: 0.2090
> val_loss: 13.9189 - val_abstract_content_output_loss: 4.3942 - val_detail_content_output_loss: 4.4796 - val_detail_title_output_loss: 4.8497
> val_abstract_content_output_accuracy: 0.0395 - val_detail_content_output_accuracy: 0.0621 - val_detail_title_output_accuracy: 0.0621
### v3x24x00x00r09: batch_size -> 43; learning_rate -> 5e-4, Nadam; fine_tuning -> 0.06;
> Report number: 20220413-045407
> Result: Epoch 501/501
> loss: 9.6580 - abstract_content_output_loss: 3.1884 - detail_content_output_loss: 3.0293 - detail_title_output_loss: 3.2347
> abstract_content_output_accuracy: 0.2304 - detail_content_output_accuracy: 0.2748 - detail_title_output_accuracy: 0.2431
> val_loss: 15.4234 - val_abstract_content_output_loss: 5.1774 - val_detail_content_output_loss: 5.0242 - val_detail_title_output_loss: 5.0158
> val_abstract_content_output_accuracy: 0.0155 - val_detail_content_output_accuracy: 0.0388 - val_detail_title_output_accuracy: 0.0930
### v3x24x00x00r10: batch_size -> 57; learning_rate -> 5e-4, Nadam; fine_tuning -> 0;
> Report number: 20220413-142511
> Result: Epoch 201/201
> loss: 10.1112 - abstract_content_output_loss: 3.4184 - detail_content_output_loss: 3.1731 - detail_title_output_loss: 3.3228
> abstract_content_output_accuracy: 0.2070 - detail_content_output_accuracy: 0.2344 - detail_title_output_accuracy: 0.2063
> val_loss: 14.8021 - val_abstract_content_output_loss: 4.7535 - val_detail_content_output_loss: 4.9315 - val_detail_title_output_loss: 4.9202
> val_abstract_content_output_accuracy: 0.0526 - val_detail_content_output_accuracy: 0.1111 - val_detail_title_output_accuracy: 0.0702
### v3x24x00x00r11: batch_size -> 29; learning_rate -> 5e-4, Nadam; fine_tuning -> 0.06;
> Report number: 20220414-024139
> Result: Epoch 201/201
> loss: 9.7535 - abstract_content_output_loss: 3.2551 - detail_content_output_loss: 3.0513 - detail_title_output_loss: 3.2203
> abstract_content_output_accuracy: 0.2252 - detail_content_output_accuracy: 0.2702 - detail_title_output_accuracy: 0.2315
> val_loss: 14.8119 - val_abstract_content_output_loss: 4.8130 - val_detail_content_output_loss: 4.7233 - val_detail_title_output_loss: 5.0493
> val_abstract_content_output_accuracy: 0.0483 - val_detail_content_output_accuracy: 0.0759 - val_detail_title_output_accuracy: 0.0276
### v3x24x00x00r12: batch_size -> 57; learning_rate -> 5e-4, Nadam; fine_tuning -> 0;
> Report number:
> Result: Epoch 201/201
> loss: 10.1341 - abstract_content_output_loss: 3.4622 - detail_content_output_loss: 3.1814 - detail_title_output_loss: 3.2897
> abstract_content_output_accuracy: 0.1811 - detail_content_output_accuracy: 0.2421 - detail_title_output_accuracy: 0.227
> val_loss: 14.1235 - val_abstract_content_output_loss: 4.4869 - val_detail_content_output_loss: 4.6163 - val_detail_title_output_loss: 4.8195
> val_abstract_content_output_accuracy: 0.0585 - val_detail_content_output_accuracy: 0.0409 - val_detail_title_output_accuracy: 0.0526
