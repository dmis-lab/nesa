# NETS: Neural Event Time Scheduler Utilizing User Intention and Calendar Context
Donghyeon Kim and Jinhyuk Lee et al.

# Prerequisites
* [PyTorch](http://pytorch.org/) 0.4.0
* (Optional) NVIDIA GPU (memory size: 8GB or greater)
    * [CUDA](https://developer.nvidia.com/cuda-downloads) 9.1
    * [cuDNN](https://developer.nvidia.com/cudnn) 7
* A [Google](https://www.google.com) account

# Download sample data (Check if working directory is "nets")
```
$ wget -P data https://s3-us-west-1.amazonaws.com/ml-study0/nets/sample_data.csv
```

# Download word, character dictionaries
```
$ wget -P data https://s3-us-west-1.amazonaws.com/ml-study0/nets/preprocess_20180429_dict.pkl
```

# Download pretrained NETS model
```
$ wget -P data https://s3-us-west-1.amazonaws.com/ml-study0/nets/nets_gradclip_180501_5_1.pth
```

# Run NETS w/ sample data
```
# Set PYTHONPATH environment variable
$ export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run
$ python3 test.py
```

# Run NETS w/ your calendar data
* Important: Download client_secret.json to the project folder before running get_google_calendar_events.py
(See https://developers.google.com/google-apps/calendar/quickstart/python)
```
$ python3 get_google_calendar_events.py
```
* Check if <primary_calendar_id>_events.csv file is in __data__ directory.
* Event fields (12-column)
    * \[email_address, title, duration_minute, register_time, start_time, start_iso_year, start_iso_week, week_register_sequence, register_start_week_distance, register_start_day_distance, is_recurrent, start_time_slot\]
    * Sorted by year, week, and sequence in a week
    * Time slot range: 0 ~ 335
        * 30 minutes * 48 slots * 7 days = 1 week
    * Example: example@example.com,Cafe with J,60,2017-09-19 11:21:43,2017-09-23 10:00:00,2017,38,4,0,1,False,260
* Results of the model could be different for each dataset.
```
$ python3 test.py --input_path ./data/<primary_calendar_id>_events.csv
```
