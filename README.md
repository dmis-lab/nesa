# NETS: Neural Event Time Scheduler Utilizing User Intention and Calendar Context
Donghyeon Kim and Jinhyuk Lee et al.

# Main components
* nVidia GPU (memory size: 4GB or greater)
    * CUDA 8.0.61
    * cuDNN 6.0.21
* python 3.5.2
* pytorch 0.2.0.post3
* Google account

# Download sample data
```
$ wget -P data https://s3-us-west-1.amazonaws.com/ml-man/nets/sample_events.csv
```

# Download word, character dictionaries
```
$ wget -P data https://s3-us-west-1.amazonaws.com/ml-man/nets/nets_sm_w0_dict.pkl
```

# Download pretrained NETS model
```
$ wget -P data https://s3-us-west-1.amazonaws.com/ml-man/nets/conv_nets.pth
```

# Run NETS w/ sample data
```
# Set PYTHONPATH environment variable
$ export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run
$ python3 test.py
```

# Run NETS w/ your data
* Important: Download client_secret.json to the project folder before running get_google_calendar_events.py
(see https://developers.google.com/google-apps/calendar/quickstart/python)
```
$ python3 get_google_calendar_events.py
```
* Check if <primary_calendar_id>_events.txt file is in __data__ directory
* Event fields (11-column)
    * [email_address, title, duration_minute, register_time, start_time, start_iso_year, start_iso_week, week_register_sequence, register_start_week_distance, register_start_day_distance, start_time_slot]
    * Sorted by year, week and sequence in a week
    * Time slot range: 0 ~ 335
        * 30 minutes * 48 * 7 = 1 week
    * Example: example@mail.com,Cafe with J,60,2017-09-19 11:21:43,2017-09-23 10:00:00,2017,38,4,0,1,260

```
$ python3 test.py --test_path ./data/<primary_calendar_id>_events.txt
```

# Known Issues
* RuntimeError: the given numpy array has zero-sized dimensions. Zero-sized dimensions are not supported in PyTorch
    * Use pytorch version 0.2.0 instead of 0.3.0
        * [virtualenv](https://virtualenv.pypa.io/en/stable/) can be a solution
