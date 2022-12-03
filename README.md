# NESA: Neural Event Scheduling Assistant
NESA is a deep learning-based event scheduling assistant which can recommend most suitable times for new calendar events of each user. This repository provides the official implementation of NESA. Due to the dataset privacy issue, we instead provide the pre-processing code for your own google calendar data. Please refer to our paper, [Learning User Preferences and Understanding Calendar Contexts for Event Scheduling (Kim et al., CIKM 2018)](https://arxiv.org/abs/1809.01316), for more details of our model.

## Citation
```
@inproceedings{kim2018learning,
  title={Learning User Preferences and Understanding Calendar Contexts for Event Scheduling},
  author={Kim, Donghyeon and Lee, Jinhyuk and Choi, Donghee and Choi, Jaehoon and Kang, Jaewoo},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={337--346},
  year={2018},
  organization={ACM}
}
```

## Prerequisites
* [Python 3](https://www.python.org/downloads/)
* [PyTorch](http://pytorch.org/) 1.13.0+
* (Optional) NVIDIA GPU (memory size: 8GB or greater)
    * [CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/cudnn)
* A [Google](https://www.google.com) account
* git lfs

## Installation
Check if working directory is "nesa".

### Download word vector file and decompress it to your __home_dir/nlp__
- [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)

## Run NESA with the sample data
```
# Set PYTHONPATH environment variable
$ export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run
$ python3 test.py
```

## (Optional) Run NESA w/ your calendar data
* Important: Download client_secret.json to the project folder before running get_google_calendar_events.py
(See https://developers.google.com/google-apps/calendar/quickstart/python)
* Important: Modify CLIENT_SECRET_FILE value of get_google_calendar_events.py
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

## License
Apache License 2.0
