import argparse
import csv
from dateutil.parser import parse
import httplib2
import os
import string
import sys

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

from datetime import date, timedelta
from operator import itemgetter

# Based on https://developers.google.com/google-apps/calendar/quickstart/python
CLIENT_SECRET_FILE = 'client_secret.json'  # Use your secret file
calendarId = 'primary'  # Use calendar 'ID' unless primary
output_dir = './data'
MINUTE_NORM = 30
print_valid_events = False
allow_non_eng_users = True
allow_inactive_users = True

if not os.path.exists(
        os.path.join(
            os.path.join(os.path.expanduser('~'), '.credentials'),
            'calendar-python-quickstart.json')) \
        and not os.path.exists(CLIENT_SECRET_FILE):
    print(
        'Not found client_secret.json and credentials. See '
        'https://developers.google.com/google-apps/calendar/quickstart/python')
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()


# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/calendar-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/calendar.readonly'
APPLICATION_NAME = 'Google Calendar Events Fetching & Preprocessing'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'calendar-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else:  # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials


def filter_title(title):
    title_lc = title.lower()
    return title is None \
        or '' == title.strip() \
           or 'NULL' == title \
           or '(No title)' == title \
           or 'New Event' == title \
           or title.startswith('✔ ') \
           or title_lc.startswith('cancelled') \
           or title_lc.startswith('canceled') \
           or title.startswith('Call From ') \
           or title.startswith('Call To ') \
           or title.startswith('Missed Call From ') \
           or title.startswith('Flight to ') \
           or title.startswith('Stay at ') \
           or title.startswith('I entered http://') \
           or title.startswith('I exited http://')


def is_valid_duration(duration):
    # most likely auto-generated phone call events
    if duration.seconds % 60 > 0:
        return False
    # skip too long events: days or over 12 hrs
    if duration.days > 0 or duration.seconds > 3600 * 12:
        return False
    # skip length equal or less than 0 duration
    if duration.seconds // 60 <= 0 or duration.days < 0:
        return False
    return True


def dict_count(cnt_dict, cnt_key):
    num = cnt_dict.get(cnt_key)
    cnt_dict[cnt_key] = 1 if num is None else num + 1


# https://stackoverflow.com/a/14191915
def get_week_distance(dt1, dt2):
    monday1 = (dt1 - timedelta(days=dt1.weekday(), hours=dt1.hour,
                               minutes=dt1.minute, seconds=dt1.second))
    monday2 = (dt2 - timedelta(days=dt2.weekday(), hours=dt2.hour,
                               minutes=dt2.minute, seconds=dt2.second))
    return (monday2 - monday1).days // 7


def filter_user(events, title_idx, valid_week_evt_cnt_dict,
                min_num_events=100, active_avg_num_week_events=1.75,
                max_allow_non_eng_rate=0.02):
    def get_non_eng_rate(_events, _title_idx):
        _num_title_chars = 0
        _non_printable_count = 0
        for evt in _events:
            title = evt[_title_idx]
            _num_title_chars += len(title)
            for c in title:
                if c not in string.printable:
                    _non_printable_count += 1
        return _non_printable_count, _num_title_chars

    if not allow_non_eng_users:
        # non english rate > 0.02
        non_printable_count, num_title_chars = \
            get_non_eng_rate(events, title_idx)
        non_eng_rate = non_printable_count / num_title_chars
        if non_eng_rate > max_allow_non_eng_rate:
            print('Please run for English users: non_eng_rate=%.2f'
                  % non_eng_rate, '>', max_allow_non_eng_rate)
            return True

    # event num < 100 -> inactive or very new
    if len(events) < min_num_events:
        print('Please run for more active users: #events=%d'
              % len(events), '<', min_num_events)
        return True

    if not allow_inactive_users:
        # average week event num < 1.75 -> inactive
        avg_num_week_events = len(events) / len(valid_week_evt_cnt_dict)
        if avg_num_week_events < active_avg_num_week_events:
            print('Please run for more active users: avg_num_week_events=%.2f'
                  % avg_num_week_events, '<', active_avg_num_week_events)
            return True
    return False


def delete_invalid_chars_4_filename(file_name, invalid_chars):
    valid_file_name = ''
    for c in file_name:
        if c not in invalid_chars:
            valid_file_name += c
    return valid_file_name


def write_csv(_output_csv_path, events):
    with open(_output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        event_writer = csv.writer(csv_file, quotechar='"')
        for evt_features in events:
            event_writer.writerow(evt_features)


def main():
    """Shows basic usage of the Google Calendar API.

    Creates a Google Calendar API service object and outputs a list of the next
    10 events on the user's calendar.
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('calendar', 'v3', http=http)

    primary_calendar_id = None
    page_token = None
    while True:
        calendar_list = service.calendarList().\
            list(pageToken=page_token).execute()
        for calendar_list_entry in calendar_list['items']:
            # print(calendar_list_entry['summary'])
            if calendar_list_entry.get('primary') is True:
                primary_calendar_id = calendar_list_entry['id']
        page_token = calendar_list.get('nextPageToken')
        if not page_token:
            break

    print('Getting calendar events of', primary_calendar_id)
    # from datetime import datetime
    # now = datetime.utcnow().isoformat() + 'Z'  # UTC
    request = service.events().list(
        calendarId=calendarId,
        # timeMin=now,
        # maxResults=10,
        singleEvents=True,
        orderBy='startTime')

    num_events = 0
    events_total = list()
    valid_events = list()
    week_sequence_dict = dict()
    filtered_events_num = dict()
    valid_week_evt_cnt_dict = dict()

    while request is not None:
        response = request.execute()
        events = response.get('items', [])

        if not events:
            print('No upcoming events found.')

        events_total.extend(events)
        num_events += len(events)
        print(num_events)

        request = service.events().list_next(request, response)

    # sort by register time
    reg_sorted_events = list()
    for event in events_total:
        event['created_dt'] = parse(event['created'])
        reg_sorted_events.append(event)
    reg_sorted_events = sorted(reg_sorted_events, key=itemgetter('created_dt'))

    for event in reg_sorted_events:
        # filtering: invalid titles
        if event.get('summary') is None:
            dict_count(filtered_events_num, 'no title')
            continue
        elif filter_title(event['summary']):
            dict_count(filtered_events_num, 'invalid title')
            continue

        # filtering: skip all-day events
        if event['start'].get('dateTime') is None:
            dict_count(filtered_events_num, 'all-day')
            continue

        # filtering: year 0 is out of range
        created = event['created']
        if '0000' == created[:4] or '1900' == created[:4]:
            dict_count(filtered_events_num, 'year 0 or 1900')
            continue

        register_dt = parse(created)

        start = event['start'].get('dateTime', event['start'].get('date'))
        start_dt = parse(start)

        # filtering: a past
        if start_dt.toordinal() < register_dt.toordinal():
            dict_count(filtered_events_num, 'past')
            continue

        end = event['end'].get('dateTime', event['end'].get('date'))
        end_dt = parse(end)

        # filtering: phone call or too long duration
        duration = end_dt - start_dt
        if not is_valid_duration(duration):
            dict_count(filtered_events_num, 'invalid duration')
            continue

        start_iso_year, start_iso_week_num, _ = \
            date(start_dt.year, start_dt.month, start_dt.day).isocalendar()

        yw = start_iso_year * 100 + start_iso_week_num
        if yw in valid_week_evt_cnt_dict:
            valid_week_evt_cnt_dict[yw] += 1
        else:
            valid_week_evt_cnt_dict[yw] = 1

        week_register_sequence = week_sequence_dict.get(
            (start_iso_year, start_iso_week_num))
        if week_register_sequence is None:
            week_register_sequence = 0  # start with 0
        else:
            week_register_sequence += 1
        week_sequence_dict[(start_iso_year, start_iso_week_num)] = \
            week_register_sequence

        register_start_week_distance = get_week_distance(
            register_dt, start_dt)
        day_distance = \
            date(start_dt.year, start_dt.month, start_dt.day) - \
            date(register_dt.year, register_dt.month, register_dt.day)

        recurring_event_id = event.get('recurringEventId')
        is_recurrent = recurring_event_id is not None

        # y
        start_time_slot = \
            start_dt.minute // MINUTE_NORM \
            + start_dt.hour * int(60 / MINUTE_NORM) \
            + start_dt.weekday() * int((60 * 24) / MINUTE_NORM)

        # If you change the order of event features,
        # check itemgetter parameters below.
        evt_features = list()
        evt_features.append(primary_calendar_id)  # originally, user id
        # evt_features.append(event['iCalUID'])

        # title, duration, register time, start time
        evt_features.append(event['summary'])
        evt_features.append(duration.seconds // 60)  # minute
        evt_features.append(register_dt)
        evt_features.append(start_dt)

        # sort by year, week, register sequence in a week
        evt_features.append(start_iso_year)
        evt_features.append(start_iso_week_num)
        evt_features.append(week_register_sequence)

        # distance between register and start
        evt_features.append(register_start_week_distance)
        evt_features.append(day_distance.days)

        # is recurrent?
        evt_features.append(is_recurrent)

        # y
        evt_features.append(start_time_slot)

        valid_events.append(evt_features)

        if print_valid_events:
            print(evt_features)

    print('\n#events', num_events)
    print('#valid_events', len(valid_events))
    for fek in filtered_events_num:
        print('#filtered_events (%s)' % fek, filtered_events_num.get(fek))

    # MUST modify if you update orders of event features
    title_idx = 1
    start_year_idx = 5
    start_week_idx = 6
    reg_seq_idx = 7
    if filter_user(valid_events, title_idx, valid_week_evt_cnt_dict):
        return

    # write events to .csv
    # Unix, Windows
    invalid_chars = ['\0', '\\', '/', '*', '?', '"', '<', '>', '|']
    valid_file_name = delete_invalid_chars_4_filename(primary_calendar_id,
                                                      invalid_chars)
    output_file = output_dir + '/' + valid_file_name + '_events.csv'

    # sort by year, week, and register_sequence
    write_csv(
        output_file, sorted(valid_events, key=itemgetter(start_year_idx,
                                                         start_week_idx,
                                                         reg_seq_idx)))
    print('Saved', output_file)


if __name__ == '__main__':
    main()
