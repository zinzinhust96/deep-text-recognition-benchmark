import re
import numpy as np

FILE_NAME = 'date.txt'

SUFFIX_DELI = ['/', '-', '.']

def extract_date(text):
    # PREPROCESS: replace all 'o' -> '0', ' ' -> ''
    text = text.replace('o', '0')
    text = text.replace('O', '0')
    text = text.replace(' ', '')
    # text = re.sub('[^0-9]', '', text)
    # print('preprocessed text: ', text)
    
    # leave out prefix first
    try:
        do_not_capture_prefix = re.search(r'(####|[A-Za-z])?[\.|\:]?([0-9a-zA-Z.\-\/\年月日 ]+)', text).group(2).strip()
        # print('do_not_capture_prefix: ', do_not_capture_prefix)
    except:
        return None


    return do_not_capture_prefix

def extract_dmy(raw_text):
    #print('raw text: ', raw_text)

    # return None if no date is detected
    if raw_text is None: return None

    # if raw_text contains characters, return None
    #if re.search(r'[^0-9|年|月|日|\.|\-|\/]', raw_text) is not None: return None

    # if first character does not make a year, return None
    # if not raw_text[0].isdigit() or (len(raw_text) > 1 and not raw_text[1].isdigit()): return None
    # if int(raw_text[:2]) > 30 or int(raw_text[:2]) < 19: return None

    # remove non numeric and non japanese date string
    # raw_text = re.sub('[^0-9|年|月|日|\.|\-|\/]', '', raw_text)

    #print('BEFORE EXTRACE DMY: ', raw_text)

    # extract japanese text
    jpn_date_no_deli = re.sub('[\.|\-|\/]', '', raw_text)
    capture_group = re.search(r'(19|2019|2[0-3]|202[0-3])年\ ?(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})月\ ?(0[1-9]|1[0-9]|2[0-9]|3[0-1]|[1-9])日', jpn_date_no_deli)
    if capture_group is None:
        capture_group = re.search(r'(19|2019|2[0-3]|202[0-3])年\ ?(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})月', jpn_date_no_deli)

    if capture_group is not None:
        year = int(capture_group.group(1))
        month = int(capture_group.group(2))
        day = int(capture_group.group(3)) if len(capture_group.groups()) == 3 is not None else None
        if year < 100:
            year+= 2000
        if year > 2023:
            return None
        return {'Y': year, 'M': month, 'D': day}

   # not japanese text
    groups = []
    group_lengths = []

    ########### NamTP, get last index of number
    tmp_id = 0
    for i in range(len(raw_text)):
        if raw_text[i].isdigit():
            tmp_id = i
    raw_text = raw_text[:tmp_id+1]
    #print('AFTER REMOVE NON NUMBER', raw_text)

    # count the deliminator
    deliminator = re.search(r'^.*?(\.|\-|\/)[^$]*$', raw_text)
    if deliminator is not None:
        deli_count = raw_text.count(deliminator.group(1))
    else:
        deli_count = 0


    
    # if deli_count > 2 --> model is likely to output more deliminator than expected
    if deli_count > 2:
        print('REMOVE ALL DELIMINATOR')
        # it is better to remove all deliminator in this case
        date_no_deli = re.sub('[\.|\-|\/]', '', raw_text)
        #print('DATE NO DELI', date_no_deli)

        # year + month
        capture_group_1 = re.search(r'(1[0-9]|2[0-3]|201[0-9]|202[0-3])(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})', date_no_deli)
        group_length_1 = capture_group_1.end() - capture_group_1.start() if capture_group_1 is not None else 0
        capture_group_2 = re.search(r'(201[0-9]|202[0-3]|1[0-9]|2[0-3])(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})', date_no_deli)
        group_length_2 = capture_group_2.end() - capture_group_2.start() if capture_group_2 is not None else 0
        
        # year + month + day
        capture_group_3 = re.search(r'(1[0-9]|2[0-3]|201[0-9]|202[0-3])(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})(0[1-9]|1[0-9]|2[0-9]|3[0-1]|[1-9])', date_no_deli)
        group_length_3 = capture_group_3.end() - capture_group_3.start() if capture_group_3 is not None else 0
        capture_group_4 = re.search(r'(201[0-9]|202[0-3]|1[0-9]|2[0-3])(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})(0[1-9]|1[0-9]|2[0-9]|3[0-1]|[1-9])', date_no_deli)
        group_length_4 = capture_group_4.end() - capture_group_4.start() if capture_group_4 is not None else 0

        # construct groups and groups length based on raw text having deliminator or not
        # if first_deli is not None:
        groups = [capture_group_1, capture_group_2, capture_group_3, capture_group_4]
        group_lengths = [group_length_1, group_length_2, group_length_3, group_length_4]
        # else:
            # groups = [capture_group_3, capture_group_4, capture_group_1, capture_group_2]
            # group_lengths = [group_length_3, group_length_4, group_length_1, group_length_2]

        #print('GROUP', groups, group_lengths)

        max_group_index = np.argmax(np.array(group_lengths))
        max_group_length = group_lengths[max_group_index]
        if max_group_length < 4:
            return None
        selected_capture_group = groups[max_group_index]

    else:
        print('KEEP ALL DELIMINATOR')
        # keep deliminator to easily separate day month year
        text_date = raw_text

        # year + deliminator + month + deliminator + day
        capture_group = re.search(r'(201[0-9]|202[0-3]|1[0-9]|2[0-3])[-\/\.]\ ?(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})[-\/\.]\ ?(0[1-9]|1[0-9]|2[0-9]|3[0-1]|[1-9])', text_date)
        groups.append(capture_group)
        group_length = capture_group.end() - capture_group.start() if capture_group is not None else 0
        group_lengths.append(group_length) 
        # year + deliminator + month
        capture_group = re.search(r'(201[0-9]|202[0-3])[-\/\.]\ ?(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})', text_date)
        groups.append(capture_group)
        group_length = capture_group.end() - capture_group.start() if capture_group is not None else 0
        group_lengths.append(group_length)
        # year + month + day
        capture_group = re.search(r'(201[0-9]|202[0-3]|1[0-9]|2[0-3])(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})(0[1-9]|1[0-9]|2[0-9]|3[0-1]|[1-9])', text_date)
        groups.append(capture_group)
        group_length = capture_group.end() - capture_group.start() if capture_group is not None else 0
        group_lengths.append(group_length)
        # year + month
        capture_group = re.search(r'(201[0-9]|202[0-3])(0[1-9]|1[0-2]|[1-9]|[A-Z]{3,})', text_date)
        groups.append(capture_group)
        group_length = capture_group.end() - capture_group.start() if capture_group is not None else 0
        group_lengths.append(group_length)

        # print(group_lengths)

        max_group_index = np.argmax(np.array(group_lengths))
        max_group_length = group_lengths[max_group_index]
        if max_group_length < 5:
            return None
        selected_capture_group = groups[max_group_index]

    if selected_capture_group:
        year = int(selected_capture_group.group(1))
        month = int(selected_capture_group.group(2))
        day = int(selected_capture_group.group(3)) if len(selected_capture_group.groups()) == 3 is not None else None
        if year < 100:
            year+= 2000
        if year > 2023:
            return None
        return {'Y': year, 'M': month, 'D': day}
    return None

def extract_dmy_from_text(text):
    date_extracted = extract_date(text)
    # print('date_extracted: ', date_extracted)
    date_dict = extract_dmy(date_extracted)
    return date_dict

if __name__== "__main__" :
    with open(FILE_NAME) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for txt in content:
        date_dict = extract_dmy_from_text(txt)
        print(txt, '\t-\t', date_dict, '\n')


