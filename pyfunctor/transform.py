def filter_func(dataset, bool_func):
    new_dicts = []
    for row in dataset:
        if bool_func(row):
            new_dicts.append(row)
    return new_dicts


def map_func(dataset, functor):
    values = []
    for row in dataset:
        values.append(functor(row))
    return values;

def reducebykey_func(dataset, aggregator, key_extractor = lambda row: row[0], value_extractor = lambda row: row[1]): 
    pairs = {}
    for row in dataset:
        key = key_extractor(row)
        value = value_extractor(row)
        if key in pairs:
            pairs[key] = aggregator(pairs[key], value)
        else:
            pairs[key] = value
    
    return pairs;

def reduce_func(dataset, aggregator) :
    result = dataset[0]
    for i in range(len(dataset)) : 
        if i != 0 : 
            result = aggregator(result, dataset[i])
            
    return result

def merge_func(dataset_1, dataset_2, functor):
    assert(len(dataset_1) == len(dataset_2))
    results = [];
    for i in range(len(dataset_1)):
        results.append(functor(dataset_1[i], dataset_2[i]))
    return results

def joinbykey_func(left_dataset, right_dataset, left_key = lambda left_record : left_record[0], right_key = lambda right_record: right_record[0], left_value = lambda left_record: left_record[1], right_value = lambda right_record : right_record[1]):
    dt = {}
    for right_record in right_dataset:
        key = right_key(right_record)
        assert(right_key(right_record) not in dt)
        dt[key] = right_value(right_record)
        
    result_dataset = []    
    for left_record in left_dataset:
        key = left_key(left_record)
        result = (key, left_value(left_record), dt[key])
        result_dataset.append(result)
        
    return result_dataset

def joinbykey_left_func(left_dataset, right_dataset, non_exist_value = None, left_key = lambda left_record : left_record[0], right_key = lambda right_record: right_record[0], left_value = lambda left_record: left_record[1], right_value = lambda right_record : right_record[1]):
    dt = {}
    for right_record in right_dataset:
        key = right_key(right_record)
        assert(right_key(right_record) not in dt)
        dt[key] = right_value(right_record)
        
    result_dataset = []    
    for left_record in left_dataset:
        key = left_key(left_record)
        result = ()
        if key not in dt:
            result = (key, left_value(left_record), non_exist_value)
        else:
            result = (key, left_value(left_record), dt[key])
        result_dataset.append(result)
        
    return result_dataset

def joinbykey_outer_func(left_dataset, right_dataset, non_exist_value = None, left_key = lambda left_record : left_record[0], right_key = lambda right_record: right_record[0], left_value = lambda left_record: left_record[1], right_value = lambda right_record : right_record[1]):
    dt = {}
    for right_record in right_dataset:
        key = right_key(right_record)
        assert(right_key(right_record) not in dt)
        dt[key] = right_value(right_record)

    used_dt_key = set()
        
    result_dataset = []    
    for left_record in left_dataset:
        key = left_key(left_record)
        result = ()
        if key not in dt:
            result = (key, left_value(left_record), non_exist_value)
        else:
            used_dt_key.add(key)
            result = (key, left_value(left_record), dt[key])
        result_dataset.append(result)
        
    for key in dt:
        if key not in used_dt_key:
            result_dataset.append((key, non_exist_value, dt[key]))
    return result_dataset

def groupbykey_func(dataset, key_extractor = lambda row: row[0], value_extractor = lambda row: row[1]):                                
    pairs = {}                                                                                                                                 
    for row in dataset:                                                                                                                        
        key = key_extractor(row)                                                                                                               
        value = value_extractor(row) 
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(value)                                                                                         
    return pairs;   

# every functor generates an array
def flatmap_func(dataset, functor) :
    results = []
    for row in dataset:
        array = functor(row)
        for e in array:
            results.append(e)
    return results

def indexleft_func(dataset):
    result = []
    for i in range(len(dataset)):
        result.append((i, dataset[i]))
    return result

def indexleft_flat_func(dataset):
    result = []
    for i in range(len(dataset)):
        record = dataset[i]
        record = [i] + record
        result.append(record)
    return result
    
def indexright_func(dataset):
    result = []
    for i in range(len(dataset)):
        result.append((dataset[i], i))
    return result


def select_func(dataset, functor):
    for row in dataset:
        if functor(row):
            return row
    return []

def first(dataset):
    print(dataset[0])

def print_rows(dataset, topk = -1):
    if topk == -1:
        topk = len(dataset)
    for i in range(topk):
        print(dataset[i])

def split_func(dataset, split_functor):                                                                                                                                                                
    A_data = []                                                                                                                                                                                        
    B_data = []       
    for row in dataset:
        if split_functor(row):
            A_data.append(row)
        else:
            B_data.append(row)
    return [A_data, B_data]
    
def dump_dataset(output_filename, dataset):
    with open(output_filename, 'w') as fout:
        for row in dataset:
            fout.write("%s\n" % row)

def flat_func(dataset):
    return [item for lst in dataset for item in lst]

def wordcount_func(dataset, ascend = False):
    temp = map_func(dataset, lambda r : (r, 1))
    wordcount = reducebykey_func(temp, lambda x, y : x + y)

    counts = [(k, c) for k, c in wordcount.items()]

    if ascend:
        counts.sort(key = lambda p : (p[1], p[0]))
    else:
        counts.sort(key = lambda p : (-p[1], p[0]))
    return counts
