from optparse import OptionParser
from bigdl.dataset.transformer import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
import math
import numpy as np
from datetime import datetime

# choose the fields you want to use as float features for learning
values_field_names = [ 'c_pkts_all:3', 's_pkts_all:17', 'c_bytes_uniq:7', 's_bytes_uniq:21', 'c_pkts_data:8', 's_pkts_data:22']
# choose field with the flow's domain name
label_field_name = 'c_tls_SNI:116'


def choose_fields(l, field_indices):
    list_with_chosen_items = []
    for ind, ele in enumerate(field_indices):
        list_with_chosen_items.append(l[field_indices[ind]])
    return list_with_chosen_items


def get_class(l, label_index, topn_names):
    class_name = l[label_index]
    if class_name in topn_names:
        class_index = topn_names.index(class_name) + 2
    else:
        class_index = 1
    return class_index

def to_categorical(y, n_cat):
    y_out = np.zeros(n_cat)
    y_out[y-1] = 1.0
    return y_out


def parse_info_data (sc, options, info_folder):
    # get headers
    datafile = sc.textFile(options.dataPath)
    rdd = datafile.filter(lambda l: l.startswith('c_ack_cnt'))
    headers = rdd.take(1)[0].split(',')
    # get topn classe names and frequencies
    label_index = headers.index(label_field_name)
    datafile = sc.textFile(options.dataPath)
    rdd = datafile.filter(lambda l: not l.startswith('c_ack_cnt'))
    rdd = rdd.map(lambda ln: ln.split(','))
    # class frequency, \xc3\xa0 la mapred
    rdd = rdd.map(lambda ln: (ln[label_index], 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending=False)
    topn_names_and_freq = rdd.take(options.topn)
    total_samples = rdd.map(lambda a: a[1]).reduce(lambda a,b: a + b)
    topn_samples = reduce(lambda a,b: a + b, map(lambda a: a[1], topn_names_and_freq))
    other_class_samples = total_samples - topn_samples
    topn_names = [ele[0] for ni,ele in enumerate(topn_names_and_freq)]
    topn_counts = [other_class_samples] + [float(ele[1]) for ni,ele in enumerate(topn_names_and_freq)]
    # fraction of samples to take from each class so that all classes get the same number of samples
    topn_frac = {}
    for t_i, t in enumerate(topn_counts):
        topn_frac[t_i+1] = topn_names_and_freq[-1][1] / (1.0*t)
    # get maximum values for the chosen fields
    values_field_indices = []
    for _,v in enumerate(values_field_names):
        values_field_indices.append(headers.index(v))
    datafile = sc.textFile(options.dataPath)
    rdd = datafile.filter(lambda l: not l.startswith('c_ack_cnt'))
    rdd = rdd.map(lambda ln: ln.split(','))
    rdd = rdd.map(lambda ln: np.array(choose_fields(ln, values_field_indices), dtype=np.float32))
    rdd = rdd.reduce(lambda a, b: np.maximum(a,b, dtype=np.float32))
    max_values = rdd
    for m_i in range(0, max_values.shape[0]):
        if max_values[m_i] == 0:
            max_values[m_i] = 1.0

    rdd = sc.parallelize([{"headers": headers, "topn_names": topn_names, "topn_frac": topn_frac, "values_field_indices": values_field_indices, "max_values": max_values, "topn_counts": topn_counts}])
    rdd.saveAsPickleFile(info_folder)

def get_info_data(sc, info_folder):
    rdd = sc.pickleFile(info_folder)
    info_dict = rdd.take(1)[0]
    return info_dict


def get_data(sc, options, info_data):
    label_index = info_data["headers"].index(label_field_name)
    datafile = sc.textFile(options.dataPath)
    rdd = datafile.filter(lambda l: not l.startswith('c_ack_cnt'))
    rdd = rdd.map(lambda ln: ln.split(','))
    rdd = rdd.map(lambda ln: [np.array(choose_fields(ln, info_data["values_field_indices"]), dtype=np.float32), get_class(ln, label_index, info_data["topn_names"])])
    rdd = rdd.map(lambda ln: [ln[1], [np.sqrt(ln[0]/info_data["max_values"]), ln[1]]])
    rdd = rdd.sampleByKey(False, info_data["topn_frac"])
    rdd = rdd.map(lambda ln: Sample.from_ndarray(ln[1][0], ln[1][1]))
    return rdd

def get_data_traintest_split(sc, train_perc, options, info_data):
    rdd = get_data(sc, options, info_data)
    (trainrdd, testrdd) = rdd.randomSplit([train_perc, 1-train_perc])
    return (trainrdd, testrdd)


def build_model(num_inputs, num_outputs, options):
    model = Sequential()

    layers = options.dnnLayers
    inputs = [num_inputs] + layers.split(':')[0:-1]
    outputs = layers.split(':')

    for n_i in range(0, len(inputs)):
        model.add(Linear(int(inputs[n_i]), int(outputs[n_i])))
        model.add(ReLU())
        model.add(Dropout(0.2))

    model.add(Linear(int(outputs[-1]), num_outputs))
    model.add(SoftMax())
    return model


def get_end_trigger(options):
    """
    When to end the optimization based on input option.
    """
    if options.endTriggerType.lower() == "epoch":
        return MaxEpoch(options.endTriggerNum)
    else:
        return MaxIteration(options.endTriggerNum)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--action", dest="action", default="train")
    parser.add_option("--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("--endTriggerType", dest="endTriggerType", default="epoch")
    parser.add_option("--endTriggerNum", type=int, dest="endTriggerNum", default="20")
    parser.add_option("--dnnLayers", dest="dnnLayers", default="1024:512")
    parser.add_option("--infoDataFile", dest="infoDataFile", default="/dev/null")
    parser.add_option("--resultsFile", dest="resultsFile", default="/dev/null")
    parser.add_option("--dataPath", dest="dataPath", default="/dev/null")
    parser.add_option("--modelPath", dest="modelPath", default="/dev/null")
    parser.add_option("--trainingPercentage", dest="trainingPercentage", type=float, default="0.8")
    parser.add_option("--topn", dest="topn", type=int, default="20")

    (options, args) = parser.parse_args(sys.argv)

    # init spark and bigdl
    myconf = create_spark_conf()
    # allow spark to overwrite hdfs files
    myconf.set("spark.hadoop.validateOutputSpecs", "False")
    sc = SparkContext(appName="infer_names_from_flows_bigdl", conf=myconf)
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    # take the tstat output and get info -- top domains, etc
    if options.action == "parse_data":
        parse_info_data(sc, options, options.infoDataFile)
        info_data = get_info_data(sc, options.infoDataFile)
        print(info_data)
    elif options.action == "print_parse_data":
        info_data = get_info_data(sc, options.infoDataFile)
        print (info_data)

    # train the specified model on the tstat dataset
    elif options.action == "train":
        info_data = get_info_data(sc, options.infoDataFile)
        (train_data, test_data) = get_data_traintest_split(sc, options.trainingPercentage, options, info_data)
        # configure and run learning
        optimizer = Optimizer(
            model=build_model(len(values_field_names), options.topn+1, options),
            training_rdd=train_data,
            criterion=CrossEntropyCriterion(),
            optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002),
            end_trigger=get_end_trigger(options),
            batch_size=options.batchSize)
        optimizer.set_checkpoint(get_end_trigger(options), options.modelPath)
        trained_model = optimizer.optimize()
        #trained_model.saveModel(options.modelPath, over_write=True)
        # compute and save accuracy results from the test data
        results = trained_model.evaluate(test_data, options.batchSize, [Top1Accuracy()])
        rdd = sc.parallelize(results).map(lambda result: "train, {}, {}, {}, {}, {}, {}, {}".format(options.trainingPercentage, result.result,        result.total_num, len (info_data["topn_counts"]), info_data["topn_counts"][-1], datetime.now(), options.infoDataFile))
        rdd.saveAsTextFile(options.resultsFile)


    # run a previously saved model and corresponding topn names on a (test) tstat dataset
    elif options.action == "test":
        info_data = get_info_data(sc, options.infoDataFile)
        test_data = get_data(sc, options, info_data)
        model = Model.load(options.modelPath)
        results = model.evaluate(test_data, options.batchSize, [Top1Accuracy()])
        # save results
        rdd = sc.parallelize(results).map(lambda result: "test, {}, {}, {}, {}, {}, {}".format(result.result, result.total_num, len (info_data["topn_counts"]), info_data["topn_counts"][-1], datetime.now(), options.infoDataFile))
        rdd.saveAsTextFile(options.resultsFile)

    sc.stop()
