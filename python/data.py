def load_xes(path):
    log = xes_importer.apply(path)
    return log

def logs_to_df(log):
    df = pm4py.convert_to_dataframe(log)
    return df