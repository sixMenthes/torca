import polars as pl
import matplotlib.pyplot as plt


SEED = 59
CLIP_LEN = 5 # in seconds, depending on the model
ALPHA = 0.5 # upsampling factor for continued pre-training

PQ_PATH='./ds/DCLDE_w_Buzzes.parquet'
TEST_HYDROS = [
    "StraitofGeorgia", # JASCO hydrophone for the calltype data quality
    'NorthBc', # DFO_CRP hydrophone, for low SR, TKW examples and NRKW examples
]
LOW_SR_HYDROS = [       # These are only used for reconstruction, not classification
    'orcasound_lab', 
    'MS_671879205',
    'Field_HTI',
    'KB_67383303',
    'RB_335826997',
    'KB_67424266',
    'HE_67424266',
    'RB_67424266',
    'HE_67391498',
    'MS_6897',
    'KB_5354',
    'MS_5360',
    'Field_SondTrap'
]


def load_df():
    return pl.read_parquet(PQ_PATH)


def get_total_time(df:pl.DataFrame, label=None):
    if not label:
        return df['Duration'].sum()
    return df.filter(pl.col('Labels')==label)['Duration'].sum()

def get_srkw_ct_distribution(df:pl.DataFrame):
    return df.filter(pl.col('Ecotype')=='SRKW').group_by('CalltypeCategory').agg(
        (pl.col('Duration').sum()/60).alias('CT total duration (mins)'),
        (pl.col('Duration').len()).alias('CT counts')
    ).sort(by='CalltypeCategory').drop_nulls()

def get_label_distribution(df:pl.DataFrame):
    return df.group_by('Labels').agg(
        (pl.col('Duration').sum()/60).alias('Label total duration (mins)'),
        pl.col('Duration').len().alias('Label counts')
    ).sort(by='Labels')

def per_hydro_undersample(label:str, tgt_duration:float, df:pl.DataFrame):
    duration_per_hydro = tgt_duration / df['Dataset'].n_unique()
    strat_per_hydro = (df.filter(pl.col('Labels') == label)
                       .sample(fraction=1.0, shuffle=True, seed=SEED)
                       .sort("Dataset")
                       .with_columns(pl.col("Duration").cum_sum().over('Dataset').alias('cumul'))
                       .filter(pl.col('cumul') <= duration_per_hydro)
                       .drop('cumul'))

    rest = df.filter(pl.col('Labels') != label)
    return pl.concat([strat_per_hydro, rest])


def build_test_dataset(df:pl.DataFrame):
    df = df.filter(pl.col('Dataset').is_in(TEST_HYDROS))
    srkw_time = get_total_time(df, 'SRKW')
    classes_to_undersample = ['Background', 'HW']
    for c in classes_to_undersample:
        df = per_hydro_undersample(c, srkw_time*1.5, df)
    return df


def build_finetune_dataset(df:pl.DataFrame):
    df = df.filter(
        ~pl.col('Dataset').is_in(TEST_HYDROS),
        ~pl.col('Dataset').is_in(LOW_SR_HYDROS),
        pl.col('Labels') != 'KW_und')
    srkw_time = get_total_time(df, 'SRKW')
    classes_to_undersample = ['Background', 'HW']
    for c in classes_to_undersample:
        df = per_hydro_undersample(c, srkw_time, df)
    return df


def build_pretrain_dataset(df: pl.DataFrame):

    pass



def plot_split(axs, ds):

    label_distribution = get_label_distribution(ds)
    ct_distribution = get_srkw_ct_distribution(ds)
    axs[0].bar(label_distribution['Labels'], label_distribution['Label total duration (mins)'])
    axs[1].bar(ct_distribution['CalltypeCategory'], ct_distribution['CT total duration (mins)'])

def plot_splits(train_ds, test_ds):

    fig, axs = plt.subplots(2,2)
    plot_split(axs[0], train_ds)
    plot_split(axs[1], test_ds)

    plt.show()












def main():
    df = load_df()
    train = build_finetune_dataset(df)
    test = build_test_dataset(df)

    print(f"Total duration of the annotated train set:\t{train['Duration'].sum()/(60**2)} hours")
    print(f"Total duration of the unannotated train set:\t{train.group_by('Soundfile').agg(
        (pl.col('FileEndSec').max() - pl.col('FileBeginSec').min()).alias('span'))['span'].sum()/(60**2)} hours")

    print(f"Total duration of the annotated test set:\t{test['Duration'].sum()/(60**2)} hours")
    print(f"Total duration of the unannotated test set:\t{test.group_by('Soundfile').agg(
        (pl.col('FileEndSec').max() - pl.col('FileBeginSec').min()).alias('span'))['span'].sum()/(60**2)} hours")

    plot_splits(train, test)


if __name__ == '__main__':
    main()



