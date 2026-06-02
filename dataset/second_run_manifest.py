import polars as pl
import random
import matplotlib.pyplot as plt


SEED = 59
CLIP_LEN = 5 # in seconds, depending on the model
ALPHA = 0.5 # upsampling factor for continued pre-training

PQ_PATH='./ds/DCLDE_w_Buzzes.parquet'

COLUMNS=[
    'ID',
    'Soundfile',
    'Dataset',
    'Labels',
    'FileBeginSec',
    'FileEndSec',
    'Duration',
    'CalltypeCategory',
    'Buzz',
    'NewPath',
]


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
    return (pl.read_parquet(PQ_PATH)
            .select(*COLUMNS)
            .rename({"FileBeginSec":"SampleBeginSec", 
                     "Buzz": "SampleEndSec"}))


def get_total_time(df:pl.DataFrame, label=None):
    if not label:
        return df['Duration'].sum()
    return df.filter(pl.col('Labels')==label)['Duration'].sum()

def get_srkw_ct_distribution(df:pl.DataFrame):
    return df.filter(pl.col('Labels')=='SRKW').group_by('CalltypeCategory').agg(
        (pl.col('Duration').sum()/60).alias('CT total duration (mins)'),
        (pl.col('Duration').len()).alias('CT counts')
    ).sort(by='CalltypeCategory').drop_nulls()

def get_label_distribution(df:pl.DataFrame):
    return df.group_by('Labels').agg(
        (pl.col('Duration').sum()/60).alias('Label total duration (mins)'),
        pl.col('Duration').len().alias('Label counts')
    ).sort(by='Labels')

def jitter_sample(row:pl.Series):
    offset = random.uniform(-1.0, 1.0)
    row['SampleBeginSec'] = max(row['SampleBeginSec'].item() + offset, 0)
    row['SampleEndSec'] = min(row['SampleEndSec'].item() + offset, row['FileEndSec'].item())
    return row


def undersample_ft(label:str, tgt_duration:float, df:pl.DataFrame):
    bar = tgt_duration / df['Dataset'].n_unique()
    und_bar = (df.filter(pl.col('Labels') == label)
               .with_columns(pl.col('Duration').sum().over('Dataset').alias('total_dur'))
               .filter(pl.col('total_dur') <= bar)
               .drop('total_dur'))
    remaining_dur = tgt_duration - und_bar['Duration'].sum()
    above_bar = (df.filter(pl.col('Labels') == label)
               .join(und_bar, on='ID', how='anti')
               .sample(fraction=1.0, shuffle=True, seed=SEED)
               .with_columns((pl.col('Duration').cum_sum().over('Dataset') - \
                              pl.col('Duration')).alias('cumul'))
               .filter(pl.col('cumul') <= (remaining_dur / pl.col('Dataset').n_unique()))
               .drop('cumul'))

    return pl.concat([und_bar, above_bar, df.filter(pl.col('Labels') != label)])


def oversample_ft(label:str, tgt_duration:float, df:pl.DataFrame):
    df = (df.filter(pl.col('Labels') == label)
            .sample(fraction=1.0, shuffle=True, seed=SEED)
            .with_columns(pl.col('Duration').cum_sum().alias('cumul'))
            .filter(pl.col('cumul') <= tgt_duration)
            .map_rows(jitter_sample)
            .drop('cumul'))

def build_test_dataset(df:pl.DataFrame):
    df = df.filter(pl.col('Dataset').is_in(TEST_HYDROS))
    return df


def build_finetune_dataset(df:pl.DataFrame):
    duration = pl.col('SampleEndSec') - pl.col('SampleBeginSec')
    center_time = pl.col('SampleBeginSec') + (pl.col('Duration') / 2.0)
    new_start_time = pl.max_horizontal(pl.lit(0), center_time - CLIP_LEN/2.0)
    new_end_time = (new_start_time + CLIP_LEN)
    
    df = (df.filter( 
            ~pl.col('Dataset').is_in(TEST_HYDROS),
            ~pl.col('Dataset').is_in(LOW_SR_HYDROS),
            pl.col('Labels') != 'KW_und')
        .with_columns(
            duration.alias('true_duration'),
            center_time.alias('center_time'),
            new_start_time.alias('new_start_time'),
            new_end_time.alias('new_end_time')
        ))


    srkw_time = get_total_time(df, 'SRKW')
    classes_to_undersample = ['Background', 'HW']
    for c in classes_to_undersample:
        df = undersample_ft(c, srkw_time, df)
    classes_to_oversample = {'OKW': srkw_time/4, 'NRKW': srkw_time/2}
    for c, dur in classes_to_oversample.items():
        df = pl.concat([df, oversample_ft(c, dur, df)])
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
    # Somewhere here I should drop KW_und for fine-tuning
    train = build_finetune_dataset(df)
    test = build_test_dataset(df)

    print(f"Total duration of the annotated train set:\t{train['Duration'].sum()/(60**2)} hours")
    print(f"Total duration of the unannotated train set:\t{train.group_by('Soundfile').agg(
        (pl.col('SampleEndSec').max() - pl.col('SampleBeginSec').min()).alias('span'))['span'].sum()/(60**2)} hours")

    print(f"Total duration of the annotated test set:\t{test['Duration'].sum()/(60**2)} hours")
    print(f"Total duration of the unannotated test set:\t{test.group_by('Soundfile').agg(
        (pl.col('SampleEndSec').max() - pl.col('SampleBeginSec').min()).alias('span'))['span'].sum()/(60**2)} hours")

    plot_splits(train, test)


if __name__ == '__main__':
    main()



