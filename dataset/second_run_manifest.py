import polars as pl
import matplotlib.pyplot as plt


SEED = 59
CLIP_LEN = 5 # in seconds, depending on the model
ALPHA = 0.5 # upsampling factor for continued pre-training

PQ_PATH='./dataset/try_parquet.parquet'

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
    "StrGeoS1"
]

LOW_SR_HYDROS = [       # These are only used for reconstruction, not classification
    'NorthBc', # DFO_CRP hydrophone, for low SR, TKW examples and NRKW examples
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

CLASS_TO_BALANCE = {
    'Background': 0.8,
    'HW': 0.8,
    'NRKW': 0.35,
    'OKW': 0.2,
    'KW_und': 0
}


def load_df():
    return (pl.read_parquet(PQ_PATH)
            .select(*COLUMNS)
            .rename({"FileBeginSec":"SampleBeginSec", 
                     "FileEndSec": "SampleEndSec",
                     "Duration": "SampleDuration"}))

def stratified_sampling(label:str, tgt_duration:float, df:pl.DataFrame, curr_duration=0, seed=SEED):
    seed += 1
    remaining_duration = tgt_duration - curr_duration
    if remaining_duration < 300:
        return df.clear()

    pool = df.filter(pl.col('Labels') == label)
    duration_per_hydro = remaining_duration / pool['Dataset'].n_unique()
    cumul_per_hydro = pl.col('SampleDuration').cum_sum().over('Dataset')

    new_samples = (pool
                   .sample(fraction=1.0, shuffle=True)
                   .filter(cumul_per_hydro <= duration_per_hydro))

    added = new_samples['SampleDuration'].sum()

    if added == 0:
        return df.clear()

    return pl.concat([new_samples, stratified_sampling(label, tgt_duration, df, curr_duration+added, seed=seed)])

def build_test_dataset(df:pl.DataFrame):
    df = df.filter(pl.col('Dataset').is_in(TEST_HYDROS))
    return df

def build_finetune_dataset(df:pl.DataFrame):
    duration = pl.col('SampleEndSec') - pl.col('SampleBeginSec')
    center_time = pl.col('SampleBeginSec') + (duration / 2.0)
    new_start_time = pl.max_horizontal(pl.lit(0), center_time - CLIP_LEN/2.0)
    new_end_time = (new_start_time + CLIP_LEN)
    
    pool = (df.filter( 
            ~pl.col('Dataset').is_in(TEST_HYDROS),
            ~pl.col('Dataset').is_in(LOW_SR_HYDROS))
        .with_columns(
            duration.alias('true_duration'),
            center_time.alias('center_time'),
            new_start_time.alias('new_start_time'),
            new_end_time.alias('new_end_time')
        ))

    rest = pool.filter(~pl.col('Labels').is_in(CLASS_TO_BALANCE.keys()))

    srkw_time = df.filter(pl.col('Labels') == 'SRKW')['SampleDuration'].sum()

    strat_samples = []
    for c, frac in CLASS_TO_BALANCE.items():
        dur = srkw_time * frac
        strat_samples.append(stratified_sampling(c, dur, pool, seed=SEED))
    
    return pl.concat([rest, *strat_samples])


def build_pretrain_dataset(df: pl.DataFrame):

    pass


def get_srkw_ct_distribution(df:pl.DataFrame):
    return df.filter(pl.col('Labels')=='SRKW').group_by('CalltypeCategory').agg(
        (pl.col('SampleDuration').sum()/60).alias('CT total duration (mins)'),
        (pl.col('SampleDuration').len()).alias('CT counts')
    ).sort(by='CalltypeCategory').drop_nulls()

def get_label_distribution(df:pl.DataFrame):
    return df.group_by('Labels').agg(
        (pl.col('SampleDuration').sum()/60).alias('Label total duration (mins)'),
        pl.col('SampleDuration').len().alias('Label counts')
    ).sort(by='Labels')



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

    print(f"Total duration of the annotated train set:\t{train['SampleDuration'].sum()/(60**2)} hours")
    print(f"Total duration of the unannotated train set:\t{train.group_by('Soundfile').agg(
        (pl.col('SampleEndSec').max() - pl.col('SampleBeginSec').min()).alias('span'))['span'].sum()/(60**2)} hours")

    print(f"Total duration of the annotated test set:\t{test['SampleDuration'].sum()/(60**2)} hours")
    print(f"Total duration of the unannotated test set:\t{test.group_by('Soundfile').agg(
        (pl.col('SampleEndSec').max() - pl.col('SampleBeginSec').min()).alias('span'))['span'].sum()/(60**2)} hours")

    plot_splits(train, test)


if __name__ == '__main__':
    main()



