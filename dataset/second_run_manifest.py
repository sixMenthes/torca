import polars as pl
import numpy as np
import matplotlib.pyplot as plt


SEED = 59
CLIP_LEN = 5 # in seconds, depending on the model
ALPHA = 0.5 # upsampling factor for continued pre-training

IN_PATH='./dataset/DCLDE_w_Buzzes.parquet'

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
    "StrGeoS1"         # 
]

VAL_HYDROS = [
    "CarmanahPt" # we need to actually mark it on the dataset......
]

LOW_SR_HYDROS = [       # These are only used for reconstruction, not classification
    'NorthBc',          # DFO_CRP hydrophone, for low SR, TKW examples and NRKW examples
    'WVanIsl',          # DFO_CRP hydrophone, for low SR, TKW examples and NRKW examples
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
    return (pl.read_parquet(IN_PATH)
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
    test = df.filter(pl.col('Dataset').is_in(TEST_HYDROS))
    test.write_parquet('./dataset/manifest_out-of-sample.parquet')
    return test

def build_finetune_train_dataset(df:pl.DataFrame):
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

def build_curator_dataset(finetune_df:pl.DataFrame):
    tgt_class = finetune_df.filter(pl.col('Labels') == 'Background')
    other_class = (finetune_df
                   .filter(pl.col('Labels') != 'Background')
                   .sample(n=tgt_class.height, shuffle=True, seed=SEED))
    pl.concat([tgt_class, other_class]).write_parquet('./dataset/manifest_curator.parquet')



def build_pretrain_dataset(df: pl.DataFrame):
    pass


def get_srkw_ct_distribution(df:pl.DataFrame):
    return df.filter(pl.col('Labels')=='SRKW').group_by('CalltypeCategory').agg(
        (pl.col('SampleDuration').sum()/60).alias('CT total duration (mins)'),
        (pl.col('SampleDuration').len()).alias('CT counts')
    ).sort(by='CalltypeCategory').drop_nulls()

def get_label_distribution(df:pl.DataFrame, hydrophone:str=None):
    if hydrophone:
        df = df.filter(pl.col('Dataset') == hydrophone)
    return df.group_by('Labels').agg(
        (pl.col('SampleDuration').sum()/60).alias('Label total duration (mins)'),
        pl.col('SampleDuration').len().alias('Label counts')
    ).sort(by='Labels')

def save_legend(hydro_dict, path, ncol=4):
      fig = plt.figure(figsize=(0.01, 0.01))
      handles = [plt.plot([], [], marker='s', ls='none', color=c, label=h)[0]
                 for h, c in hydro_dict.items()]
      fig.legend(handles=handles, loc='center', ncol=ncol, frameon=False)
      fig.savefig(path, bbox_inches='tight', dpi=200)
      plt.close(fig)




def plot_split(axs, ds, hydro_dict):


    ct_distribution = get_srkw_ct_distribution(ds)
    all_labels = ds.select(pl.col('Labels').unique().sort())
    bottom = np.zeros(all_labels.height)
    for h in hydro_dict.keys():
        ld = get_label_distribution(ds, h)
        aligned = all_labels.join(ld, on='Labels', how='left').fill_null(0)
        vals = aligned['Label total duration (mins)'].to_numpy()
        axs[0].bar(aligned['Labels'], vals, label=h, bottom=bottom, color=hydro_dict[h])
        bottom += vals

    axs[1].bar(ct_distribution['CalltypeCategory'], ct_distribution['CT total duration (mins)'])

def plot_splits(train_ds:pl.DataFrame, test_ds:pl.DataFrame):

    hydros = pl.concat([train_ds.select('Dataset'), test_ds.select('Dataset')])['Dataset'].unique()
    colors = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)
    hydro_dict = {h: colors[i] for i, h in enumerate(sorted(hydros))}
    save_legend(hydro_dict, './dataset/hydro_legend.png', ncol=4)
    fig, axs = plt.subplots(2,2)
    plot_split(axs[0], train_ds, hydro_dict)
    plot_split(axs[1], test_ds, hydro_dict)

    leg_fig = plt.figure(figsize=(6, 4))
    handles = [plt.plot([], [], marker='s', ls='none', color=c, label=h)[0] for h, c in hydro_dict.items()]
    leg_fig.legend(handles=handles, loc='center', ncol=4, frameon=False)

    fig.tight_layout()
    plt.show()


def main():
    df = load_df()
    # Somewhere here I should drop KW_und for fine-tuning
    test = build_test_dataset(df)
    ft_train = build_finetune_train_dataset(df)
    ft_train.write_parquet('./dataset/manifest_ft-train.parquet')
    build_curator_dataset(ft_train)

    print(f"Total duration of the finetune train set:\t{ft_train['SampleDuration'].sum()/(60**2)} hours")

    print(f"Total duration of the annotated test set:\t{test['SampleDuration'].sum()/(60**2)} hours")
    plot_splits(ft_train, test)


if __name__ == '__main__':
    main()



