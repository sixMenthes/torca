import polars as pl
from torch.utils.data import DataLoader

from util.pylogger import get_pylogger
from models.components.fbank_frontend import make_frontend
from torca_datamodule import LabelDataModule, collate_fn_skip
from torca_transforms import RandomShift, RandomGain, AddBackgroundNoise
from selfdistill_dataset import SelfDistillDataset, WaveformViewAug

log = get_pylogger(__name__)


def _cap_per_hydrophone(df, cap, seed, stratify_col=None, what="pool"):
    """Cap each hydrophone's contribution at `cap` clips, sampling without replacement.

    Sites already at or below the cap are returned untouched, so this can only ever
    remove clips from the sites that dominate. It follows that a cap cannot manufacture
    diversity that is not in the data: the achievable ceiling is set by how many sites
    contribute a meaningful number in the first place.

    With `stratify_col`, allocation is PROPORTIONAL — each group keeps its share of the
    site's mix and only the total shrinks. A group is never reduced below one clip, so a
    class present at a site cannot vanish from it entirely through rounding; that makes
    the totals land within a few clips of the cap rather than exactly on it, which does
    not matter for anything downstream.

    Sampling is seeded, so the pool is identical across runs, across the two backbone
    arms, and between a run and the report that describes it. That is not a nicety: the
    ablation compares cells that must differ only in the treatment under test, and a
    pool that reshuffled per run would add a second difference nobody controlled.
    """
    if not cap:
        return df

    kept, dropped = [], 0
    for (site,), sub in df.group_by(["Dataset"], maintain_order=True):
        if sub.height <= cap:
            kept.append(sub)
            continue
        dropped += sub.height - cap
        if stratify_col is None:
            kept.append(sub.sample(n=cap, shuffle=True, seed=seed))
            continue
        frac = cap / sub.height
        parts = []
        for (_g,), grp in sub.group_by([stratify_col], maintain_order=True):
            n = min(grp.height, max(1, int(round(grp.height * frac))))
            parts.append(grp.sample(n=n, shuffle=True, seed=seed))
        kept.append(pl.concat(parts))

    out = pl.concat(kept)
    if dropped:
        log.info(
            f"{what}: capped at {cap} clips per hydrophone, dropped {dropped} "
            f"({df.height} -> {out.height})"
        )
    return out


class SelfDistillDataModule(LabelDataModule):
    """Self-supervised adaptation datamodule (EMA self-distillation).

    Reuses LabelDataModule's manifest loading, GCS download and hydrophone
    bookkeeping, but differs from the classification pool in three ways:
      * the pool INCLUDES low-SR hydrophones and is label-agnostic — no class
        loss runs here, and more channel variety helps the invariance objective;
      * only ``test_hydros`` and ``val_hydros`` are held out (test must stay
        unseen; val is kept for monitoring);
      * each item is a (teacher, student) two-view pair of augmented WAVEFORMS,
        with the mel/PCEN front-end deferred to the model.
    """

    def _ssl_pool_uncapped(self):
        # Everything except held-out test/val; low-SR INCLUDED (the classifier
        # pool excludes it, we don't). No Labels filter — SSL is label-agnostic.
        held_out = self.test_hydros + self.val_hydros
        return self.df.filter(~pl.col("Dataset").is_in(held_out))

    def build_selfdistill_set(self):
        """The adaptation pool, capped so that no one hydrophone dominates it.

        Uncapped, this pool was 60% WVanIsl and 23.6% NorthBc: two sites supplying
        83.6% of everything the model adapts on, and both of them low-sample-rate sites
        that no probe ever evaluates. Measured as an inverse Simpson index — the number
        of equally sized hydrophones that would give the same probability of two random
        clips sharing a site — 26 hydrophones amounted to an effective 2.4. A cap of
        10,000 raises that to 8.8 and leaves ~51k clips.

        Two problems are fixed by the one rule. The first is channel concentration,
        which matters because a representation adapted overwhelmingly on one recording
        chain narrows onto it, and the chains we evaluate on are not that one. The
        second is content: WVanIsl is 84.2% humpback and NorthBc 58.2%, so the pool as a
        whole was roughly 65% humpback in a study about killer whale ecotypes. Because
        the cap bites hardest exactly on the two humpback-heavy sites, it takes that to
        about 31% without needing a separate per-class ceiling.

        Allocation is PROPORTIONAL within each site: every class keeps its share of that
        site's mix and only the total shrinks. The alternative, equal allocation, would
        flatten each site's class balance, which would be a much more invasive change to
        the data than the concentration problem calls for.
        """
        return _cap_per_hydrophone(
            self._ssl_pool_uncapped(),
            self.dataset_configs.get("max_clips_per_hydro", None),
            seed=int(self.dataset_configs.get("subsample_seed", 59)),
            stratify_col="Labels",
            what="adaptation pool",
        )

    def _ssl_background_bank(self):
        """Background clips used as ADDITIVE NOISE in the student view, capped per site.

        This is the de-confounding lever: the teacher sees a clip clean, the student
        sees it buried in noise from a DIFFERENT hydrophone, and the only way to predict
        the teacher's tokens is to stop representing the channel. The bank is therefore
        what the phrase "cross-hydrophone noise" actually refers to, which makes it what
        cell C5 (background.p=0.0) is the control FOR. Uncapped it was 43.8% NorthBc and
        28.8% WVanIsl, an effective 3.5 channels out of 15, and at those shares C4
        against C5 would be testing whether adding those two sites' noise helps rather
        than whether cross-hydrophone noise helps.

        Drawn from the FULL manifest rather than from the capped adaptation pool, and
        capped separately. A noise source does not need to be a training example, so
        narrowing the bank to the capped pool would discard channel variety for nothing.

        No stratification here: every clip in the bank is Background by construction, so
        there is nothing to stratify on.
        """
        held_out = self.test_hydros + self.val_hydros
        bank = self.df.filter(
            pl.col("Labels") == "Background", ~pl.col("Dataset").is_in(held_out)
        )
        bank = _cap_per_hydrophone(
            bank,
            self.dataset_configs.get("max_bank_clips_per_hydro", None),
            seed=int(self.dataset_configs.get("subsample_seed", 59)),
            stratify_col=None,
            what="background bank",
        )
        return bank.get_column("LocalPath").to_list()

    def _build_view_aug(self, view_cfg, sr, max_length, bank):
        """Build one view's augmentation pipeline from config.

        Ops with p == 0 are dropped entirely rather than constructed-and-skipped, so
        sweeping a single op off costs one override and doesn't pay for e.g. the
        background bank's file IO. Order is fixed (shift -> gain -> background):
        background noise must be mixed at the SNR of the final signal, so it goes last.
        """
        ops = []
        if not view_cfg:
            return WaveformViewAug(ops)

        shift = view_cfg.get("shift", None)
        if shift and shift.p > 0:
            ops.append(RandomShift(
                max_shift_samples=int(shift.max_shift_seconds * sr), p=shift.p))

        gain = view_cfg.get("gain", None)
        if gain and gain.p > 0:
            ops.append(RandomGain(
                min_gain_db=gain.min_gain_db, max_gain_db=gain.max_gain_db, p=gain.p))

        bg = view_cfg.get("background", None)
        if bg and bg.p > 0:
            ops.append(AddBackgroundNoise(
                background_paths=bank, target_length=max_length, sample_rate=sr,
                min_snr_db=bg.min_snr_db, max_snr_db=bg.max_snr_db, p=bg.p))

        return WaveformViewAug(ops)

    def _build_frontend(self, sr):
        """Fbank front-end for this dataset's backbone, or None to leave it in the model.

        `model_name` lives in the DATASET config because the dataset is what produces
        the features now — and the two backbones need genuinely different fbanks
        (BEATs scales by 2**15 and skips mean-subtraction; Bird-MAE does the opposite
        and pads to a fixed target_length).

        The sample-rate cross-check is the important part: model_name and the
        transform's sample_rate come from different config files, and a BEATs
        front-end fed 32 kHz audio would produce a perfectly valid-looking
        spectrogram that is simply wrong. Fail loudly instead.
        """
        name = self.dataset_configs.get("model_name", None)
        if not name:
            return None
        expected = {"BEATs": 16000, "BirdMAE": 32000}[name]
        if sr != expected:
            raise ValueError(
                f"dataset model_name={name} expects {expected} Hz but the transform "
                f"gives {sr} Hz — the dataset yaml and module/network disagree"
            )
        return make_frontend(name, sample_rate=sr,
                             target_length=int(self.transform_config.target_length))

    def setup(self, stage: str):
        if stage in ("fit", None):
            sr = int(self.transform_config.input.sample_rate)
            max_length = int(sr * self.transform_config.clip_duration)
            bank = self._ssl_background_bank()
            frontend = self._build_frontend(sr)

            # Asymmetric views: teacher clean, student strong. This asymmetry is what
            # turns the masked-prediction objective into a denoising / channel-
            # invariance objective — the cheapest form of the "multiple views"
            # iBOT/DINO use, no second head or multi-crop yet.
            aug_cfg = self.transform_config.get("augmentations", {})
            teacher_aug = self._build_view_aug(aug_cfg.get("teacher", {}), sr, max_length, bank)
            student_aug = self._build_view_aug(aug_cfg.get("student", {}), sr, max_length, bank)

            self.ssl_set = SelfDistillDataset(
                self.build_selfdistill_set(),
                teacher_aug,
                student_aug,
                sample_rate=sr,
                max_length=max_length,
                frontend=frontend,
            )

            # Held-out RECORDING CONDITION (CarmanahPt), which is the right thing to
            # monitor for an invariance objective: it answers "is the masked-prediction
            # task improving on a channel the model never adapted on", not just on the
            # training channels. Same augmentation policy as training (the student view
            # keeps cross-hydrophone noise, so this measures denoising of an unseen
            # channel), but deterministic — each clip is augmented and masked
            # identically every epoch, so the curve moves only when the model does.
            self.val_ssl_set = SelfDistillDataset(
                self.df.filter(pl.col("Dataset").is_in(self.val_hydros)),
                teacher_aug,
                student_aug,
                sample_rate=sr,
                max_length=max_length,
                frontend=frontend,
                deterministic=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.ssl_set,
            num_workers=self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=True,
            persistent_workers=self.train_loader_configs.get(
                "persistent_workers", False
            ),
            pin_memory=self.train_loader_configs.get("pin_memory", False),
            collate_fn=collate_fn_skip,
        )

    def val_dataloader(self):
        """Two-view val loader. MUST override the inherited one.

        LabelDataModule.val_dataloader returns a loader over `self.val_set`, which this
        datamodule never builds — so the moment a validation_step exists, the inherited
        version raises AttributeError. shuffle=False keeps batch composition fixed
        across epochs, which the deterministic augmentation relies on.
        """
        return DataLoader(
            self.val_ssl_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=False,
            persistent_workers=self.val_loader_configs.get(
                "persistent_workers", False
            ),
            pin_memory=self.val_loader_configs.get("pin_memory", False),
            collate_fn=collate_fn_skip,
        )

    def test_dataloader(self):
        """There is deliberately no test stage on this path.

        MIMDistillation has no test_step, and should not: the SSL objective is not the
        study's metric, and computing it over the test hydrophones would touch a split
        the protocol keeps sealed until the end. Test evaluation is
        probe.probe_split_protocol — fit on train hydrophones, choose C by GroupKFold
        within train, touch test exactly once.

        Overridden because the inherited version reads self.test_set, which setup()
        never builds here (it handles stage "fit" only), so calling it would raise an
        AttributeError that says nothing about why. Same reason val_dataloader is
        overridden above.
        """
        raise NotImplementedError(
            "no test stage for self-distillation — evaluate with probe_selfdistill.py "
            "(source=adapted ckpt_path=...), which fits the probe on train hydrophones "
            "and touches the sealed test split exactly once"
        )

    def probe_dataloader(self, batch_size=32):
        """LABELLED clips from ONE validation hydrophone, for the online ecotype probe.

        Single hydrophone by design. With the recording condition held constant the
        probe cannot score well by reading the channel instead of the call, so the
        number is a clean read on whether ecotype is linearly separable, which is the
        quantity the adaptation is meant to improve.

        This reads `online_probe_hydro` rather than `val_hydros` because those two
        lists now serve different purposes. `val_hydros` carries more than one site so
        that `val/loss`, the self-supervised masked-prediction loss that drives
        checkpoint selection, is measured across several unseen channels instead of
        one. That is the right choice for selection and the wrong choice here: with
        several validation hydrophones the online probe can reach a high score by
        learning which channel a clip came from, and hydrophone identity is correlated
        with ecotype in this dataset (Cpe_Elz is roughly 89 percent TKW), so the
        shortcut is a large one. Pinning the probe to a single site keeps the two
        signals honest independently.

        Falls back to the first entry of `val_hydros` when the key is absent, so an
        older dataset config still composes.
        """
        from probe_features import build_loader

        hydro = self.dataset_configs.get("online_probe_hydro", None)
        if not hydro:
            hydro = list(self.val_hydros)[0]
            log.warning(
                f"no online_probe_hydro in the dataset config; falling back to "
                f"'{hydro}'. With {len(self.val_hydros)} validation hydrophones the "
                f"online probe can exploit a channel shortcut if this is not pinned."
            )
        labelled = self.df.filter(
            (pl.col("Dataset") == hydro)
            & pl.col("Labels").is_in(list(self.labels))
        )
        log.info(f"online probe: {labelled.height} labelled clips from '{hydro}'")
        return self._probe_loader(labelled, batch_size)

    def code_usage_dataloader(self, background, max_per_hydro=200, batch_size=32):
        """TRAIN-hydrophone clips for the code-usage probe, split on Background.

        `background=True` gives Background-labelled clips, `False` gives the labelled
        vocalisations. Tokenising both and comparing their code histograms asks how much
        of the codebook is spent representing NOISE rather than calls, which is the
        direct test of whether a 1000-code codebook has capacity to spare for recording
        condition. Splitting the Background histogram by hydrophone then asks whether it
        is in fact spending it that way.

        TRAIN hydrophones, not validation, and this is not a shortcut. The validation
        sites are Cpe_Elz and StrGeoS1, holding 48 and 3 Background clips between them:
        two sites, and no basis for a site-versus-code measurement at all. The reported
        nuisance figure is computed on train for the same reason — see
        probe.probe_nuisance_background — so both live on the same population.

        Background-only is what makes the site comparison about CHANNEL rather than
        content. Hydrophone identity correlates with ecotype here (Cpe_Elz is ~89% TKW),
        so a site measurement over all clips can succeed by reading the vocalisation
        instead of the recording condition. With no orca present, what separates sites is
        instrument response, noise floor, depth and self-noise, which is the confound.

        Low-SR sites are excluded, matching the reported metric's pool. They would also
        be separable on bandwidth alone, which would inflate any site signal for a reason
        that has nothing to do with the representation.

        `max_per_hydro` caps each site so no one hydrophone dominates the pooled
        histogram — the same concentration problem the adaptation pool had, and here it
        would directly distort the entropy being measured. It also keeps the per-epoch
        cost down. Absolute values therefore will not match the offline probe; read the
        trend.
        """
        held = list(self.test_hydros) + list(self.val_hydros) + list(self.low_sr_hydros)
        is_bg = pl.col("Labels") == "Background"
        frame = self.df.filter(
            (is_bg if background else (~is_bg & pl.col("Labels").is_in(list(self.labels))))
            & ~pl.col("Dataset").is_in(held)
        )
        what = "background" if background else "vocalisation"
        frame = _cap_per_hydrophone(
            frame, max_per_hydro,
            seed=int(self.dataset_configs.get("subsample_seed", 59)),
            what=f"code-usage probe ({what})",
        )
        log.info(
            f"code-usage probe [{what}]: {frame.height} clips over "
            f"{frame.get_column('Dataset').n_unique()} train hydrophones"
        )
        return self._probe_loader(frame, batch_size)

    def train_probe_dataloader(self, max_per_hydro=300, batch_size=32):
        """ALL labelled TRAIN-hydrophone clips, Background included, capped per site.

        One loader feeding two probes, because both want features on the same population
        and a second pass over several thousand clips through a ViT-B is the dominant
        cost of the callback. `TrainCVProbe` extracts features once and then reuses the
        matrix: the ecotype probe scores every row, which is the four-way problem over
        `labels` = [Background, HW, SRKW, TKW] that the reported metric also solves, and
        the nuisance probe scores the Background rows alone.

        TRAIN hydrophones, and this is the point of the whole loader. The reported task
        number comes from `probe_selfdistill.py`, which fits on train and scores once on
        the sealed test split, so it cannot be watched during training without spending
        the test split on hyperparameter search. Cross-validating inside train gives a
        signal that can be read every validation epoch at no methodological cost, because
        there is nothing here to burn.

        Low-SR sites are excluded along with test and val, matching the offline probe's
        pool. They are separable on bandwidth alone, which would inflate the nuisance
        figure for a reason that has nothing to do with the representation.

        `max_per_hydro` caps each site, stratified by label so each site keeps its class
        mix. Without it HaroStraitSouth alone would supply a large share of the Background
        rows, and for the nuisance probe the hydrophone IS the label, so an uncapped pool
        makes the class balance an artefact of how much each site happened to record.
        """
        held = list(self.test_hydros) + list(self.val_hydros) + list(self.low_sr_hydros)
        frame = self.df.filter(
            pl.col("Labels").is_in(list(self.labels))
            & ~pl.col("Dataset").is_in(held)
        )
        frame = _cap_per_hydrophone(
            frame, max_per_hydro,
            seed=int(self.dataset_configs.get("subsample_seed", 59)),
            stratify_col="Labels",
            what="train-CV probe",
        )
        log.info(
            f"train-CV probe: {frame.height} clips over "
            f"{frame.get_column('Dataset').n_unique()} train hydrophones"
        )
        return self._probe_loader(frame, batch_size)

    def _probe_loader(self, frame, batch_size):
        """Shared loader construction for the two online probes.

        Both need the same thing — labelled clips decoded exactly as the SSL dataset
        decodes them — and building it twice would be two places for the sample rate or
        the clip duration to drift apart from what the model was adapted on.
        """
        from probe_features import build_loader

        return build_loader(
            frame, int(self.transform_config.input.sample_rate),
            self.clip_duration, self.label_map, self.call_map,
            batch_size=batch_size, num_workers=self.val_loader_configs.num_workers,
        )
