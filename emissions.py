"""Energy and CO2eq accounting for a run, logged alongside the metrics.

Wraps CodeCarbon. Two entry points, because the study has two kinds of run:
  * `EmissionsLogger`  — a Lightning callback for the training cells (C3-C5)
  * `track_emissions`  — a context manager for the probe scripts (C0-C2), which
    have no Trainer but still push ~200k clips through a backbone

Two things are easy to get wrong here and both are handled:

**Offline mode.** The default `EmissionsTracker` geolocates via a web request to
find your grid's carbon intensity. Alliance compute nodes have NO internet, so
that hangs or fails. `offline=True` (the default here) uses
`OfflineEmissionsTracker` with an explicit country code instead.

**Region matters enormously.** Canada's *national average* intensity is dominated
by Alberta/Saskatchewan coal and gas, while the Calcul Quebec clusters run on a
grid that is ~95% hydro. Reporting the national average for a Quebec-hosted job
overstates emissions by roughly an order of magnitude. Set `region` (e.g.
"quebec") whenever you know where the job actually ran — see the config.

Never fatal: if codecarbon is missing or the tracker fails, this logs a warning
and the run continues. Losing an energy number is not a reason to lose a run.
"""

from contextlib import contextmanager

from lightning.pytorch.callbacks import Callback

from util.pylogger import get_pylogger

log = get_pylogger(__name__)


def _make_tracker(offline, country_iso_code, region, project_name, output_dir,
                  measure_power_secs):
    try:
        from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    except ImportError:
        log.warning("codecarbon not installed — skipping emissions tracking "
                    "(pip install codecarbon)")
        return None

    kwargs = dict(
        project_name=project_name,
        output_dir=output_dir,
        measure_power_secs=measure_power_secs,
        log_level="error",          # codecarbon is chatty at info level
        save_to_file=True,          # emissions.csv next to the run, as a backup
    )
    try:
        if offline:
            return OfflineEmissionsTracker(
                country_iso_code=country_iso_code, region=region, **kwargs
            )
        return EmissionsTracker(**kwargs)
    except Exception as e:
        log.warning(f"could not start emissions tracker: {e}")
        return None


def _metrics(tracker):
    """Pull the numbers out of a stopped tracker, tolerating schema differences."""
    emissions = tracker.stop()
    out = {"emissions/co2eq_kg": float(emissions or 0.0)}
    data = getattr(tracker, "final_emissions_data", None)
    if data is not None:
        for key, name in [
            ("energy_consumed", "emissions/energy_kwh"),
            ("gpu_energy", "emissions/gpu_energy_kwh"),
            ("cpu_energy", "emissions/cpu_energy_kwh"),
            ("ram_energy", "emissions/ram_energy_kwh"),
            ("duration", "emissions/duration_s"),
        ]:
            val = getattr(data, key, None)
            if val is not None:
                out[name] = float(val)
    return out


class EmissionsLogger(Callback):
    """Log energy + CO2eq for a training run to whatever logger the Trainer has."""

    def __init__(self, offline=True, country_iso_code="CAN", region=None,
                 project_name=None, measure_power_secs=30):
        super().__init__()
        self.offline = offline
        self.country_iso_code = country_iso_code
        self.region = region
        self.project_name = project_name
        self.measure_power_secs = measure_power_secs
        self.tracker = None

    def on_fit_start(self, trainer, pl_module):
        self.tracker = _make_tracker(
            self.offline, self.country_iso_code, self.region,
            self.project_name or getattr(trainer.logger, "name", "torca"),
            str(trainer.log_dir or trainer.default_root_dir or "."),
            self.measure_power_secs,
        )
        if self.tracker is not None:
            try:
                self.tracker.start()
            except Exception as e:
                log.warning(f"emissions tracker failed to start: {e}")
                self.tracker = None

    def _finish(self, trainer):
        if self.tracker is None:
            return
        try:
            metrics = _metrics(self.tracker)
        except Exception as e:
            log.warning(f"emissions tracker failed to stop cleanly: {e}")
            self.tracker = None
            return
        self.tracker = None
        log.info(f"run emissions: {metrics.get('emissions/co2eq_kg', 0):.6f} kg CO2eq, "
                 f"{metrics.get('emissions/energy_kwh', 0):.4f} kWh")

        # Write straight to the loggers, NOT via pl_module.log_dict: Lightning's
        # logging machinery is only connected inside the train/val/test loops, and
        # on_fit_end runs after they have torn down — self.log() there is dropped or
        # raises, so the number would silently never arrive.
        for logger in (trainer.loggers or []):
            try:
                logger.log_metrics(metrics, step=trainer.global_step)
            except Exception as e:
                log.warning(f"could not log emissions to {type(logger).__name__}: {e}")

    def on_fit_end(self, trainer, pl_module):
        self._finish(trainer)

    def on_exception(self, trainer, pl_module, exception):
        # a crashed run still burned the energy; record it before propagating
        self._finish(trainer)


@contextmanager
def track_emissions(logger=None, offline=True, country_iso_code="CAN", region=None,
                    project_name="torca_probe", output_dir=".", measure_power_secs=30):
    """Context manager for scripts with no Trainer (the probe cells).

    `logger` is any lightning logger; if given, the metrics are pushed to it, else
    they are only printed. Yields the metrics dict, populated on exit.
    """
    tracker = _make_tracker(offline, country_iso_code, region, project_name,
                            output_dir, measure_power_secs)
    metrics = {}
    if tracker is not None:
        try:
            tracker.start()
        except Exception as e:
            log.warning(f"emissions tracker failed to start: {e}")
            tracker = None
    try:
        yield metrics
    finally:
        if tracker is not None:
            try:
                metrics.update(_metrics(tracker))
                log.info(f"run emissions: {metrics.get('emissions/co2eq_kg', 0):.6f} kg "
                         f"CO2eq, {metrics.get('emissions/energy_kwh', 0):.4f} kWh")
                if logger is not None:
                    logger.log_metrics(metrics)
            except Exception as e:
                log.warning(f"emissions tracker failed to stop cleanly: {e}")
