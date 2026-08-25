"""Dear PyGui implementation of the native experiment diagnostics workbench."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import time

import dearpygui.dearpygui as dpg

from .catalog import ExperimentCatalog
from .activity import (
    ActivityEventKind,
    ActivityFrame,
    ActivityPhaseRun,
    ActivityStreamReader,
    reader_for_run,
)
from .model import ExperimentRun, ProportionEstimate
from .structure import (
    DensityOverview,
    Neighborhood,
    NetworkStructure,
    StructureUnavailable,
    open_structure,
)


PALETTE = {
    "basalt": (16, 24, 32, 255),
    "basalt_2": (22, 33, 43, 255),
    "basalt_3": (29, 43, 54, 255),
    "axon": (216, 228, 232, 255),
    "muted": (133, 153, 161, 255),
    "ionic": (88, 185, 182, 255),
    "spike": (242, 193, 78, 255),
    "inhibitory": (216, 108, 90, 255),
    "dendrite": (133, 131, 198, 255),
    "quiet": (57, 75, 86, 255),
    "danger": (237, 104, 104, 255),
}

SERIES_COLORS = (
    PALETTE["ionic"],
    PALETTE["spike"],
    PALETTE["dendrite"],
    PALETTE["inhibitory"],
    (117, 168, 219, 255),
    (144, 196, 137, 255),
)

METRICS = {
    "Calibration accuracy": "calibration_accuracy",
    "Training accuracy": "training_accuracy",
    "Teacher replay rate": "replay_rate",
    "Generalization gap": "generalization_gap",
    "Median decision margin (pA)": "margin_median_pa",
    "Minimum decision margin (pA)": "margin_min_pa",
    "State separation (1-cos)": "state_separation",
    "Recurrent connection density": "recurrent_density",
    "Input assembly fraction": "assembly_fraction",
    "Wall time (s)": "wall_seconds",
}

DIMENSIONS = {
    "Configured neurons": "neurons",
    "Training tasks": "data_scale",
    "Inhibitory fraction": "inhibitory_fraction",
    "Input assembly size": "assembly_size",
    "Seed": "seed",
}

GROUPINGS = {
    "Codebook": "codebook",
    "Readout rule": "readout",
    "Schedule": "schedule",
    "Supervision": "supervision",
    "Backend": "backend",
}

ACTIVITY_EVENT_FILTERS: dict[str, ActivityEventKind | None] = {
    "Any event": None,
    "Phase change": "phase",
    "State sample": "state",
    "Zero-spike token": "zero",
}

ACTIVITY_SPEEDS = (
    "0.001×",
    "0.005×",
    "0.01×",
    "0.05×",
    "0.1×",
    "1×",
    "10×",
)

TAGS = {
    "primary": "hbv.primary",
    "status": "hbv.status",
    "experiment": "hbv.experiment",
    "search": "hbv.search",
    "catalog_info": "hbv.catalog_info",
    "cards": "hbv.cards",
    "center": "hbv.center",
    "ledger": "hbv.ledger",
    "ledger_handlers": "hbv.ledger_handlers",
    "run_table": "hbv.run_table",
    "details": "hbv.details",
    "x_choice": "hbv.x_choice",
    "metric_choice": "hbv.metric_choice",
    "group_choice": "hbv.group_choice",
    "plot": "hbv.plot",
    "x_axis": "hbv.x_axis",
    "y_axis": "hbv.y_axis",
    "structure_gid": "hbv.structure.gid",
    "structure_hops": "hbv.structure.hops",
    "structure_info": "hbv.structure.info",
    "structure_canvas": "hbv.structure.canvas",
    "density_info": "hbv.structure.density_info",
    "density_canvas": "hbv.structure.density_canvas",
    "activity_status": "hbv.activity.status",
    "activity_plot": "hbv.activity.plot",
    "activity_x_axis": "hbv.activity.x_axis",
    "activity_y_axis": "hbv.activity.y_axis",
    "activity_cursor": "hbv.activity.cursor",
    "activity_phase_tape": "hbv.activity.phase_tape",
    "activity_play": "hbv.activity.play",
    "activity_slider": "hbv.activity.slider",
    "activity_speed": "hbv.activity.speed",
    "activity_event_filter": "hbv.activity.event_filter",
    "activity_event": "hbv.activity.event",
    "activity_frame": "hbv.activity.frame",
    "activity_state": "hbv.activity.state",
    "workspace_tabs": "hbv.workspace.tabs",
    "outcomes_tab": "hbv.workspace.outcomes",
    "structure_tab": "hbv.workspace.structure",
    "activity_tab": "hbv.workspace.activity",
}

FONTS = {
    "body": "hbv.font.body",
    "heading": "hbv.font.heading",
    "display": "hbv.font.display",
    "mono": "hbv.font.mono",
}


def _compact_count(value: int) -> str:
    absolute = abs(value)
    for divisor, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if absolute >= divisor:
            scaled = value / divisor
            return f"{scaled:.2f}{suffix}" if abs(scaled) < 10 else f"{scaled:.1f}{suffix}"
    return f"{value:,}"


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _interval(estimate: ProportionEstimate) -> str:
    return f"95% CI {_percent(estimate.low)}–{_percent(estimate.high)}"


def _mix_color(background, foreground, amount: float):
    amount = min(1.0, max(0.0, amount))
    return tuple(
        round(background[index] + (foreground[index] - background[index]) * amount)
        for index in range(4)
    )


def _phase_color(phase: str):
    return {
        "training": PALETTE["ionic"],
        "state_diagnostic": PALETTE["dendrite"],
        "train_evaluation": PALETTE["spike"],
        "calibration": PALETTE["inhibitory"],
    }.get(phase, PALETTE["muted"])


def _font_if_available(path: str, size: int, tag: str) -> bool:
    if not Path(path).is_file():
        return False
    dpg.add_font(path, size, tag=tag)
    return True


def _install_fonts() -> None:
    with dpg.font_registry():
        body = _font_if_available(r"C:\Windows\Fonts\segoeui.ttf", 16, FONTS["body"])
        heading = _font_if_available(
            r"C:\Windows\Fonts\bahnschrift.ttf", 19, FONTS["heading"]
        )
        display = _font_if_available(
            r"C:\Windows\Fonts\bahnschrift.ttf", 29, FONTS["display"]
        )
        mono = _font_if_available(
            r"C:\Windows\Fonts\CascadiaMono.ttf", 14, FONTS["mono"]
        )
    if body:
        dpg.bind_font(FONTS["body"])
    if not heading and body:
        FONTS["heading"] = FONTS["body"]
    if not display and body:
        FONTS["display"] = FONTS["body"]
    if not mono and body:
        FONTS["mono"] = FONTS["body"]


def _install_theme() -> None:
    with dpg.theme(tag="hbv.theme.global"):
        with dpg.theme_component(dpg.mvAll):
            for target, color in (
                (dpg.mvThemeCol_WindowBg, PALETTE["basalt"]),
                (dpg.mvThemeCol_ChildBg, PALETTE["basalt_2"]),
                (dpg.mvThemeCol_PopupBg, PALETTE["basalt_2"]),
                (dpg.mvThemeCol_Text, PALETTE["axon"]),
                (dpg.mvThemeCol_TextDisabled, PALETTE["muted"]),
                (dpg.mvThemeCol_Border, PALETTE["quiet"]),
                (dpg.mvThemeCol_Separator, PALETTE["quiet"]),
                (dpg.mvThemeCol_FrameBg, PALETTE["basalt_3"]),
                (dpg.mvThemeCol_FrameBgHovered, (40, 61, 73, 255)),
                (dpg.mvThemeCol_FrameBgActive, (46, 72, 84, 255)),
                (dpg.mvThemeCol_Button, PALETTE["basalt_3"]),
                (dpg.mvThemeCol_ButtonHovered, (48, 75, 86, 255)),
                (dpg.mvThemeCol_ButtonActive, PALETTE["ionic"]),
                (dpg.mvThemeCol_Header, (37, 59, 70, 255)),
                (dpg.mvThemeCol_HeaderHovered, (47, 75, 86, 255)),
                (dpg.mvThemeCol_HeaderActive, (52, 91, 98, 255)),
                (dpg.mvThemeCol_CheckMark, PALETTE["ionic"]),
                (dpg.mvThemeCol_SliderGrab, PALETTE["ionic"]),
                (dpg.mvThemeCol_TabSelected, (42, 78, 86, 255)),
                (dpg.mvThemeCol_TableHeaderBg, PALETTE["basalt_3"]),
                (dpg.mvThemeCol_TableRowBgAlt, (19, 30, 39, 255)),
            ):
                dpg.add_theme_color(target, color)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 3)
    dpg.bind_theme("hbv.theme.global")

    with dpg.theme(tag="hbv.theme.card"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, PALETTE["basalt_3"])
            dpg.add_theme_color(dpg.mvThemeCol_Border, (49, 72, 84, 255))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)

    for index, color in enumerate(SERIES_COLORS):
        with dpg.theme(tag=f"hbv.theme.series.{index}"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerFill,
                    color,
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerOutline,
                    PALETTE["axon"],
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_Marker,
                    dpg.mvPlotMarker_Circle,
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_MarkerSize,
                    5.5,
                    category=dpg.mvThemeCat_Plots,
                )

    with dpg.theme(tag="hbv.theme.activity.line"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(
                dpg.mvPlotCol_Line,
                PALETTE["spike"],
                category=dpg.mvThemeCat_Plots,
            )

    with dpg.theme(tag="hbv.theme.activity.cursor"):
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(
                dpg.mvPlotCol_MarkerFill,
                PALETTE["ionic"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_MarkerOutline,
                PALETTE["axon"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_Marker,
                dpg.mvPlotMarker_Diamond,
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_MarkerSize,
                8.0,
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_LineWeight,
                1.5,
                category=dpg.mvThemeCat_Plots,
            )


class Workbench:
    def __init__(
        self,
        catalog: ExperimentCatalog,
        initial_experiment: str | None,
        activity_stream: Path | None = None,
        structure_override: NetworkStructure | None = None,
    ) -> None:
        self.catalog = catalog
        available = catalog.experiment_ids
        preferred = initial_experiment or (
            "visualizer_demo" if "visualizer_demo" in available else available[0]
        )
        if preferred not in available:
            raise ValueError(f"unknown initial experiment: {preferred}")
        self.experiment_id = preferred
        self.search = ""
        self.selected_run = catalog.for_experiment(preferred)[0]
        self._external_activity_stream = activity_stream
        self._structure_override = structure_override
        self._loaded_run_id: str | None = None
        self.structure: NetworkStructure | None = None
        self.structure_issue: str | None = None
        self.neighborhood: Neighborhood | None = None
        self.density_overview: DensityOverview | None = None
        self.activity_reader: ActivityStreamReader | None = None
        self.activity_position = 0
        self.activity_playing = False
        self._last_activity_poll = 0.0
        self._last_activity_advance = time.perf_counter()
        self._build()

    def _heading(self, text: str, *, color=None, font: str = "heading", **kwargs):
        item = dpg.add_text(text, color=color or PALETTE["axon"], **kwargs)
        dpg.bind_item_font(item, FONTS[font])
        return item

    def _build(self) -> None:
        with dpg.window(
            tag=TAGS["primary"],
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
        ):
            with dpg.group(horizontal=True):
                with dpg.group(width=490):
                    self._heading("HUMAN BRAIN / NEURAL OBSERVATORY", font="display")
                    dpg.add_text(
                        "Experiment diagnostics for developing intelligent neuronal networks",
                        color=PALETTE["muted"],
                    )
                dpg.add_spacer(width=25)
                dpg.add_text("WORKING MEASUREMENTS", color=PALETTE["ionic"])
                dpg.add_text("•", color=PALETTE["quiet"])
                dpg.add_text("read-only", color=PALETTE["muted"])
                dpg.add_text("•", color=PALETTE["quiet"])
                dpg.add_text("trace diagnostics do not gate outcomes", color=PALETTE["muted"])
            dpg.add_separator()

            with dpg.group(horizontal=True, horizontal_spacing=10):
                with dpg.child_window(width=270, height=-1, border=True):
                    self._build_left_panel()
                with dpg.child_window(
                    width=-370,
                    height=-1,
                    border=False,
                    tag=TAGS["center"],
                ):
                    self._build_center_panel()
                with dpg.child_window(width=350, height=-1, border=True):
                    self._build_right_panel()

        with dpg.item_handler_registry(tag=TAGS["ledger_handlers"]):
            dpg.add_item_resize_handler(callback=self._on_center_resize)
        dpg.bind_item_handler_registry(TAGS["center"], TAGS["ledger_handlers"])
        self.refresh()

    def _build_left_panel(self) -> None:
        self._heading("EXPERIMENT COHORT")
        dpg.add_text(
            "Compare arms that share a protocol and curriculum.",
            color=PALETTE["muted"],
            wrap=235,
        )
        dpg.add_spacer(height=3)
        dpg.add_combo(
            self.catalog.experiment_ids,
            default_value=self.experiment_id,
            tag=TAGS["experiment"],
            width=-1,
            callback=self._on_experiment,
        )
        dpg.add_input_text(
            tag=TAGS["search"],
            hint="Filter arm names",
            width=-1,
            callback=self._on_search,
        )
        dpg.add_spacer(height=8)
        self._heading("PLOT ENCODING")
        dpg.add_text("X dimension", color=PALETTE["muted"])
        dpg.add_combo(
            tuple(DIMENSIONS),
            default_value="Configured neurons",
            tag=TAGS["x_choice"],
            width=-1,
            callback=self._on_plot_control,
        )
        dpg.add_text("Outcome / diagnostic", color=PALETTE["muted"])
        dpg.add_combo(
            tuple(METRICS),
            default_value="Calibration accuracy",
            tag=TAGS["metric_choice"],
            width=-1,
            callback=self._on_plot_control,
        )
        dpg.add_text("Color groups", color=PALETTE["muted"])
        dpg.add_combo(
            tuple(GROUPINGS),
            default_value="Codebook",
            tag=TAGS["group_choice"],
            width=-1,
            callback=self._on_plot_control,
        )
        dpg.add_spacer(height=12)
        self._heading("CATALOG HEALTH")
        dpg.add_text("", tag=TAGS["catalog_info"], color=PALETTE["muted"], wrap=235)
        dpg.add_spacer(height=12)
        self._heading("MEASUREMENT LEVELS")
        self._key("RESULT", "task outcomes and margins", PALETTE["ionic"])
        self._key("STATE", "representation snapshots", PALETTE["dendrite"])
        self._key("TRACE", "spikes and membrane time series", PALETTE["spike"])

    def _key(self, label: str, description: str, color) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text(label, color=color)
            dpg.add_text(description, color=PALETTE["muted"])

    def _build_center_panel(self) -> None:
        with dpg.tab_bar(tag=TAGS["workspace_tabs"]):
            with dpg.tab(label="Outcomes", tag=TAGS["outcomes_tab"]):
                self._build_outcomes_panel()
            with dpg.tab(label="Structure", tag=TAGS["structure_tab"]):
                self._build_structure_panel()
            with dpg.tab(label="Activity", tag=TAGS["activity_tab"]):
                self._build_activity_panel()

    def _build_outcomes_panel(self) -> None:
        dpg.add_group(tag=TAGS["cards"], horizontal=True, horizontal_spacing=8)
        dpg.add_spacer(height=3)
        self._heading("POPULATION LEDGER")
        dpg.add_text(
            "Logarithmic scale context, enabled E/I composition, and the current 2% concurrency reference.",
            color=PALETTE["muted"],
        )
        dpg.add_drawlist(width=760, height=150, tag=TAGS["ledger"])
        dpg.add_spacer(height=2)
        with dpg.plot(
            tag=TAGS["plot"],
            height=290,
            width=-1,
            no_title=True,
            crosshairs=True,
            no_menus=False,
        ):
            dpg.add_plot_axis(dpg.mvXAxis, label="Configured neurons", tag=TAGS["x_axis"])
            dpg.add_plot_axis(dpg.mvYAxis, label="Calibration accuracy", tag=TAGS["y_axis"])
        dpg.add_spacer(height=3)
        self._heading("RUN MATRIX")
        with dpg.table(
            tag=TAGS["run_table"],
            height=255,
            width=-1,
            header_row=True,
            row_background=True,
            resizable=True,
            freeze_rows=1,
            scrollY=True,
            clipper=True,
            borders_innerH=True,
        ):
            dpg.add_table_column(label="Arm", init_width_or_weight=2.4)
            dpg.add_table_column(label="N")
            dpg.add_table_column(label="Seed")
            dpg.add_table_column(label="Codebook")
            dpg.add_table_column(label="Readout")
            dpg.add_table_column(label="Train")
            dpg.add_table_column(label="Calib")
            dpg.add_table_column(label="Margin pA")

    def _build_structure_panel(self) -> None:
        self._heading("ADDRESS-SPACE LENS")
        dpg.add_text(
            "Exact outgoing connectivity around one cell; the view stays bounded while the address space does not.",
            color=PALETTE["muted"],
            wrap=820,
        )
        with dpg.group(horizontal=True):
            dpg.add_text("Root GID", color=PALETTE["muted"])
            dpg.add_input_int(
                default_value=0,
                min_value=0,
                min_clamped=True,
                width=170,
                tag=TAGS["structure_gid"],
                callback=self._on_structure_control,
                on_enter=True,
            )
            dpg.add_text("Hops", color=PALETTE["muted"])
            dpg.add_combo(
                ("1", "2", "3"),
                default_value="2",
                width=80,
                tag=TAGS["structure_hops"],
                callback=self._on_structure_control,
            )
            dpg.add_button(label="Inspect", callback=self._on_structure_control)
        dpg.add_text("", tag=TAGS["structure_info"], color=PALETTE["ionic"], wrap=830)
        dpg.add_drawlist(width=760, height=405, tag=TAGS["structure_canvas"])
        self._heading("CONNECTION DENSITY / ADDRESS BINS")
        dpg.add_text("", tag=TAGS["density_info"], color=PALETTE["muted"])
        dpg.add_drawlist(width=760, height=205, tag=TAGS["density_canvas"])

    def _build_activity_panel(self) -> None:
        self._heading("EXPERIMENT REPLAY / TOKEN ACTIVITY")
        dpg.add_text(
            "Full-run, disk-backed replay of exact token frames. State remains sampled at its recorded cadence; no biological state is interpolated.",
            color=PALETTE["muted"],
            wrap=820,
        )
        dpg.add_text("", tag=TAGS["activity_status"], color=PALETTE["ionic"], wrap=830)
        dpg.add_drawlist(width=760, height=34, tag=TAGS["activity_phase_tape"])
        with dpg.plot(
            tag=TAGS["activity_plot"],
            height=260,
            width=-1,
            no_title=True,
            crosshairs=True,
        ):
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Recorded token frame",
                tag=TAGS["activity_x_axis"],
            )
            dpg.add_plot_axis(
                dpg.mvYAxis,
                label="Population firing rate (Hz)",
                tag=TAGS["activity_y_axis"],
            )
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Prev event",
                callback=self._on_activity_event,
                user_data=-1,
            )
            dpg.add_button(
                label="Prev frame",
                callback=self._on_activity_step,
                user_data=-1,
            )
            dpg.add_button(
                label="Play",
                tag=TAGS["activity_play"],
                callback=self._on_activity_play,
            )
            dpg.add_button(
                label="Next frame",
                callback=self._on_activity_step,
                user_data=1,
            )
            dpg.add_button(
                label="Next event",
                callback=self._on_activity_event,
                user_data=1,
            )
            dpg.add_combo(
                tuple(ACTIVITY_EVENT_FILTERS),
                default_value="Any event",
                width=145,
                tag=TAGS["activity_event_filter"],
            )
        with dpg.group(horizontal=True):
            dpg.add_slider_int(
                default_value=0,
                min_value=0,
                max_value=0,
                width=610,
                tag=TAGS["activity_slider"],
                callback=self._on_activity_scrub,
            )
            dpg.add_combo(
                ACTIVITY_SPEEDS,
                default_value="0.01×",
                width=95,
                tag=TAGS["activity_speed"],
                callback=self._on_activity_speed,
            )
        dpg.add_text("", tag=TAGS["activity_event"], color=PALETTE["spike"])
        dpg.add_text("", tag=TAGS["activity_frame"], color=PALETTE["axon"])
        self._heading("SAMPLED NETWORK STATE")
        dpg.add_drawlist(width=760, height=290, tag=TAGS["activity_state"])

    def _build_right_panel(self) -> None:
        self._heading("SELECTED RUN")
        dpg.add_text("", tag=TAGS["status"], color=PALETTE["ionic"], wrap=315)
        dpg.add_separator()
        dpg.add_group(tag=TAGS["details"])

    def _filtered_runs(self) -> tuple[ExperimentRun, ...]:
        runs = self.catalog.for_experiment(self.experiment_id)
        needle = self.search.casefold().strip()
        if needle:
            runs = tuple(run for run in runs if needle in run.arm.casefold())
        return runs

    def refresh(self) -> None:
        runs = self._filtered_runs()
        if runs and self.selected_run not in runs:
            self.selected_run = runs[0]
        self._load_observation_sources()
        dpg.set_value(
            TAGS["catalog_info"],
            f"{len(self.catalog.runs):,} valid ladder runs\n"
            f"{len(self.catalog.issues):,} unsupported or invalid JSON files\n"
            f"{len(runs):,} runs in this view",
        )
        self._refresh_cards()
        self._refresh_plot(runs)
        self._refresh_table(runs)
        self._refresh_details()
        self._refresh_ledger()
        self._refresh_structure()
        self._refresh_activity_timeline()

    def _load_observation_sources(self) -> None:
        if self._loaded_run_id == self.selected_run.run_id:
            return
        self._loaded_run_id = self.selected_run.run_id
        self.structure = None
        self.structure_issue = None
        self.neighborhood = None
        self.density_overview = None
        try:
            self.structure = self._structure_override or open_structure(self.selected_run)
            gid = min(
                int(dpg.get_value(TAGS["structure_gid"])),
                self.structure.info.configured_neurons - 1,
            )
            dpg.set_value(TAGS["structure_gid"], gid)
            dpg.configure_item(
                TAGS["structure_gid"],
                max_value=self.structure.info.configured_neurons - 1,
                max_clamped=True,
            )
            hops = int(dpg.get_value(TAGS["structure_hops"]))
            self.neighborhood = self.structure.neighborhood(gid, hops=hops)
            self.density_overview = self.structure.density_overview()
        except (KeyError, TypeError, ValueError, StructureUnavailable) as error:
            self.structure_issue = str(error)

        activity_path = (
            self._external_activity_stream
            if self._external_activity_stream is not None
            else reader_for_run(self.catalog.root, self.selected_run).path
        )
        self.activity_reader = ActivityStreamReader(activity_path)
        self.activity_reader.poll()
        self.activity_position = max(0, self.activity_reader.total_frames - 1)
        self.activity_playing = False
        self._last_activity_advance = time.perf_counter()

    def _refresh_cards(self) -> None:
        dpg.delete_item(TAGS["cards"], children_only=True)
        run = self.selected_run
        gap = run.outcome.generalization_gap
        cards = (
            (
                "CALIBRATION",
                _percent(run.outcome.calibration.fraction),
                _interval(run.outcome.calibration),
                PALETTE["ionic"],
            ),
            (
                "GENERALIZATION GAP",
                f"{100.0 * gap.difference:+.1f} pp" if gap else "n/a",
                (
                    f"95% CI {100.0 * gap.low:+.1f}–{100.0 * gap.high:+.1f} pp"
                    if gap
                    else "train − calibration"
                ),
                PALETTE["dendrite"],
            ),
            (
                "DECISION MARGIN",
                f"{run.outcome.margin_median_pa:+.3f} pA",
                f"minimum {run.outcome.margin_min_pa:+.3f} pA",
                PALETTE["spike"] if run.outcome.margin_median_pa > 0 else PALETTE["inhibitory"],
            ),
        )
        for label, value, note, color in cards:
            with dpg.child_window(
                width=215,
                height=112,
                parent=TAGS["cards"],
                border=True,
                no_scrollbar=True,
            ) as card:
                dpg.add_text(label, color=PALETTE["muted"])
                self._heading(value, color=color, font="display")
                dpg.add_text(note, color=PALETTE["muted"], wrap=190)
            dpg.bind_item_theme(card, "hbv.theme.card")

    def _refresh_plot(self, runs: tuple[ExperimentRun, ...]) -> None:
        dpg.delete_item(TAGS["y_axis"], children_only=True)
        x_label = dpg.get_value(TAGS["x_choice"])
        metric_label = dpg.get_value(TAGS["metric_choice"])
        group_label = dpg.get_value(TAGS["group_choice"])
        dimension_id = DIMENSIONS[x_label]
        metric_id = METRICS[metric_label]
        group_id = GROUPINGS[group_label]
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for run in runs:
            value = run.metric(metric_id)
            if value is not None and math.isfinite(value):
                grouped[run.group(group_id)].append((run.dimension(dimension_id), value))
        for index, (name, points) in enumerate(sorted(grouped.items())):
            points.sort()
            series = dpg.add_scatter_series(
                [point[0] for point in points],
                [point[1] for point in points],
                label=name,
                parent=TAGS["y_axis"],
            )
            dpg.bind_item_theme(series, f"hbv.theme.series.{index % len(SERIES_COLORS)}")
        dpg.configure_item(
            TAGS["x_axis"],
            label=x_label,
            scale=(dpg.mvPlotScale_Log10 if dimension_id == "neurons" else dpg.mvPlotScale_Linear),
        )
        dpg.configure_item(TAGS["y_axis"], label=metric_label, scale=dpg.mvPlotScale_Linear)
        if grouped:
            dpg.fit_axis_data(TAGS["x_axis"])
            dpg.fit_axis_data(TAGS["y_axis"])

    def _refresh_table(self, runs: tuple[ExperimentRun, ...]) -> None:
        dpg.delete_item(TAGS["run_table"], children_only=True, slot=1)
        for run in runs:
            with dpg.table_row(parent=TAGS["run_table"]):
                dpg.add_selectable(
                    label=run.arm,
                    default_value=run == self.selected_run,
                    span_columns=True,
                    callback=self._on_run_selected,
                    user_data=run.run_id,
                )
                dpg.add_text(_compact_count(run.population.configured_neurons))
                dpg.add_text(str(run.seed))
                dpg.add_text(run.codebook)
                dpg.add_text(run.readout_rule)
                dpg.add_text(_percent(run.outcome.training.fraction))
                dpg.add_text(_percent(run.outcome.calibration.fraction))
                dpg.add_text(f"{run.outcome.margin_median_pa:+.3f}")

    def _refresh_details(self) -> None:
        dpg.delete_item(TAGS["details"], children_only=True)
        run = self.selected_run
        dpg.set_value(TAGS["status"], f"{run.run_id}\n{run.population.model_family} • {run.simulation_backend}")
        with dpg.tab_bar(parent=TAGS["details"], draw_selected_overline=True):
            with dpg.tab(label="Outcome"):
                self._detail_proportion("Training semantic cases", run.outcome.training)
                self._detail_proportion("Calibration semantic cases", run.outcome.calibration)
                self._detail_proportion("Teacher-token replay", run.outcome.replay)
                dpg.add_separator()
                self._detail_value("Median decision margin", f"{run.outcome.margin_median_pa:+.6f} pA")
                self._detail_value("Minimum decision margin", f"{run.outcome.margin_min_pa:+.6f} pA")
                self._detail_value("Decision samples", f"{run.outcome.margin_count:,}")
                self._detail_value("Wall time", f"{run.outcome.wall_seconds:.3f} s")
            with dpg.tab(label="Neuroscience"):
                p = run.population
                self._detail_value("Enabled neurons", f"{p.enabled_neurons:,}")
                self._detail_value("Excitatory / inhibitory", f"{p.excitatory_neurons:,} / {p.inhibitory_neurons:,}")
                ratio = p.excitation_inhibition_ratio
                self._detail_value("E:I population ratio", f"{ratio:.3f}:1" if ratio else "no inhibitory cells")
                self._detail_value("Input assembly sparsity", _percent(p.input_assembly_fraction))
                self._detail_value("Recurrent connection density", _percent(p.recurrent_connection_density))
                self._detail_value("Estimated contacts", f"{p.total_contacts:,}")
                dpg.add_separator()
                self._heading("REPRESENTATION STATE")
                if run.state_separation.available:
                    self._detail_value(
                        "Eligibility separation",
                        self._maybe_number(run.state_separation.eligibility_median_1minuscos),
                    )
                    self._detail_value(
                        "Synaptic-current separation",
                        self._maybe_number(run.state_separation.synaptic_current_median_1minuscos),
                    )
                else:
                    dpg.add_text(
                        "Not recorded. Re-run with --with-state-separation to sample first-decision representations.",
                        color=PALETTE["muted"],
                        wrap=300,
                    )
                dpg.add_separator()
                reader = self.activity_reader
                if reader is not None and reader.header is not None:
                    self._heading("ACTIVITY STREAM")
                    self._detail_value("Buffered token frames", f"{len(reader.frames):,}")
                    self._detail_value(
                        "Live-state sample",
                        f"{len(reader.header.sample_gids):,} cells / every "
                        f"{reader.header.state_stride:,} frames",
                    )
                    dpg.add_text(
                        "Spike identities within integration steps are not recorded; ISI CV, Fano factor, and STTC remain unavailable.",
                        color=PALETTE["muted"],
                        wrap=300,
                    )
                else:
                    self._heading("ACTIVITY TRACE REQUIRED")
                    dpg.add_text(
                        "Use --record-observatory for token spike totals, firing rate, and sampled membrane/current/eligibility state. Spike-identity metrics still require a kernel event recorder.",
                        color=PALETTE["muted"],
                        wrap=300,
                    )
            with dpg.tab(label="Protocol"):
                for label, value in (
                    ("N", f"{run.population.configured_neurons:,}"),
                    ("Seed", str(run.seed)),
                    ("Training tasks", str(run.data_scale)),
                    ("Readout", run.readout_rule),
                    ("Codebook", run.codebook),
                    ("Schedule", run.schedule),
                    ("Supervision", run.supervision),
                    ("Simulation", run.simulation_backend),
                    ("Ridge backend", run.ridge_backend),
                ):
                    self._detail_value(label, value)
                dpg.add_separator()
                self._heading("PROVENANCE")
                self._detail_value("Git", run.git_head[:12])
                self._detail_value("Curriculum", run.curriculum_sha256[:16])
                self._detail_value("File", run.source_path.name)
                if run.extensions:
                    self._detail_value("Extensions", ", ".join(sorted(run.extensions)))
                with dpg.tree_node(label="Effective configuration"):
                    text = json.dumps(run.configuration, sort_keys=True, indent=2)
                    item = dpg.add_input_text(
                        default_value=text,
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=300,
                    )
                    dpg.bind_item_font(item, FONTS["mono"])

    def _detail_proportion(self, label: str, estimate: ProportionEstimate) -> None:
        self._heading(label)
        dpg.add_text(
            f"{estimate.successes:,} / {estimate.total:,}   {_percent(estimate.fraction)}",
            color=PALETTE["ionic"],
        )
        dpg.add_text(_interval(estimate), color=PALETTE["muted"])

    def _detail_value(self, label: str, value: str) -> None:
        dpg.add_text(label, color=PALETTE["muted"])
        item = dpg.add_text(value, color=PALETTE["axon"])
        dpg.bind_item_font(item, FONTS["mono"])

    @staticmethod
    def _maybe_number(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "not recorded"

    def _refresh_ledger(self) -> None:
        dpg.delete_item(TAGS["ledger"], children_only=True, slot=2)
        center_width, _ = dpg.get_item_rect_size(TAGS["center"])
        width = center_width - 18 if center_width >= 318 else 760
        dpg.configure_item(TAGS["ledger"], width=int(width))
        p = self.selected_run.population
        left, right = 25.0, float(width - 25)
        usable = right - left
        log_min, log_max = 2.0, 10.0
        marker_log = min(log_max, max(log_min, math.log10(p.configured_neurons)))
        marker_x = left + usable * (marker_log - log_min) / (log_max - log_min)

        dpg.draw_line((left, 22), (right, 22), color=PALETTE["quiet"], thickness=1, parent=TAGS["ledger"])
        for power in range(2, 11):
            x = left + usable * (power - log_min) / (log_max - log_min)
            dpg.draw_line((x, 17), (x, 28), color=PALETTE["quiet"], parent=TAGS["ledger"])
            dpg.draw_text((x - 8, 31), f"10^{power}", color=PALETTE["muted"], size=11, parent=TAGS["ledger"])
        dpg.draw_triangle(
            (marker_x, 9),
            (marker_x - 6, 16),
            (marker_x + 6, 16),
            color=PALETTE["spike"],
            fill=PALETTE["spike"],
            parent=TAGS["ledger"],
        )
        dpg.draw_text(
            (max(left, marker_x - 35), 0),
            _compact_count(p.configured_neurons),
            color=PALETTE["spike"],
            size=13,
            parent=TAGS["ledger"],
        )

        y0, y1 = 64.0, 94.0
        dpg.draw_rectangle((left, y0), (right, y1), fill=PALETTE["quiet"], color=PALETTE["quiet"], parent=TAGS["ledger"])
        enabled_width = usable * p.enabled_neurons / p.configured_neurons
        excit_width = enabled_width * p.excitatory_neurons / max(1, p.enabled_neurons)
        dpg.draw_rectangle((left, y0), (left + excit_width, y1), fill=PALETTE["ionic"], color=PALETTE["ionic"], parent=TAGS["ledger"])
        dpg.draw_rectangle((left + excit_width, y0), (left + enabled_width, y1), fill=PALETTE["inhibitory"], color=PALETTE["inhibitory"], parent=TAGS["ledger"])
        dpg.draw_text((left, 47), "ENABLED POPULATION / E:I COMPOSITION", color=PALETTE["muted"], size=11, parent=TAGS["ledger"])
        dpg.draw_text((left + 6, 70), f"E  {_compact_count(p.excitatory_neurons)}", color=PALETTE["basalt"], size=12, parent=TAGS["ledger"])
        if p.inhibitory_neurons:
            dpg.draw_text((max(left + 6, left + excit_width + 5), 70), f"I  {_compact_count(p.inhibitory_neurons)}", color=PALETTE["basalt"], size=12, parent=TAGS["ledger"])

        reference_width = max(3.0, enabled_width * 0.02)
        dpg.draw_rectangle((left, 111), (right, 118), fill=PALETTE["quiet"], color=PALETTE["quiet"], parent=TAGS["ledger"])
        dpg.draw_rectangle((left, 111), (left + reference_width, 118), fill=PALETTE["spike"], color=PALETTE["spike"], parent=TAGS["ledger"])
        dpg.draw_text(
            (left, 122),
            f"2% concurrent-activity reference  ≤ {_compact_count(p.concurrent_activity_reference_neurons)} neurons  •  not an observed rate",
            color=PALETTE["muted"],
            size=11,
            parent=TAGS["ledger"],
        )

    def _refresh_structure(self) -> None:
        dpg.delete_item(TAGS["structure_canvas"], children_only=True, slot=2)
        dpg.delete_item(TAGS["density_canvas"], children_only=True, slot=2)
        width = self._center_canvas_width(TAGS["structure_canvas"])
        dpg.configure_item(TAGS["structure_canvas"], width=int(width))
        dpg.configure_item(TAGS["density_canvas"], width=int(width))
        if self.structure is None or self.neighborhood is None:
            issue = self.structure_issue or "No queryable structure is recorded."
            dpg.set_value(TAGS["structure_info"], issue)
            dpg.set_value(TAGS["density_info"], "No density evidence available.")
            dpg.draw_text(
                (25, 35),
                "STRUCTURE UNAVAILABLE",
                color=PALETTE["inhibitory"],
                size=16,
                parent=TAGS["structure_canvas"],
            )
            dpg.draw_text(
                (25, 68),
                issue,
                color=PALETTE["muted"],
                size=12,
                parent=TAGS["structure_canvas"],
            )
            return

        view = self.neighborhood
        info = self.structure.info
        dpg.set_value(
            TAGS["structure_info"],
            f"{info.representation}  •  {len(view.nodes):,} / "
            f"{info.configured_neurons:,} neurons shown  •  {len(view.edges):,} exact edges"
            + (f"  •  {view.omitted_nodes:,}+ nodes omitted by view cap" if not view.complete else ""),
        )
        positions = self._structure_positions(view, width, 405.0)
        nodes_by_gid = {node.gid: node for node in view.nodes}
        for edge in view.edges:
            source = positions.get(edge.source_gid)
            target = positions.get(edge.target_gid)
            if source is None or target is None:
                continue
            role = nodes_by_gid[edge.source_gid].outgoing_role
            color = (
                PALETTE["inhibitory"]
                if role == "inhibitory"
                else PALETTE["quiet"]
            )
            self._draw_directed_edge(source, target, color)
        show_all_labels = len(view.nodes) <= 42
        for node in view.nodes:
            x, y = positions[node.gid]
            color = {
                "inhibitory": PALETTE["inhibitory"],
                "excitatory": PALETTE["ionic"],
                "dormant": PALETTE["muted"],
            }.get(node.outgoing_role, PALETTE["dendrite"])
            radius = 15 if node.gid == view.root_gid else 9
            dpg.draw_circle(
                (x, y),
                radius,
                color=PALETTE["axon"] if node.gid == view.root_gid else color,
                fill=color,
                thickness=2,
                parent=TAGS["structure_canvas"],
            )
            if show_all_labels or node.gid == view.root_gid:
                dpg.draw_text(
                    (x + radius + 3, y - 7),
                    str(node.gid),
                    color=PALETTE["axon"],
                    size=11,
                    parent=TAGS["structure_canvas"],
                )
        dpg.draw_text(
            (18, 378),
            "cyan = excitatory  •  red = inhibitory  •  arrows = exact outgoing contacts",
            color=PALETTE["muted"],
            size=11,
            parent=TAGS["structure_canvas"],
        )
        self._draw_density_overview(width)

    def _structure_positions(
        self,
        view: Neighborhood,
        width: float,
        height: float,
    ) -> dict[int, tuple[float, float]]:
        center = (width / 2.0, height / 2.0 - 8.0)
        by_depth: dict[int, list[int]] = defaultdict(list)
        for gid, depth in view.depth_by_gid.items():
            by_depth[depth].append(gid)
        positions = {view.root_gid: center}
        radius_limit = min(width * 0.43, height * 0.42)
        for depth in range(1, view.hops + 1):
            gids = sorted(by_depth.get(depth, ()))
            if not gids:
                continue
            radius = radius_limit * depth / view.hops
            for index, gid in enumerate(gids):
                angle = -math.pi / 2.0 + 2.0 * math.pi * index / len(gids)
                positions[gid] = (
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                )
        return positions

    def _draw_directed_edge(
        self,
        source: tuple[float, float],
        target: tuple[float, float],
        color,
    ) -> None:
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        length = math.hypot(dx, dy)
        if length < 1.0:
            dpg.draw_circle(
                (source[0] + 13, source[1] - 13),
                11,
                color=color,
                thickness=1,
                parent=TAGS["structure_canvas"],
            )
            return
        unit_x, unit_y = dx / length, dy / length
        start = (source[0] + unit_x * 11, source[1] + unit_y * 11)
        end = (target[0] - unit_x * 12, target[1] - unit_y * 12)
        dpg.draw_line(
            start,
            end,
            color=color,
            thickness=1,
            parent=TAGS["structure_canvas"],
        )
        perpendicular = (-unit_y, unit_x)
        arrow_back = (end[0] - unit_x * 7, end[1] - unit_y * 7)
        dpg.draw_triangle(
            end,
            (
                arrow_back[0] + perpendicular[0] * 3,
                arrow_back[1] + perpendicular[1] * 3,
            ),
            (
                arrow_back[0] - perpendicular[0] * 3,
                arrow_back[1] - perpendicular[1] * 3,
            ),
            color=color,
            fill=color,
            parent=TAGS["structure_canvas"],
        )

    def _draw_density_overview(self, width: float) -> None:
        overview = self.density_overview
        if overview is None:
            dpg.set_value(TAGS["density_info"], "No density evidence available.")
            return
        qualifier = "all sources / exact" if overview.exact else "sampled sources"
        dpg.set_value(
            TAGS["density_info"],
            f"{overview.sampled_sources:,} {qualifier}  •  rows = source address bins  •  columns = target address bins",
        )
        bin_count = len(overview.bins)
        left, top, bottom = 34.0, 8.0, 25.0
        usable_width = width - left - 10.0
        usable_height = 205.0 - top - bottom
        cell_width = usable_width / bin_count
        cell_height = usable_height / bin_count
        maximum = max(max(row) for row in overview.bins) or 1
        denominator = math.log1p(maximum)
        for row_index, row in enumerate(overview.bins):
            for column_index, count in enumerate(row):
                amount = math.log1p(count) / denominator
                color = _mix_color(PALETTE["basalt_3"], PALETTE["ionic"], amount)
                x0 = left + column_index * cell_width
                y0 = top + row_index * cell_height
                dpg.draw_rectangle(
                    (x0, y0),
                    (x0 + cell_width + 0.5, y0 + cell_height + 0.5),
                    color=color,
                    fill=color,
                    parent=TAGS["density_canvas"],
                )
        dpg.draw_text(
            (left, 183),
            "target GID  0 → N",
            color=PALETTE["muted"],
            size=10,
            parent=TAGS["density_canvas"],
        )
        dpg.draw_text(
            (3, 8),
            "S\nR\nC",
            color=PALETTE["muted"],
            size=10,
            parent=TAGS["density_canvas"],
        )

    def _refresh_activity_timeline(self) -> None:
        reader = self.activity_reader
        dpg.delete_item(TAGS["activity_y_axis"], children_only=True, slot=1)
        dpg.delete_item(TAGS["activity_phase_tape"], children_only=True, slot=2)
        if reader is None:
            return
        if reader.header is None:
            if reader.issue is not None:
                status = reader.issue
            elif self._external_activity_stream is not None:
                status = f"Waiting for {reader.path}"
            else:
                status = "No activity stream for this run. Add --record-observatory to a new experiment."
            dpg.set_value(TAGS["activity_status"], status)
            dpg.set_value(TAGS["activity_frame"], "No recorded frames")
            dpg.set_value(TAGS["activity_event"], "")
            self._draw_activity_empty(status)
            return
        if not reader.caught_up:
            state = (
                f"indexing {reader.indexed_bytes / 1_048_576:.1f} / "
                f"{reader.source_size / 1_048_576:.1f} MiB"
            )
        else:
            state = "complete" if reader.complete else "live / awaiting more frames"
        dpg.set_value(
            TAGS["activity_status"],
            f"{reader.header.run_id}  •  {reader.total_frames:,} indexed frames  •  {state}  •  "
            f"{len(reader.header.sample_gids):,} sampled cells every "
            f"{reader.header.state_stride:,} frames",
        )
        if reader.total_frames == 0:
            dpg.set_value(TAGS["activity_event"], "")
            dpg.set_value(TAGS["activity_frame"], "Waiting for the first recorded frame")
            dpg.configure_item(TAGS["activity_slider"], max_value=0)
            dpg.set_value(TAGS["activity_slider"], 0)
            self._draw_activity_empty(
                "The stream header is ready; no token frame has arrived yet."
            )
            return
        if reader.total_frames:
            frame_indices, firing_rates = reader.timeline_points()
            series = dpg.add_line_series(
                frame_indices,
                firing_rates,
                parent=TAGS["activity_y_axis"],
            )
            dpg.bind_item_theme(series, "hbv.theme.activity.line")
            cursor = dpg.add_scatter_series(
                [],
                [],
                tag=TAGS["activity_cursor"],
                parent=TAGS["activity_y_axis"],
            )
            dpg.bind_item_theme(cursor, "hbv.theme.activity.cursor")
            dpg.fit_axis_data(TAGS["activity_x_axis"])
            dpg.fit_axis_data(TAGS["activity_y_axis"])
        self.activity_position = min(
            self.activity_position,
            max(0, reader.total_frames - 1),
        )
        dpg.configure_item(
            TAGS["activity_slider"],
            max_value=max(0, reader.total_frames - 1),
        )
        dpg.set_value(TAGS["activity_slider"], self.activity_position)
        self._draw_phase_tape(reader.phase_runs, reader.total_frames)
        self._refresh_activity_frame()

    def _draw_phase_tape(
        self,
        phase_runs: tuple[ActivityPhaseRun, ...],
        frame_count: int,
    ) -> None:
        width = self._center_canvas_width(TAGS["activity_phase_tape"])
        dpg.configure_item(TAGS["activity_phase_tape"], width=int(width))
        if not phase_runs or frame_count == 0:
            return
        left, usable = 6.0, width - 12.0
        for phase_run in phase_runs:
            x0 = left + usable * phase_run.start_position / frame_count
            x1 = left + usable * phase_run.end_position / frame_count
            color = _phase_color(phase_run.phase)
            dpg.draw_rectangle(
                (x0, 6),
                (x1, 27),
                color=color,
                fill=color,
                parent=TAGS["activity_phase_tape"],
            )
            if x1 - x0 > 85:
                dpg.draw_text(
                    (x0 + 5, 8),
                    phase_run.phase.replace("_", " "),
                    color=PALETTE["basalt"],
                    size=10,
                    parent=TAGS["activity_phase_tape"],
                )

    def _refresh_activity_frame(self) -> None:
        reader = self.activity_reader
        if reader is None or reader.header is None or reader.total_frames == 0:
            return
        self.activity_position = min(self.activity_position, reader.total_frames - 1)
        frame = reader.frame_at(self.activity_position)
        state_frame = reader.state_frame_at_or_before(self.activity_position)
        event = reader.event_at(self.activity_position)
        state_note = (
            f"state frame {state_frame.index:,} "
            f"(frame-index delta {frame.index - state_frame.index:,})"
            if state_frame is not None
            else "state not sampled yet"
        )
        dpg.set_value(
            TAGS["activity_event"],
            f"EVENT  •  {event.label}" if event is not None else "ordinary token boundary",
        )
        dpg.set_value(
            TAGS["activity_frame"],
            f"position {self.activity_position + 1:,} / {reader.total_frames:,}  •  "
            f"frame {frame.index:,}  •  t={frame.biological_time_ms:,.3f} ms  •  "
            f"{frame.phase.replace('_', ' ')}  •  token {frame.token_id}  •  "
            f"{frame.spike_count:,} spikes  •  {frame.population_firing_rate_hz:.3f} Hz  •  "
            f"density {100.0 * frame.neuron_step_activity_density:.4f}%  •  {state_note}",
        )
        dpg.set_value(TAGS["activity_slider"], self.activity_position)
        if dpg.does_item_exist(TAGS["activity_cursor"]):
            dpg.set_value(
                TAGS["activity_cursor"],
                [[float(frame.index)], [frame.population_firing_rate_hz]],
            )
        if state_frame is not None:
            self._draw_activity_state(state_frame)
        else:
            self._draw_activity_empty("No sampled state has arrived yet.")

    def _draw_activity_empty(self, message: str) -> None:
        dpg.delete_item(TAGS["activity_state"], children_only=True, slot=2)
        width = self._center_canvas_width(TAGS["activity_state"])
        dpg.configure_item(TAGS["activity_state"], width=int(width))
        dpg.draw_text(
            (24, 35),
            "TRACE UNAVAILABLE",
            color=PALETTE["spike"],
            size=16,
            parent=TAGS["activity_state"],
        )
        dpg.draw_text(
            (24, 68),
            message,
            color=PALETTE["muted"],
            size=11,
            parent=TAGS["activity_state"],
        )

    def _draw_activity_state(self, frame: ActivityFrame) -> None:
        reader = self.activity_reader
        if reader is None or reader.header is None or not frame.has_state:
            return
        assert frame.voltage_relative_mv is not None
        assert frame.synaptic_current_pa is not None
        assert frame.eligibility is not None
        dpg.delete_item(TAGS["activity_state"], children_only=True, slot=2)
        width = self._center_canvas_width(TAGS["activity_state"])
        dpg.configure_item(TAGS["activity_state"], width=int(width))
        channels = (
            ("V relative (mV)", frame.voltage_relative_mv, PALETTE["ionic"]),
            ("I synaptic (pA)", frame.synaptic_current_pa, PALETTE["spike"]),
            ("Eligibility", frame.eligibility, PALETTE["dendrite"]),
        )
        left, right = 118.0, 12.0
        usable = width - left - right
        display_count = min(len(reader.header.sample_gids), max(1, int(usable // 3)))
        for row, (label, values, foreground) in enumerate(channels):
            y0 = 18.0 + row * 78.0
            bins = self._mean_bins(values, display_count)
            minimum, maximum = min(bins), max(bins)
            span = maximum - minimum
            cell_width = usable / len(bins)
            dpg.draw_text(
                (8, y0 + 5),
                label,
                color=PALETTE["muted"],
                size=11,
                parent=TAGS["activity_state"],
            )
            dpg.draw_text(
                (8, y0 + 25),
                f"{minimum:.3g} → {maximum:.3g}",
                color=PALETTE["axon"],
                size=10,
                parent=TAGS["activity_state"],
            )
            for index, value in enumerate(bins):
                amount = 0.5 if span == 0.0 else (value - minimum) / span
                color = _mix_color(PALETTE["basalt_3"], foreground, amount)
                x0 = left + index * cell_width
                dpg.draw_rectangle(
                    (x0, y0),
                    (x0 + cell_width + 0.4, y0 + 55),
                    color=color,
                    fill=color,
                    parent=TAGS["activity_state"],
                )
        first_gid, last_gid = reader.header.sample_gids[0], reader.header.sample_gids[-1]
        suffix = (
            f"{len(reader.header.sample_gids):,} exact samples"
            if display_count == len(reader.header.sample_gids)
            else f"{len(reader.header.sample_gids):,} samples averaged into {display_count:,} display bins"
        )
        dpg.draw_text(
            (left, 258),
            f"GID {first_gid:,} → {last_gid:,}  •  {suffix}",
            color=PALETTE["muted"],
            size=10,
            parent=TAGS["activity_state"],
        )

    @staticmethod
    def _mean_bins(values: tuple[float, ...], bin_count: int) -> tuple[float, ...]:
        if bin_count >= len(values):
            return values
        bins = []
        for index in range(bin_count):
            start = index * len(values) // bin_count
            end = (index + 1) * len(values) // bin_count
            bins.append(sum(values[start:end]) / (end - start))
        return tuple(bins)

    def _center_canvas_width(self, tag: str) -> float:
        center_width, _ = dpg.get_item_rect_size(TAGS["center"])
        width = center_width - 18 if center_width >= 318 else 760
        item_width, _ = dpg.get_item_rect_size(tag)
        return width if item_width < 300 else width

    def tick(self) -> None:
        reader = self.activity_reader
        if reader is None:
            return
        now = time.perf_counter()
        if now - self._last_activity_poll >= 0.25:
            previous_count = reader.total_frames
            previous_complete = reader.complete
            previous_header = reader.header
            previous_issue = reader.issue
            was_at_end = self.activity_position >= max(0, previous_count - 1)
            added = reader.poll()
            self._last_activity_poll = now
            if added or (
                reader.complete != previous_complete
                or reader.header != previous_header
                or reader.issue != previous_issue
                or not reader.caught_up
            ):
                if was_at_end and not self.activity_playing and reader.total_frames:
                    self.activity_position = reader.total_frames - 1
                self._refresh_activity_timeline()
        if (
            not self.activity_playing
            or reader.header is None
            or reader.total_frames == 0
        ):
            return
        speed = float(str(dpg.get_value(TAGS["activity_speed"])).replace("×", ""))
        seconds_per_frame = reader.header.token_interval_ms / 1_000.0 / speed
        elapsed = now - self._last_activity_advance
        advance = int(elapsed / seconds_per_frame)
        if advance < 1:
            return
        self._last_activity_advance += advance * seconds_per_frame
        last_position = reader.total_frames - 1
        self.activity_position = min(last_position, self.activity_position + advance)
        if self.activity_position == last_position and reader.complete:
            self.activity_playing = False
            dpg.configure_item(TAGS["activity_play"], label="Play")
        self._refresh_activity_frame()

    def _on_experiment(self, sender, app_data, user_data) -> None:
        self.experiment_id = str(app_data)
        runs = self.catalog.for_experiment(self.experiment_id)
        self.selected_run = runs[0]
        self.refresh()

    def _on_search(self, sender, app_data, user_data) -> None:
        self.search = str(app_data)
        self.refresh()

    def _on_plot_control(self, sender, app_data, user_data) -> None:
        self._refresh_plot(self._filtered_runs())

    def _on_run_selected(self, sender, app_data, user_data) -> None:
        run = next((candidate for candidate in self._filtered_runs() if candidate.run_id == user_data), None)
        if run is not None:
            self.selected_run = run
            self.refresh()

    def _on_structure_control(self, sender, app_data, user_data) -> None:
        if self.structure is None:
            return
        gid = int(dpg.get_value(TAGS["structure_gid"]))
        hops = int(dpg.get_value(TAGS["structure_hops"]))
        try:
            self.neighborhood = self.structure.neighborhood(gid, hops=hops)
            self.structure_issue = None
        except ValueError as error:
            self.neighborhood = None
            self.structure_issue = str(error)
        self._refresh_structure()

    def _on_activity_play(self, sender, app_data, user_data) -> None:
        reader = self.activity_reader
        if reader is None or reader.total_frames == 0:
            return
        if self.activity_playing:
            self._pause_activity()
            return
        if reader.complete and self.activity_position >= reader.total_frames - 1:
            self.activity_position = 0
        self.activity_playing = True
        self._last_activity_advance = time.perf_counter()
        dpg.configure_item(TAGS["activity_play"], label="Pause")
        self._refresh_activity_frame()

    def _on_activity_scrub(self, sender, app_data, user_data) -> None:
        self._set_activity_position(int(app_data))

    def _on_activity_step(self, sender, app_data, user_data) -> None:
        self._set_activity_position(self.activity_position + int(user_data))

    def _on_activity_event(self, sender, app_data, user_data) -> None:
        reader = self.activity_reader
        if reader is None or reader.total_frames == 0:
            return
        label = str(dpg.get_value(TAGS["activity_event_filter"]))
        kind = ACTIVITY_EVENT_FILTERS.get(label)
        event = (
            reader.previous_event(self.activity_position, kind)
            if int(user_data) < 0
            else reader.next_event(self.activity_position, kind)
        )
        if event is not None:
            self._set_activity_position(event.position)

    def _on_activity_speed(self, sender, app_data, user_data) -> None:
        self._last_activity_advance = time.perf_counter()

    def _pause_activity(self) -> None:
        self.activity_playing = False
        dpg.configure_item(TAGS["activity_play"], label="Play")

    def _set_activity_position(self, position: int) -> None:
        reader = self.activity_reader
        if reader is None or reader.total_frames == 0:
            return
        self._pause_activity()
        self.activity_position = min(max(0, position), reader.total_frames - 1)
        self._refresh_activity_frame()

    def _on_center_resize(self, sender, app_data, user_data) -> None:
        self._refresh_ledger()
        self._refresh_structure()
        if self.activity_reader is not None and self.activity_reader.header is not None:
            self._draw_phase_tape(
                self.activity_reader.phase_runs,
                self.activity_reader.total_frames,
            )
            self._refresh_activity_frame()


def run_workbench(
    catalog: ExperimentCatalog,
    *,
    initial_experiment: str | None = None,
    activity_stream: Path | None = None,
    structure_override: NetworkStructure | None = None,
) -> None:
    dpg.create_context()
    try:
        dpg.configure_app(docking=False, docking_space=False, init_file="")
        _install_fonts()
        _install_theme()
        workbench = Workbench(
            catalog,
            initial_experiment,
            activity_stream,
            structure_override,
        )
        dpg.create_viewport(
            title="Human Brain — Neural Observatory",
            width=1600,
            height=980,
            min_width=1180,
            min_height=760,
            clear_color=PALETTE["basalt"],
            vsync=True,
        )
        dpg.setup_dearpygui()
        dpg.set_primary_window(TAGS["primary"], True)
        dpg.show_viewport()
        first_frame = True
        while dpg.is_dearpygui_running():
            workbench.tick()
            dpg.render_dearpygui_frame()
            if first_frame:
                workbench._on_center_resize(None, None, None)
                first_frame = False
    finally:
        dpg.destroy_context()
