"""
mapmaker.py
~~~~~~~~~~~~

This module implements a self‑contained helper class for generating
choropleth maps from ranked GABRIEL outputs.  The ``MapMaker`` can
produce county‑, state‑ or country‑level maps depending on the
``map_type`` parameter, and it optionally normalises values as
z‑scores.  All mapping logic resides in this file, replacing the need
for the separate ``create_county_choropleth`` function.
"""

from __future__ import annotations

import os
import json
import re
import requests
from typing import Callable, Iterable, Optional, Sequence, List

import numpy as np
import pandas as pd
import plotly.express as px


class MapMaker:
    """Utility for generating geographic choropleth maps from a data frame.

    The input data frame should contain at least one column of numeric
    scores and one or more columns identifying the geographic unit
    (county FIPS codes, two‑letter US state abbreviations or ISO‑3
    country codes).  Individual maps are rendered using Plotly and
    written to ``save_dir`` with names derived from the value column.

    Parameters
    ----------
    df:
        DataFrame containing the data to plot.  Each row should
        correspond to a geographic region.
    fips_col:
        Name of the column containing five‑digit county FIPS codes.
    state_col:
        Name of the column containing two‑letter US state abbreviations.
    country_col:
        Name of the column containing ISO‑3 country codes.
    save_dir:
        Directory to which map files will be written.  If ``None``,
        a ``maps`` subdirectory in the current working directory is used.
    z_score:
        Whether to convert values to z‑scores before plotting.
    color_scale:
        Name of the Plotly colour scale to apply.  Defaults to
        ``"RdBu"`` (diverging) when z‑scores are enabled and
        ``"Viridis"`` otherwise.
    map_type:
        Determines the map produced: ``"county"``, ``"state"``
        or ``"country"`` (with ``"global"`` as an alias).
    static_formats:
        Optional static image formats to export alongside the HTML output.
        Defaults to ``("png",)``; set to ``None`` or an empty list to skip
        static exports.
    html_output:
        Whether to write interactive HTML maps. Disable this when producing
        static-only map folders.
    map_width, map_height:
        Pixel dimensions used for static image exports and HTML layout sizing.
    static_scale:
        Scale multiplier passed to Plotly static image export.
    color_range:
        Optional fixed numeric color range, such as ``(0, 100)`` for raw ratings.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        fips_col: Optional[str] = None,
        state_col: Optional[str] = None,
        country_col: Optional[str] = None,
        save_dir: Optional[str] = None,
        z_score: bool = True,
        color_scale: str = "RdBu",
        map_type: str = "county",
        static_formats: Optional[Sequence[str] | str] = ("png",),
        html_output: bool = True,
        map_width: int = 1200,
        map_height: int = 780,
        static_scale: int = 3,
        color_range: Optional[Sequence[float]] = None,
    ) -> None:
        self.df = df.copy()
        self.fips_col = fips_col
        self.state_col = state_col
        self.country_col = country_col

        # normalise map_type and validate
        map_type = map_type.lower()
        if map_type not in {"county", "state", "country", "global"}:
            raise ValueError(
                "map_type must be one of 'county', 'state', 'country' or 'global'"
            )
        self.map_type = "country" if map_type == "global" else map_type

        # choose save directory
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "maps")
        save_dir = os.path.expandvars(os.path.expanduser(save_dir))
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.z_score = z_score
        self.color_scale = color_scale
        self.static_formats = self._normalize_static_formats(static_formats)
        self.html_output = html_output
        self.map_width = int(map_width)
        self.map_height = int(map_height)
        self.static_scale = int(static_scale)
        self.color_range = self._normalize_color_range(color_range)

    @staticmethod
    def _normalize_static_formats(static_formats: Optional[Sequence[str] | str]) -> List[str]:
        """Normalize requested static output formats (e.g., png, pdf)."""
        if static_formats is None:
            return []
        if isinstance(static_formats, str):
            formats = [static_formats]
        else:
            formats = list(static_formats)
        cleaned: List[str] = []
        for fmt in formats:
            normalized = str(fmt).strip().lower().lstrip(".")
            if normalized:
                cleaned.append(normalized)
        return cleaned

    @staticmethod
    def _normalize_color_range(color_range: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
        """Normalize an optional fixed color range."""
        if color_range is None:
            return None
        values = list(color_range)
        if len(values) != 2:
            raise ValueError("color_range must contain exactly two numeric values")
        low, high = float(values[0]), float(values[1])
        if not low < high:
            raise ValueError("color_range lower bound must be less than upper bound")
        return low, high

    @staticmethod
    def _label_from_column(value: str) -> str:
        """Return a readable title for a column or map label."""
        label = re.sub(r"[_\s]+", " ", str(value)).strip()
        if label.lower().startswith("county map for "):
            label = label[15:].strip()
        if label.lower().startswith("state map for "):
            label = label[14:].strip()
        if label.lower().startswith("country map for "):
            label = label[16:].strip()
        return label[:1].upper() + label[1:]

    @staticmethod
    def _safe_file_stem(value: str) -> str:
        """Create stable, readable file stems from attribute names."""
        stem = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
        stem = re.sub(r"_+", "_", stem).strip("_")
        return stem or "value"

    def _style_choropleth(
        self,
        fig,
        *,
        title: str,
        value_col: str,
        z_scored: bool,
    ) -> None:
        """Apply a cleaner shared visual style to generated maps."""
        display_title = self._label_from_column(title or value_col)
        color_title = "z score" if z_scored else "rating"
        if z_scored:
            subtitle = "County-level z-score rating"
        elif self.color_range == (0.0, 100.0):
            subtitle = "County-level rating (0-100 scale)"
        else:
            subtitle = "County-level rating"
        fig.update_traces(
            marker_line_width=0.08,
            marker_line_color="rgba(255,255,255,0.55)",
        )
        fig.update_geos(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="#eef4fb",
            landcolor="#f7f9fc",
            showlakes=True,
            showland=True,
            showframe=False,
            showcountries=False,
            subunitcolor="#d7dee8",
        )
        colorbar = {
            "title": {
                "text": color_title,
                "side": "top",
                "font": {"size": 16, "color": "#1f2937"},
            },
            "tickfont": {"size": 13, "color": "#334155"},
            "len": 0.76,
            "thickness": 18,
            "outlinewidth": 0,
            "x": 0.94,
        }
        if self.color_range is not None:
            low, high = self.color_range
            mid = (low + high) / 2
            colorbar["tickvals"] = [low, (low + mid) / 2, mid, (mid + high) / 2, high]
            colorbar["ticktext"] = [f"{value:g}" for value in colorbar["tickvals"]]
        fig.update_coloraxes(colorbar=colorbar)
        fig.update_layout(
            width=self.map_width,
            height=self.map_height,
            margin={"l": 20, "r": 34, "t": 150, "b": 24},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={
                "family": "Inter, Avenir Next, Helvetica Neue, Arial, sans-serif",
                "size": 15,
                "color": "#1f2937",
            },
            title=None,
            annotations=[
                {
                    "text": f"<b>{display_title}</b>",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.22,
                    "yanchor": "top",
                    "showarrow": False,
                    "font": {"size": 30, "color": "#111827"},
                },
                {
                    "text": subtitle,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.115,
                    "yanchor": "top",
                    "showarrow": False,
                    "font": {"size": 15, "color": "#64748b"},
                },
            ],
        )

    def _compute_zscore(self, values: np.ndarray) -> np.ndarray:
        """Compute z‑scores with safe handling of NaNs and constant arrays."""
        vals = values.astype(float)
        if len(vals) > 1 and np.nanstd(vals) > 0:
            return (vals - np.nanmean(vals)) / np.nanstd(vals)
        return np.zeros_like(vals)

    def _create_state_choropleth(
        self,
        df: pd.DataFrame,
        state_col: str,
        value_col: str,
        title: str,
        save_path: str,
    ) -> None:
        """Create and save a state‑level choropleth."""
        plot_col = value_col
        colour_scale = self.color_scale
        df_local = df.copy()
        if self.z_score:
            zs = self._compute_zscore(df_local[value_col].values)
            plot_col = f"_zscore_{value_col}"
            df_local[plot_col] = zs
            colour_scale = "RdBu" if self.color_scale == "RdBu" else "PuOr"
        fig = px.choropleth(
            df_local,
            locations=state_col,
            locationmode="USA-states",
            color=plot_col,
            color_continuous_scale=colour_scale,
            scope="usa",
            range_color=self.color_range,
            hover_data={state_col: True, value_col: True},
            labels={
                state_col: "state",
                value_col: self._label_from_column(value_col),
                plot_col: "z score" if self.z_score else "rating",
            },
        )
        if self.z_score:
            fig.update_coloraxes(cmid=0)
        self._style_choropleth(
            fig,
            title=title,
            value_col=value_col,
            z_scored=self.z_score,
        )
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}:
            fig.write_image(save_path, scale=self.static_scale)
        else:
            fig.write_html(save_path)

    def _create_country_choropleth(
        self,
        df: pd.DataFrame,
        country_col: str,
        value_col: str,
        title: str,
        save_path: str,
    ) -> None:
        """Create and save a global choropleth using ISO‑3 codes."""
        plot_col = value_col
        colour_scale = self.color_scale
        df_local = df.copy()
        if self.z_score:
            zs = self._compute_zscore(df_local[value_col].values)
            plot_col = f"_zscore_{value_col}"
            df_local[plot_col] = zs
            colour_scale = "RdBu" if self.color_scale == "RdBu" else "PuOr"
        fig = px.choropleth(
            df_local,
            locations=country_col,
            locationmode="ISO-3",
            color=plot_col,
            color_continuous_scale=colour_scale,
            scope="world",
            range_color=self.color_range,
            hover_data={country_col: True, value_col: True},
            labels={
                country_col: "country",
                value_col: self._label_from_column(value_col),
                plot_col: "z score" if self.z_score else "rating",
            },
        )
        if self.z_score:
            fig.update_coloraxes(cmid=0)
        self._style_choropleth(
            fig,
            title=title,
            value_col=value_col,
            z_scored=self.z_score,
        )
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}:
            fig.write_image(save_path, scale=self.static_scale)
        else:
            fig.write_html(save_path)

    def _create_county_choropleth(
        self,
        df: pd.DataFrame,
        fips_col: str,
        value_col: str,
        title: str,
        save_path: str,
    ) -> None:
        """Create and save a county‑level choropleth with FIPS codes.

        This method inlines the logic of the old ``create_county_choropleth``
        function to avoid external dependencies.  It downloads a GeoJSON of
        U.S. counties on first use and caches it in ``~/.cache/county_geo.json``.
        """
        # pad FIPS codes to five digits
        df_local = df.copy()
        df_local[fips_col] = df_local[fips_col].astype(str).str.zfill(5)

        # find a county name column for hover text
        county_col = None
        for cand in ["county", "County", "region", "Region"]:
            if cand in df_local.columns:
                county_col = cand
                break
        if county_col is None:
            county_col = "_county_name"
            df_local[county_col] = ""

        # load or download county GeoJSON
        geojson_url = (
            "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        )
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "county_geo.json")
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            resp = requests.get(geojson_url, timeout=30)
            resp.raise_for_status()
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
        with open(cache_path, encoding="utf-8") as f:
            counties = json.load(f)

        # prepare value column (with optional z‑score)
        plot_col = value_col
        colour_scale = self.color_scale
        if self.z_score:
            zs = self._compute_zscore(df_local[value_col].values)
            plot_col = f"_zscore_{value_col}"
            df_local[plot_col] = zs
            colour_scale = "RdBu" if self.color_scale == "RdBu" else "PuOr"

        hover_data = {county_col: True, fips_col: True, value_col: True}
        fig = px.choropleth(
            df_local,
            geojson=counties,
            locations=fips_col,
            color=plot_col,
            color_continuous_scale=colour_scale,
            scope="usa",
            range_color=self.color_range,
            hover_data=hover_data,
            hover_name=county_col,
            labels={
                county_col: "county",
                fips_col: "fips",
                value_col: self._label_from_column(value_col),
                plot_col: "z score" if self.z_score else "rating",
            },
        )
        if self.z_score:
            fig.update_coloraxes(cmid=0)
        self._style_choropleth(
            fig,
            title=title,
            value_col=value_col,
            z_scored=self.z_score,
        )
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}:
            fig.write_image(save_path, scale=self.static_scale)
        else:
            fig.write_html(save_path)

    def _save_static_copy(self, save_path: str, writer: Callable[[str], None]) -> None:
        """Attempt to save a static image, warning if dependencies are missing."""
        try:
            writer(save_path)
        except Exception as exc:
            print(
                f"Warning: Unable to save static map '{save_path}'. "
                f"Install plotly kaleido for static exports. ({exc})"
            )

    def make_maps(self, value_cols: Iterable[str]) -> None:
        """Generate and save maps for each specified numeric column.

        The map type is determined by ``self.map_type``: ``"county"``
        uses FIPS codes; ``"state"`` uses two‑letter abbreviations; and
        ``"country"`` uses ISO‑3 codes.
        """
        for value_col in value_cols:
            if self.map_type == "county":
                if not self.fips_col:
                    raise ValueError("fips_col must be provided for county maps")
                base_name = f"county_map_{self._safe_file_stem(value_col)}"
                if self.html_output:
                    save_path = os.path.join(self.save_dir, f"{base_name}.html")
                    self._create_county_choropleth(
                        self.df,
                        self.fips_col,
                        value_col,
                        title=self._label_from_column(value_col),
                        save_path=save_path,
                    )
                for fmt in self.static_formats:
                    static_path = os.path.join(self.save_dir, f"{base_name}.{fmt}")
                    self._save_static_copy(
                        static_path,
                        lambda path=static_path: self._create_county_choropleth(
                            self.df,
                            self.fips_col,
                            value_col,
                            title=self._label_from_column(value_col),
                            save_path=path,
                        ),
                    )
            elif self.map_type == "state":
                if not self.state_col:
                    raise ValueError("state_col must be provided for state maps")
                base_name = f"state_map_{self._safe_file_stem(value_col)}"
                if self.html_output:
                    save_path = os.path.join(self.save_dir, f"{base_name}.html")
                    self._create_state_choropleth(
                        self.df,
                        self.state_col,
                        value_col,
                        title=self._label_from_column(value_col),
                        save_path=save_path,
                    )
                for fmt in self.static_formats:
                    static_path = os.path.join(self.save_dir, f"{base_name}.{fmt}")
                    self._save_static_copy(
                        static_path,
                        lambda path=static_path: self._create_state_choropleth(
                            self.df,
                            self.state_col,
                            value_col,
                            title=self._label_from_column(value_col),
                            save_path=path,
                        ),
                    )
            elif self.map_type == "country":
                if not self.country_col:
                    raise ValueError("country_col must be provided for country maps")
                base_name = f"country_map_{self._safe_file_stem(value_col)}"
                if self.html_output:
                    save_path = os.path.join(self.save_dir, f"{base_name}.html")
                    self._create_country_choropleth(
                        self.df,
                        self.country_col,
                        value_col,
                        title=self._label_from_column(value_col),
                        save_path=save_path,
                    )
                for fmt in self.static_formats:
                    static_path = os.path.join(self.save_dir, f"{base_name}.{fmt}")
                    self._save_static_copy(
                        static_path,
                        lambda path=static_path: self._create_country_choropleth(
                            self.df,
                            self.country_col,
                            value_col,
                            title=self._label_from_column(value_col),
                            save_path=path,
                        ),
                    )
            else:
                # should not happen due to validation in __init__
                raise ValueError(f"Unsupported map type: {self.map_type}")


def create_county_choropleth(
    df: pd.DataFrame,
    *,
    fips_col: str,
    value_col: str,
    title: str,
    save_path: str,
    z_score: bool = True,
) -> None:
    """Backward compatible helper to generate a county-level choropleth.

    This thin wrapper instantiates :class:`MapMaker` and delegates to its
    internal implementation.  It mirrors the signature of the legacy
    ``create_county_choropleth`` function used elsewhere in the codebase.
    """

    mm = MapMaker(df, fips_col=fips_col, z_score=z_score, save_dir=None, map_type="county")
    mm._create_county_choropleth(df, fips_col, value_col, title, save_path)
