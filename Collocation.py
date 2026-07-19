#Customization of paths, file names, and directory will be needed

import numpy as np
import xarray as xr
import re
import os
import glob
import time
import traceback

# USER PATHS (EDIT THESE)
SST_DIR = r"data/ERA5 post-processed daily SST statistics on single levels from 1940 to present (1-Hourly)"
SST_YEAR_GLOB = os.path.join(SST_DIR, "*{year}*.nc")

OLR_DIR = r"data/OLR"
OLR_YEAR_GLOB = os.path.join(OLR_DIR, "*{year}*.nc")

MLS_STEP3_GLOB = r"data/Aura MLS H2O Data/QC Processed L2 H2O Data/*.nc"
OUT_DIR = r"data/Step 4 Collocated Data"

W_DAILY_DIR_2004_2009 = r"data/(Hourly) ERA5 Vertical Velocity hourly data on pressure levels from 1940 to present/w500_2004_2009"
W_DAILY_DIR_2010_2015 = r"data/(Hourly) ERA5 Vertical Velocity hourly data on pressure levels from 1940 to present/w500_2010_2015"
W_DAILY_DIR_2016_2021 = r"data/(Hourly) ERA5 Vertical Velocity hourly data on pressure levels from 1940 to present/w500_2016_2021"
W_DAILY_DIR_2022_2023 = r"data/(Hourly) ERA5 Vertical Velocity hourly data on pressure levels from 1940 to present/w500_2022_2023"

# How many MLS files to process (None = all)
N_FILES = None

W_NEGATE_LAT = True
W_LAT_SHIFT_DEG = 46.0
W_DROP_INVALID_LATS = True

MAX_YEAR = 2023
MAX_TIME_UTC = np.datetime64("2023-12-31T23:59:59")

# CONSTANTS + LOGGING
MLS_EPOCH = np.datetime64("1993-01-01T00:00:00")

os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "step4_run.log")
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("")


# TIME + LON HELPERS
def mls_seconds_to_datetime64(sec_since_1993):
    sec = np.asarray(sec_since_1993)
    return MLS_EPOCH + sec.astype("timedelta64[s]").astype("timedelta64[ns]")

def lon_to_360(lon_deg):
    lon = np.asarray(lon_deg, dtype=float)
    return np.mod(lon, 360.0)

def lon_to_180(lon_deg):
    lon = np.asarray(lon_deg, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def _parse_cf_units(units: str):
    if units is None:
        raise ValueError("Time coordinate has no 'units' attribute; cannot decode numeric times.")
    m = re.match(r"^\s*(seconds|minutes|hours|days)\s+since\s+(.+?)\s*$", units, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized CF time units: {units}")
    unit = m.group(1).lower()
    origin_str = m.group(2).replace("T", " ").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", origin_str):
        origin_str += " 00:00:00"
    origin = np.datetime64(origin_str)
    return unit, origin

def _decode_time_coord_if_numeric(ds: xr.Dataset, time_name: str, verbose: bool = True) -> xr.Dataset:
    if time_name not in ds.coords and time_name not in ds.variables:
        raise KeyError(f"Dataset missing time coordinate '{time_name}'")

    t = ds[time_name]
    if np.issubdtype(t.dtype, np.datetime64):
        return ds

    units = t.attrs.get("units", None)

    unit, origin = _parse_cf_units(units)
    vals = np.asarray(t.values)
    vals_int = np.rint(vals).astype("int64")

    if unit == "seconds":
        dt = origin + vals_int.astype("timedelta64[s]")
    elif unit == "minutes":
        dt = origin + vals_int.astype("timedelta64[m]")
    elif unit == "hours":
        dt = origin + vals_int.astype("timedelta64[h]")
    elif unit == "days":
        dt = origin + vals_int.astype("timedelta64[D]")
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    ds2 = ds.assign_coords({time_name: (t.dims, dt.astype("datetime64[ns]"))})
    return ds2

def _mask_fill_values(ds: xr.Dataset, fill_threshold: float = 1e20, verbose: bool = True) -> xr.Dataset:
    ds = ds.copy()
    for v in ds.data_vars:
        if v.endswith("_bounds"):
            continue
        da = ds[v]
        if np.issubdtype(da.dtype, np.floating):
            ds[v] = da.where(np.abs(da) < fill_threshold)
    return ds

def _prep_for_interp(ds: xr.Dataset, lat_name: str, lon_name: str, verbose: bool = True) -> xr.Dataset:
    ds = ds.copy()
    if lat_name in ds.coords:
        ds = ds.sortby(lat_name)
        if verbose:
            latv = ds[lat_name].values
    if lon_name in ds.coords:
        ds = ds.sortby(lon_name)
        if verbose:
            lonv = ds[lon_name].values
    return ds

def _pick_first_existing(name_candidates, ds: xr.Dataset, kind: str):
    for n in name_candidates:
        if n in ds.coords or n in ds.variables:
            return n
    raise KeyError(f"Could not find {kind} among candidates {name_candidates}. Available coords={list(ds.coords)}, vars={list(ds.variables)}")

def _detect_lon_mode(ds_lon_coord: xr.DataArray) -> str:
    lonv = np.asarray(ds_lon_coord.values, dtype=float)
    return "180" if np.nanmin(lonv) < 0 else "360"

def _convert_profile_lon_for_dataset(dataset_lon_coord: xr.DataArray, lon_profile_raw: np.ndarray) -> np.ndarray:
    mode = _detect_lon_mode(dataset_lon_coord)
    return lon_to_180(lon_profile_raw) if mode == "180" else lon_to_360(lon_profile_raw)

def _fix_w_latitude_geometry(
    ds: xr.Dataset,
    lat_name: str,
    *,
    negate_lat: bool = True,
    lat_shift_deg: float = 46.0,
    drop_invalid_lats: bool = True,
    verbose: bool = True,
) -> xr.Dataset:
    ds = ds.copy()

    lat_vals = np.asarray(ds[lat_name].values, dtype=float)

    # Fix corrupted latitude: lat_fixed = -lat_bad + 46
    if negate_lat:
        lat_fixed = -lat_vals + lat_shift_deg
    else:
        lat_fixed = lat_vals + lat_shift_deg

    if drop_invalid_lats:
        lat_dim = ds[lat_name].dims[0]
        keep = np.isfinite(lat_fixed) & (lat_fixed >= -90.0) & (lat_fixed <= 90.0)
        if not np.any(keep):
            raise ValueError("All corrected W latitudes are outside [-90, 90]. Check W latitude correction.")
        if not np.all(keep):
            ds = ds.isel({lat_dim: np.where(keep)[0]})
            lat_fixed = lat_fixed[keep]

    ds = ds.assign_coords({lat_name: (ds[lat_name].dims, lat_fixed)})


    return ds

from collections import OrderedDict

# Keep only a few open daily W datasets to avoid re-opening constantly
_W_DAY_CACHE: "OrderedDict[str, xr.Dataset]" = OrderedDict()
_W_DAY_CACHE_MAX = 3  # 1-5 is fine; 3 is safer

def w_daily_dir_for_year(year: int) -> str:
    if 2004 <= year <= 2009:
        return W_DAILY_DIR_2004_2009
    elif 2010 <= year <= 2015:
        return W_DAILY_DIR_2010_2015
    elif 2016 <= year <= 2021:
        return W_DAILY_DIR_2016_2021
    elif 2022 <= year <= 2023:
        return W_DAILY_DIR_2022_2023
    else:
        raise ValueError(f"No W daily directory configured for year {year}")

def _open_w_day_dataset(day_D: np.datetime64, *, verbose: bool = True) -> xr.Dataset:
    day_str = str(day_D)  # 'YYYY-MM-DD'
    if day_str in _W_DAY_CACHE:
        ds = _W_DAY_CACHE.pop(day_str)
        _W_DAY_CACHE[day_str] = ds
        return ds

    year = int(day_str[:4])
    wdir = w_daily_dir_for_year(year)
    fp = os.path.join(wdir, f"w500_{day_str}.nc")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing W daily file: {fp}")


    ds = xr.open_dataset(fp, decode_times=False)

    time_name = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else None)
    if time_name is None:
        raise KeyError(f"W daily file missing time coord. file={fp} coords={list(ds.coords)} dims={list(ds.dims)}")

    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"W daily file missing lat/lon. file={fp} coords={list(ds.coords)}")

    ds = _decode_time_coord_if_numeric(ds, time_name, verbose=verbose)

    ds = _fix_w_latitude_geometry(
        ds,
        lat_name,
        negate_lat=W_NEGATE_LAT,
        lat_shift_deg=W_LAT_SHIFT_DEG,
        drop_invalid_lats=W_DROP_INVALID_LATS,
        verbose=verbose,
    )

    ds = _prep_for_interp(ds, lat_name, lon_name, verbose=verbose)

    _W_DAY_CACHE[day_str] = ds
    while len(_W_DAY_CACHE) > _W_DAY_CACHE_MAX:
        old_day, old_ds = _W_DAY_CACHE.popitem(last=False)
        try:
            old_ds.close()
        except Exception:
            pass

    return ds

def collocate_step4(
    ds_clean: xr.Dataset,
    era5_sst: xr.Dataset,
    olr_ds: xr.Dataset,
    *,
    sst_var: str = "sst",
    w_var: str = "w",
    olr_var: str = "olr",
    sst_time_candidates=("valid_time", "time"),
    sst_lat_candidates=("latitude", "lat"),
    sst_lon_candidates=("longitude", "lon"),
    drop_land: bool = True,
    verbose: bool = True,
) -> xr.Dataset:

    # MLS time
    prof_time_ns = mls_seconds_to_datetime64(ds_clean["time"].values).astype("datetime64[ns]")

    # Hard cutoff: keep only profiles through end of 2023
    keep = prof_time_ns <= MAX_TIME_UTC
    if not np.any(keep):
        raise ValueError("All profiles in this MLS file are after 2023-12-31; skipping file.")

    if not np.all(keep):
        ds_clean = ds_clean.isel(profile=np.where(keep)[0])
        prof_time_ns = prof_time_ns[keep]

    prof_day_D = prof_time_ns.astype("datetime64[D]")

    # MLS lat/lon
    latp_vals = ds_clean["lat"].values.astype(float)
    lon_raw = ds_clean["lon"].values.astype(float)

    # Detect SST coord names
    sst_time = _pick_first_existing(sst_time_candidates, era5_sst, "SST time coord")
    sst_lat  = _pick_first_existing(sst_lat_candidates,  era5_sst, "SST lat coord")
    sst_lon  = _pick_first_existing(sst_lon_candidates,  era5_sst, "SST lon coord")

    if sst_var not in era5_sst:
        raise KeyError(f"SST variable '{sst_var}' not found. Available: {list(era5_sst.data_vars)}")

    # OLR coords/variable (fixed names in files)
    olr_time = "time"
    olr_lat = "lat"
    olr_lon = "lon"

    if olr_time not in olr_ds.coords and olr_time not in olr_ds.variables:
        raise KeyError(f"OLR dataset missing time coordinate '{olr_time}'. Coords: {list(olr_ds.coords)}")
    if olr_lat not in olr_ds.coords and olr_lat not in olr_ds.variables:
        raise KeyError(f"OLR dataset missing latitude coordinate '{olr_lat}'. Coords: {list(olr_ds.coords)}")
    if olr_lon not in olr_ds.coords and olr_lon not in olr_ds.variables:
        raise KeyError(f"OLR dataset missing longitude coordinate '{olr_lon}'. Coords: {list(olr_ds.coords)}")
    if olr_var not in olr_ds.data_vars:
        raise KeyError(f"OLR variable '{olr_var}' not found. Available: {list(olr_ds.data_vars)}")

    latp = xr.DataArray(latp_vals, dims=("profile",))
    lonp_sst = _convert_profile_lon_for_dataset(era5_sst[sst_lon], lon_raw)
    lonp_olr = _convert_profile_lon_for_dataset(olr_ds[olr_lon], lon_raw)

    lonp_sst_da = xr.DataArray(lonp_sst, dims=("profile",))
    lonp_olr_da = xr.DataArray(lonp_olr, dims=("profile",))

    tp_day = xr.DataArray(prof_day_D, dims=("profile",))

    # SST: strict same UTC day + bilinear interpolation

    sst_time_da = era5_sst[sst_time]
    if len(sst_time_da.dims) != 1:
        raise ValueError(f"SST time coord '{sst_time}' is not 1D. dims={sst_time_da.dims}")

    # Use actual dim name of the SST variable (usually same as sst_time)
    sst_dim = era5_sst[sst_var].dims[0]
    if sst_dim != sst_time_da.dims[0]:
        raise ValueError(f"SST var time dim '{sst_dim}' != SST time coord dim '{sst_time_da.dims[0]}'")

    sst_days = sst_time_da.values.astype("datetime64[ns]").astype("datetime64[D]")

    # Collapse multiple times per day to one daily field if needed
    sst_var_da = era5_sst[sst_var].assign_coords({sst_time: (sst_dim, sst_days)})
    sst_var_da = sst_var_da.groupby(sst_time).mean()

    # Memory-safe: 1 day at a time
    sst_out = np.full((latp.sizes["profile"],), np.nan, dtype=np.float32)
    tp_day_vals = tp_day.values.astype("datetime64[D]")
    uniq_days, invd = np.unique(tp_day_vals, return_inverse=True)

    sst_day_vals = sst_var_da[sst_time].values.astype("datetime64[D]")

    for k, d in enumerate(uniq_days):
        idx = np.where(invd == k)[0]
        if idx.size == 0:
            continue

        day_mask = (sst_var_da[sst_time].values.astype("datetime64[D]") == np.datetime64(d, "D"))
        if not np.any(day_mask):
            raise KeyError(f"SST day {d} not found after daily grouping.")

        sst2d = sst_var_da.isel({sst_time: np.where(day_mask)[0][0]})

        sst_interp = sst2d.interp(
            {sst_lat: latp.isel(profile=idx), sst_lon: lonp_sst_da.isel(profile=idx)},
            method="linear",
        ).values

        sst_out[idx] = sst_interp.astype(np.float32)

    sst_p = xr.DataArray(sst_out, dims=("profile",))

    # OLR: strict same UTC day + bilinear interpolation

    olr_time_da = olr_ds[olr_time]
    if len(olr_time_da.dims) != 1:
        raise ValueError(f"OLR time coord '{olr_time}' is not 1D. dims={olr_time_da.dims}")

    olr_dim = olr_ds[olr_var].dims[0]
    if olr_dim != olr_time_da.dims[0]:
        raise ValueError(f"OLR var time dim '{olr_dim}' != OLR time coord dim '{olr_time_da.dims[0]}'")

    olr_out = np.full((latp.sizes["profile"],), np.nan, dtype=np.float32)

    for k, d in enumerate(uniq_days):
        idx = np.where(invd == k)[0]
        if idx.size == 0:
            continue

        olr2d = olr_ds[olr_var].sel(
            {olr_time: np.datetime64(d, "ns")},
            method="nearest",
            tolerance=np.timedelta64(36, "h"),
        )

        olr_interp = olr2d.interp(
            {olr_lat: latp.isel(profile=idx), olr_lon: lonp_olr_da.isel(profile=idx)},
            method="linear",
        ).values

        olr_out[idx] = olr_interp.astype(np.float32)

    olr_p = xr.DataArray(olr_out, dims=("profile",))

    # W: nearest hour + bilinear interpolation (day by day)

    w500_out = np.full((latp.sizes["profile"],), np.nan, dtype=np.float32)

    # Round MLS times to nearest hour (half-up)
    prof_time_hr = (prof_time_ns.astype("datetime64[m]") + np.timedelta64(30, "m"))
    prof_time_hr = prof_time_hr.astype("datetime64[h]").astype("datetime64[ns]")

    uniq_times, inv = np.unique(prof_time_hr, return_inverse=True)

    for k, t in enumerate(uniq_times):
        idx = np.where(inv == k)[0]
        if idx.size == 0:
            continue

        day = t.astype("datetime64[D]")
        ds_day = _open_w_day_dataset(day, verbose=False)

        w_time_name = "time" if "time" in ds_day.coords else ("valid_time" if "valid_time" in ds_day.coords else None)
        if w_time_name is None:
            raise KeyError(f"W day dataset missing time coord. day={day} coords={list(ds_day.coords)}")

        w_lat_name = "latitude" if "latitude" in ds_day.coords else ("lat" if "lat" in ds_day.coords else None)
        w_lon_name = "longitude" if "longitude" in ds_day.coords else ("lon" if "lon" in ds_day.coords else None)
        if w_lat_name is None or w_lon_name is None:
            raise KeyError(f"W day dataset missing lat/lon. day={day} coords={list(ds_day.coords)}")

        if w_var not in ds_day.data_vars:
            raise KeyError(f"W variable '{w_var}' not found in daily W file for {day}. vars={list(ds_day.data_vars)}")

        lonp_w_day = _convert_profile_lon_for_dataset(ds_day[w_lon_name], lon_raw)
        lonp_w_day_da = xr.DataArray(lonp_w_day, dims=("profile",))

        # Select nearest hour inside single-day file
        w2d = ds_day[w_var].sel(
            {w_time_name: np.datetime64(t)},
            method="nearest",
            tolerance=np.timedelta64(1, "h"),
        )

        w_interp = w2d.interp(
            {w_lat_name: latp.isel(profile=idx), w_lon_name: lonp_w_day_da.isel(profile=idx)},
            method="linear",
        ).values

        w500_out[idx] = w_interp.astype(np.float32)

    w_p = xr.DataArray(w500_out, dims=("profile",))

    # Attach
    ds_tagged = ds_clean.copy()
    ds_tagged["time_utc"] = (("profile",), prof_time_ns)
    ds_tagged["day_utc"] = (("profile",), prof_day_D.astype("datetime64[ns]"))  # store day stamp at midnight ns
    ds_tagged["sst"] = (("profile",), np.asarray(sst_p.values, dtype=float))
    ds_tagged["olr"] = (("profile",), np.asarray(olr_p.values, dtype=float))
    ds_tagged["w500"] = (("profile",), np.asarray(w_p.values, dtype=float))

    if verbose:
        n_nan_sst = int(np.sum(~np.isfinite(ds_tagged["sst"].values)))
        n_nan_olr = int(np.sum(~np.isfinite(ds_tagged["olr"].values)))
        n_nan_w = int(np.sum(~np.isfinite(ds_tagged["w500"].values)))

    # Drop land points by SST NaN
    if drop_land:
        ocean_mask = np.isfinite(ds_tagged["sst"].values)
        before = ds_tagged.sizes["profile"]
        ds_tagged = ds_tagged.isel(profile=np.where(ocean_mask)[0])
        after = ds_tagged.sizes["profile"]


    return ds_tagged

def open_w_for_year(year: int) -> xr.Dataset:
    # pick the correct interval folder (customize if different file sizes/configurations)
    if 2004 <= year <= 2009:
        wdir = W_DAILY_DIR_2004_2009
    elif 2010 <= year <= 2015:
        wdir = W_DAILY_DIR_2010_2015
    elif 2016 <= year <= 2021:
        wdir = W_DAILY_DIR_2016_2021
    elif 2022 <= year <= 2023:
        wdir = W_DAILY_DIR_2022_2023
    else:
        raise ValueError(f"No W daily directory configured for year {year}")

    pattern = os.path.join(wdir, f"w500_{year}-*.nc")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No W daily NetCDF files found for year {year}. Looked for: {pattern}")


    dsets = []

    for fp in files:
        # filename: w500_YYYY-MM-DD.nc
        m = re.search(r"w500_(\d{4}-\d{2}-\d{2})\.nc$", os.path.basename(fp))
        if not m:
            raise ValueError(f"Unexpected W filename format: {fp}")
        day = np.datetime64(m.group(1))  # YYYY-MM-DD (UTC day)

        ds = xr.open_dataset(fp, decode_times=False)

        # find time coordinate/dimension name
        time_name = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else None)
        if time_name is None:
            raise KeyError(f"W daily file missing time coord. File={fp} coords={list(ds.coords)} dims={list(ds.dims)}")

        # many converters store time as 0...23, 0...82800, etc. We only need hour-of-day offsets.
        tvals = np.asarray(ds[time_name].values)

        # If numeric, interpret as hours from an origin, keeping only hour offsets within the file.
        if np.issubdtype(tvals.dtype, np.number):
            # Most common case: 0..23, but works as long as it's hourly monotonic. Rebase to start-of-file.
            offsets_h = (tvals - tvals[0]).astype("int64")
        else:
            # If already datetime-like compute offsets from first timestamp
            tns = tvals.astype("datetime64[ns]")
            offsets_h = ((tns - tns[0]) / np.timedelta64(1, "h")).astype("int64")

        # Build absolute timestamps for this day
        new_time = day.astype("datetime64[ns]") + offsets_h.astype("timedelta64[h]")

        ds = ds.assign_coords({time_name: (ds[time_name].dims, new_time)})

        dsets.append(ds)

    # concatenate along time
    ds_year = xr.concat(dsets, dim="time", coords="minimal", compat="override")
    ds_year = ds_year.sortby("time")

    # safety: drop duplicate timestamps if any exist
    _, unique_idx = np.unique(ds_year["time"].values, return_index=True)
    ds_year = ds_year.isel(time=np.sort(unique_idx))

    # close the per-file handles
    for ds in dsets:
        try:
            ds.close()
        except Exception:
            pass

    return ds_year


# DATASET CACHES
sst_cache = {}
olr_cache = {}
w_cache = {}

def open_sst_for_year(year: int) -> xr.Dataset:
    """Open SST for a specific year (cache). Requires SST filenames contain the year."""
    if year in sst_cache:
        return sst_cache[year]

    pattern = SST_YEAR_GLOB.format(year=year)
    candidates = sorted(glob.glob(pattern))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No SST NetCDF found for year {year}. Looked for: {pattern}")

    ds = xr.open_mfdataset(candidates, combine="by_coords", decode_times=False)

    ds = _mask_fill_values(ds, verbose=True)

    sst_time = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)
    if sst_time is None:
        raise KeyError(f"SST dataset missing time coordinate. Coords: {list(ds.coords)}")

    ds = _decode_time_coord_if_numeric(ds, sst_time, verbose=True)

    sst_lat = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    sst_lon = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if sst_lat is None or sst_lon is None:
        raise KeyError(f"SST dataset missing lat/lon coords. Coords: {list(ds.coords)}")

    ds = _prep_for_interp(ds, sst_lat, sst_lon, verbose=True)

    # Ensure SST time is sorted and unique
    ds = ds.sortby(sst_time)
    _, unique_idx = np.unique(ds[sst_time].values, return_index=True)
    ds = ds.isel({sst_time: np.sort(unique_idx)})

    # Slice to this year
    start = np.datetime64(f"{year}-01-01")
    end = np.datetime64(f"{year+1}-01-01")
    ds = ds.sel({sst_time: slice(start, end - np.timedelta64(1, "ns"))})

    if ds.sizes.get(sst_time, 0) == 0:
        raise ValueError(f"SST has no times for year {year} after slicing. Check time decode.")


    sst_cache[year] = ds
    return ds

def open_olr_for_year(year: int) -> xr.Dataset:
    """Open daily OLR for a specific year (cache). Requires filenames contain the year."""
    if year in olr_cache:
        return olr_cache[year]

    pattern = OLR_YEAR_GLOB.format(year=year)
    candidates = sorted(glob.glob(pattern))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No OLR NetCDF found for year {year}. Looked for: {pattern}")

    ds = xr.open_mfdataset(candidates, combine="by_coords", decode_times=False)

    ds = _mask_fill_values(ds, verbose=True)

    # OLR files use: olr, time, lat, lon
    olr_time = "time"
    olr_lat = "lat"
    olr_lon = "lon"

    if olr_time not in ds.coords and olr_time not in ds.variables:
        raise KeyError(f"OLR dataset missing time coordinate '{olr_time}'. Coords: {list(ds.coords)}")
    if olr_lat not in ds.coords and olr_lat not in ds.variables:
        raise KeyError(f"OLR dataset missing latitude coordinate '{olr_lat}'. Coords: {list(ds.coords)}")
    if olr_lon not in ds.coords and olr_lon not in ds.variables:
        raise KeyError(f"OLR dataset missing longitude coordinate '{olr_lon}'. Coords: {list(ds.coords)}")
    if "olr" not in ds.data_vars:
        raise KeyError(f"OLR variable 'olr' not found. Available: {list(ds.data_vars)}")

    ds = _decode_time_coord_if_numeric(ds, olr_time, verbose=True)
    ds = _prep_for_interp(ds, olr_lat, olr_lon, verbose=True)

    # Ensure OLR time is sorted and unique
    ds = ds.sortby(olr_time)
    _, unique_idx = np.unique(ds[olr_time].values, return_index=True)
    ds = ds.isel({olr_time: np.sort(unique_idx)})

    # Slice to this year
    start = np.datetime64(f"{year}-01-01")
    end = np.datetime64(f"{year+1}-01-01")
    ds = ds.sel({olr_time: slice(start, end - np.timedelta64(1, "ns"))})

    if ds.sizes.get(olr_time, 0) == 0:
        raise ValueError(f"OLR has no times for year {year} after slicing. Check time decode.")


    olr_cache[year] = ds
    return ds

def open_w_for_year_cached(year: int):
    """Open W for a specific year (cache)"""
    if year in w_cache:
        return w_cache[year]

    ds_w = open_w_for_year(year)

    w_time_name = "time" if "time" in ds_w.coords else ("valid_time" if "valid_time" in ds_w.coords else None)
    if w_time_name is None:
        raise KeyError(f"W dataset missing time coord. Coords: {list(ds_w.coords)}")

    w_lat = "latitude" if "latitude" in ds_w.coords else ("lat" if "lat" in ds_w.coords else None)
    w_lon = "longitude" if "longitude" in ds_w.coords else ("lon" if "lon" in ds_w.coords else None)
    if w_lat is None or w_lon is None:
        raise KeyError(f"W dataset missing lat/lon coords. Coords: {list(ds_w.coords)}")

    ds_w = _decode_time_coord_if_numeric(ds_w, w_time_name, verbose=True)

    ds_w = _fix_w_latitude_geometry(
        ds_w,
        w_lat,
        negate_lat=W_NEGATE_LAT,
        lat_shift_deg=W_LAT_SHIFT_DEG,
        drop_invalid_lats=W_DROP_INVALID_LATS,
        verbose=True,
    )

    ds_w = _prep_for_interp(ds_w, w_lat, w_lon, verbose=True)

    w_cache[year] = (ds_w, w_time_name)
    return w_cache[year]

# MAIN
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mls_nc_files = sorted(glob.glob(MLS_STEP3_GLOB))
    if len(mls_nc_files) == 0:
        raise RuntimeError("No MLS files found. Check MLS_STEP3_GLOB.")

    files_to_run = mls_nc_files if N_FILES is None else mls_nc_files[:N_FILES]

    for i, fp in enumerate(files_to_run, start=1):
        t_file = time.time()

        ds_clean = None
        ds_tagged = None

        try:
            ds_clean = xr.open_dataset(fp)

            t0 = mls_seconds_to_datetime64(ds_clean["time"].values[0]).astype("datetime64[D]")
            year = int(str(t0)[:4])

            if year > MAX_YEAR:
                break

            era5_sst = open_sst_for_year(year)
            olr_ds = open_olr_for_year(year)

            ds_tagged = collocate_step4(
                ds_clean,
                era5_sst,
                olr_ds,
                sst_var="sst",
                w_var="w",
                olr_var="olr",
                drop_land=True,
                verbose=True
            )

            out_name = os.path.basename(fp).replace(".nc", "_tagged.nc")
            out_path = os.path.join(OUT_DIR, out_name)

            ds_tagged.to_netcdf(out_path)


        except Exception as e:
            pass

        finally:
            try:
                if ds_tagged is not None:
                    ds_tagged.close()
            except Exception:
                pass
            try:
                if ds_clean is not None:
                    ds_clean.close()
            except Exception:
                pass
    for ds in _W_DAY_CACHE.values():
        try:
            ds.close()
        except Exception:
            pass
    _W_DAY_CACHE.clear()

    for ds in sst_cache.values():
        try:
            ds.close()
        except Exception:
            pass
    sst_cache.clear()

    for ds in olr_cache.values():
        try:
            ds.close()
        except Exception:
            pass
    olr_cache.clear()


if __name__ == "__main__":
    main()
