import os
import numpy as np
import xarray as xr

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_values,
    codes_release,
)

def _parse_valid_datetime64(h) -> np.datetime64:
    """
    Build np.datetime64 from GRIB keys.
    ERA5 typically provides:
      dataDate = YYYYMMDD
      dataTime = HHMM
      step = forecast step (for analyses usually 0)
    We'll treat valid time as analysis time + step hours.
    """
    dataDate = int(codes_get(h, "dataDate"))   # YYYYMMDD
    dataTime = int(codes_get(h, "dataTime"))   # HHMM

    yyyy = dataDate // 10000
    mm   = (dataDate // 100) % 100
    dd   = dataDate % 100

    HH = dataTime // 100
    MN = dataTime % 100

    base = np.datetime64(f"{yyyy:04d}-{mm:02d}-{dd:02d}T{HH:02d}:{MN:02d}:00")

    # step might be absent or 0 so handle safely
    step = 0
    try:
        step = int(codes_get(h, "step"))
    except Exception:
        step = 0

    # step units are usually hours in ERA5
    # If file uses different, it's still usually ok for ERA5 pressure-level analysis.
    return base + np.timedelta64(step, "h")


def _get_lat_lon_1d(h) -> tuple[np.ndarray, np.ndarray]:
    """
    Get 1D lat and lon arrays from GRIB grid definition.
    For regular lat/lon, ERA5 uses:
      Nj, Ni
      latitudeOfFirstGridPointInDegrees, latitudeOfLastGridPointInDegrees
      longitudeOfFirstGridPointInDegrees, longitudeOfLastGridPointInDegrees
      jDirectionIncrementInDegrees, iDirectionIncrementInDegrees
    """
    ni = int(codes_get(h, "Ni"))
    nj = int(codes_get(h, "Nj"))

    lat1 = float(codes_get(h, "latitudeOfFirstGridPointInDegrees"))
    lat2 = float(codes_get(h, "latitudeOfLastGridPointInDegrees"))
    lon1 = float(codes_get(h, "longitudeOfFirstGridPointInDegrees"))
    lon2 = float(codes_get(h, "longitudeOfLastGridPointInDegrees"))

    # increments
    dlat = float(codes_get(h, "jDirectionIncrementInDegrees"))
    dlon = float(codes_get(h, "iDirectionIncrementInDegrees"))

    # Build arrays using increments rather than linspace
    # Note that lat often decreases from first to last (north to south) in GRIB
    lats = lat1 + dlat * np.arange(nj, dtype=np.float64)
    lons = lon1 + dlon * np.arange(ni, dtype=np.float64)

    # If last doesn’t match, ERA5 sometimes encodes direction via sign
    # Force-lon into [-180, 180) if desired:
    # lons = ((lons + 180) % 360) - 180

    return lats, lons


def write_daily_netcdf_from_grib(
    grib_path: str,
    out_dir: str,
    *,
    short_name: str = "w",
    level_hpa: int = 500,
    type_of_level: str = "isobaricInhPa",
    varname_out: str = "w",
    compress_level: int = 4,
):
    os.makedirs(out_dir, exist_ok=True)

    # buffers for one day
    current_day = None
    times_buf: list[np.datetime64] = []
    fields_buf: list[np.ndarray] = []

    lats = None
    lons = None
    ni = nj = None

    def flush_day(day: np.datetime64):
        """Write current buffered day to NetCDF and clear buffers"""
        nonlocal times_buf, fields_buf, lats, lons

        if len(times_buf) == 0:
            return

        # sort by time
        times = np.array(times_buf, dtype="datetime64[ns]")
        data = np.stack(fields_buf, axis=0).astype(np.float32)  # (time, lat, lon)

        order = np.argsort(times)
        times = times[order]
        data = data[order, :, :]

        day_str = str(day.astype("datetime64[D]"))
        out_path = os.path.join(out_dir, f"w500_{day_str}.nc")

        ds_out = xr.Dataset(
            {varname_out: (("time", "latitude", "longitude"), data)},
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
        )
        ds_out[varname_out].attrs.update({
            "long_name": "Vertical velocity",
            "units": "Pa s**-1",
        })

        enc = {
            varname_out: {
                "zlib": True,
                "complevel": compress_level,
                "dtype": "float32",
                "chunksizes": (min(24, data.shape[0]), data.shape[1], data.shape[2]),
            }
        }
        ds_out.to_netcdf(out_path, encoding=enc)
        ds_out.close()

        # clear for next day
        times_buf = []
        fields_buf = []

    with open(grib_path, "rb") as f:
        msg_count = 0
        kept_count = 0

        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break

            try:
                msg_count += 1

                # Fast filters
                try:
                    sn = codes_get(h, "shortName")
                except Exception:
                    sn = None
                if sn != short_name:
                    continue

                try:
                    tol = codes_get(h, "typeOfLevel")
                except Exception:
                    tol = None
                if tol != type_of_level:
                    continue

                try:
                    lev = int(codes_get(h, "level"))
                except Exception:
                    lev = None
                if lev != level_hpa:
                    continue

                # valid time
                t = _parse_valid_datetime64(h)
                day = t.astype("datetime64[D]")

                # init grid once
                if lats is None:
                    # derive grid sizes and coords
                    lats, lons = _get_lat_lon_1d(h)
                    nj = lats.size
                    ni = lons.size

                # if day changes, flush previous
                if current_day is None:
                    current_day = day
                elif day != current_day:
                    flush_day(current_day)
                    current_day = day

                # read values (1 field)
                vals = codes_get_values(h)  # 1D flattened
                arr = np.asarray(vals, dtype=np.float32).reshape(nj, ni)

                times_buf.append(t.astype("datetime64[ns]"))
                fields_buf.append(arr)
                kept_count += 1


            finally:
                codes_release(h)

    # flush last day
    if current_day is not None:
        flush_day(current_day)


if __name__ == "__main__":
    write_daily_netcdf_from_grib(
        grib_path=r"data/era5_w500/w500_2022_2023.grib",
        out_dir=r"data/era5_w500/w500_2022_2023",
    )