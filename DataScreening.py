import numpy as np
import xarray as xr
import h5py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

P_UT_MIN, P_UT_MAX = 146.7799225, 316.2277527
P_UUT_MIN, P_UUT_MAX = 146.7799225, 215.4434662
P_LUT_MIN, P_LUT_MAX = 215.4434662, 316.2277527

#3.1 Inspect an MLS L2 file to confirm variable names (one-time)
        #USAGE e.g.: inspect_h5("MLS-Aura_L2GP-H2O_v05-01_YYYYdDDD.he5")
def inspect_h5(filepath: str, max_keys: int = 200) -> None:
    """Print a shallow tree of datasets/groups so you can find variable names."""
    with h5py.File(filepath, "r") as f:
        keys = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)

        f.visititems(visitor)


#3.2 Define a field mapping (edit once to match product)
@dataclass
class MLSFields:
    q: str
    precision: Optional[str]
    pressure: str
    quality: Optional[str]
    status: Optional[str]
    convergence: Optional[str]
    latitude: Optional[str]
    longitude: Optional[str]
    time: Optional[str]

    # q: "HDFEOS/SWATHS/H2O/Data_Fields/L2gpValue"                 # water vapor (e.g. "H2O" or "H2O_MixingRatio")
    # precision: Optional["HDFEOS/SWATHS/H2O/Data Fields/H2OPrecision"]  # optional
    # pressure: "HDFEOS/SWATHS/H2O/Geolocation_Fields/Pressure"          # pressure levels (hPa or Pa)
    # quality: Optional["HDFEOS/SWATHS/H2O/Data_Fields/Quality"] # profile-level quality metric
    # status: Optional["HDFEOS/SWATHS/H2O/Data_Fields/Status"]  # profile-level integer flags (or per-level)
    # convergence: Optional["HDFEOS/SWATHS/H2O/Data_Fields/Convergence"]  # convergence indicator
    # latitude: Optional["HDFEOS/SWATHS/H2O/Geolocation_Fields/Latitude"]
    # longitude: Optional["HDFEOS/SWATHS/H2O/Geolocation_Fields/Longitude"]
    # time: Optional["HDFEOS/SWATHS/H2O/Geolocation_Fields/Time"]    # time variable (optional)

FIELDS = MLSFields(
    q="HDFEOS/SWATHS/H2O/Data Fields/L2gpValue",
    precision="HDFEOS/SWATHS/H2O/Data Fields/H2OPrecision",
    pressure="HDFEOS/SWATHS/H2O/Geolocation Fields/Pressure",
    quality="HDFEOS/SWATHS/H2O/Data Fields/Quality",
    status="HDFEOS/SWATHS/H2O/Data Fields/Status",
    convergence="HDFEOS/SWATHS/H2O/Data Fields/Convergence",
    latitude="HDFEOS/SWATHS/H2O/Geolocation Fields/Latitude",
    longitude="HDFEOS/SWATHS/H2O/Geolocation Fields/Longitude",
    time="HDFEOS/SWATHS/H2O/Geolocation Fields/Time"
)

#3.3 Read the raw arrays
def read_dataset(f: h5py.File, path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path not in f:
        raise KeyError(f"Dataset not found: {path}")
    return f[path][...]

def load_mls_l2(filepath: str, fields: MLSFields) -> Dict[str, Optional[np.ndarray]]:
    with h5py.File(filepath, "r") as f:
        out = {
            "q": read_dataset(f, fields.q),
            "precision": read_dataset(f, fields.precision),
            "pressure": read_dataset(f, fields.pressure),
            "quality": read_dataset(f, fields.quality),
            "status": read_dataset(f, fields.status),
            "convergence": read_dataset(f, fields.convergence),
            "lat": read_dataset(f, fields.latitude),
            "lon": read_dataset(f, fields.longitude),
            "time": read_dataset(f, fields.time),
        }
    return out

#3.4 Profile-level “hard fail” checks

def hard_fail_mask_ml2h2o_v5(q, quality, status, convergence):
    """
    ML2H2O v005 hard-fail screening:
      - Status even
      - Quality > 0.7
      - Convergence < 2.0
      - basic sanity: at least 1 finite q >0
    Returns keep_profile (n_profile,) boolean
    """
    n_prof = q.shape[0]
    keep = np.ones(n_prof, dtype=bool)

    # must exist / finite
    keep &= np.isfinite(quality)
    keep &= np.isfinite(convergence)
    keep &= np.isfinite(status)

    # exact thresholds
    keep &= (quality > 0.7)
    keep &= (convergence < 2.0)
    keep &= ((status.astype(np.int64) % 2) == 0)


    # sanity: at least some finite retrieved values
    keep &= np.any(np.isfinite(q) & (q > 0), axis=1)

    return keep

#3.5 Level-by-level QC (positive precision + physical validity + fill handling)
def level_valid_mask_ml2h2o(q, precision):
    """
    Level QC for ML2H2O:
      - q finite
      - q > 0 (required for ln(q))
      - precision finite and > 0
    Returns valid_level (n_profile, n_level) boolean
    """
    valid = np.isfinite(q) & (q > 0)

    if precision is not None:
        valid &= np.isfinite(precision) & (precision > 0)

    return valid

#3.6 Subset vertical coordinate to UT (147–316 hPa) and keep UT masks

def subset_ut(pressure, q, precision, valid_level):
    """
    pressure can be (n_level,) or (n_profile, n_level).
    Returns p_ut, q_ut, precision_ut, valid_ut
    """
    p1 = pressure[0, :] if pressure.ndim == 2 else pressure
    ut_idx = (p1 >= P_UT_MIN) & (p1 <= P_UT_MAX)

    p_ut = p1[ut_idx]
    q_ut = q[:, ut_idx]
    prec_ut = precision[:, ut_idx] if precision is not None else None
    valid_ut = valid_level[:, ut_idx]

    return p_ut, q_ut, prec_ut, valid_ut

#3.7 Enforce UUT/LUT coverage
    #Prevents a few surviving levels from producing noisy flattening metrics
        
def apply_ut_layer_coverage(p_ut, valid_ut, min_frac=0.5):
    """
    Require at least min_frac valid levels in BOTH UUT and LUT per profile.
    Returns keep_cov (n_profile,) boolean, plus fractions for diagnostics
    """
    uut_idx = (p_ut >= P_UUT_MIN) & (p_ut <= P_UUT_MAX)
    lut_idx = (p_ut >= P_LUT_MIN) & (p_ut <= P_LUT_MAX)

    uut_total = max(np.sum(uut_idx), 1)
    lut_total = max(np.sum(lut_idx), 1)

    uut_frac = np.sum(valid_ut[:, uut_idx], axis=1) / uut_total
    lut_frac = np.sum(valid_ut[:, lut_idx], axis=1) / lut_total

    keep_cov = (uut_frac >= min_frac) & (lut_frac >= min_frac)
    return keep_cov, uut_frac, lut_frac

#3.8 Compute and store ln(q) (only where valid)
def compute_lnq(q_ut, valid_ut):
    lnq = np.full_like(q_ut, np.nan, dtype=float)
    lnq[valid_ut] = np.log(q_ut[valid_ut])
    return lnq

#3.9 Record pressure levels valid after QC (per profile)
    #Keep both: boolean mask valid_ut[profile, plev] and a compact per-profile list or packed bitmask

def valid_levels_as_pressure_lists(p_ut, valid_ut, max_profiles=10):
    """
    Returns a Python list of lists with valid pressures per profile.
    (Used only for small samples since storing this for millions of profiles is heavy)
    """
    out = []
    for i in range(min(valid_ut.shape[0], max_profiles)):
        out.append(p_ut[valid_ut[i]].tolist())
    return out

def pack_valid_mask(valid_ut):
    return np.packbits(valid_ut, axis=1)

#QC Report
def qc_report_ml2h2o(
        quality: np.ndarray,
        status: np.ndarray,
        convergence: np.ndarray,
        keep_hard: np.ndarray,
        keep_cov: np.ndarray,
        sst: np.ndarray | None = None,
        sst_bins: list[tuple[float, float]] = [(300.0, 303.0), (303.0, 305.0)],
) -> dict:
    """
    Prints and returns a QC report dictionary

    Hard-fail criteria (ML2H2O v005):
      - Quality > 0.7
      - Convergence < 2.0
      - Status even (Status % 2 == 0)

    Coverage:
      - keep_cov provided by apply_ut_layer_coverage()

    If sst is provided (per-profile), reports within SST bins
    """
    n = len(keep_hard)
    finite_q = np.isfinite(quality) & np.isfinite(status) & np.isfinite(convergence)

    fail_quality = finite_q & ~(quality > 0.7)
    fail_conv = finite_q & ~(convergence < 2.0)
    fail_status = finite_q & ~((status.astype(np.int64) % 2) == 0)
    fail_any_hard = ~keep_hard

    # coverage limiting factor among those that pass hard fail
    pass_hard = keep_hard
    fail_cov_given_hard = pass_hard & ~keep_cov

    report = {
        "N_total": int(n),
        "N_fail_quality": int(np.sum(fail_quality)),
        "N_fail_convergence": int(np.sum(fail_conv)),
        "N_fail_status_even": int(np.sum(fail_status)),
        "N_fail_any_hard": int(np.sum(fail_any_hard)),
        "N_pass_hard": int(np.sum(pass_hard)),
        "N_fail_cov_given_hard": int(np.sum(fail_cov_given_hard)),
        "frac_fail_cov_given_hard": float(np.sum(fail_cov_given_hard) / max(np.sum(pass_hard), 1)),
        "N_pass_final": int(np.sum(pass_hard & keep_cov)),
    }

    def _print_block(label: str, m: np.ndarray):
        nn = int(np.sum(m))
        if nn > 0:
            ph = pass_hard & m
            fc = fail_cov_given_hard & m

    # overall

    # by SST bins
    if sst is not None:
        sst = np.asarray(sst)
        for lo, hi in sst_bins:
            m = np.isfinite(sst) & (sst >= lo) & (sst < hi)
            _print_block(f"SST in [{lo},{hi}) K", m)

    return report

#3.10 Running the functions and finishing 3.3

def process_ml2h2o_file_step3(filepath, fields=FIELDS):
    raw = load_mls_l2(filepath, fields)

    q = raw["q"]
    precision = raw["precision"]
    pressure = raw["pressure"]

    # -3.4 hard fail
    keep_hard = hard_fail_mask_ml2h2o_v5(
        q=q,
        quality=raw["quality"],
        status=raw["status"],
        convergence=raw["convergence"],
    )

    # -3.5 level QC
    valid_level = level_valid_mask_ml2h2o(q, precision)

    # -3.6 UT subset
    p_ut, q_ut, prec_ut, valid_ut = subset_ut(pressure, q, precision, valid_level)

    # -3.7 coverage rule (per profile)
    keep_cov, uut_frac, lut_frac = apply_ut_layer_coverage(p_ut, valid_ut, min_frac=0.5)

    keep_final = keep_hard & keep_cov

    #QC Report CALL
    qc_report_ml2h2o(raw["quality"], raw["status"], raw["convergence"], keep_hard, keep_cov)

    # -3.8 ln(q)
    lnq_ut = compute_lnq(q_ut, valid_ut)

    # -3.9 summaries
    uut_idx = (p_ut >= P_UUT_MIN) & (p_ut <= P_UUT_MAX)
    lut_idx = (p_ut >= P_LUT_MIN) & (p_ut <= P_LUT_MAX)
    n_valid_ut  = np.sum(valid_ut, axis=1)
    n_valid_uut = np.sum(valid_ut[:, uut_idx], axis=1)
    n_valid_lut = np.sum(valid_ut[:, lut_idx], axis=1)

    # -3.10 dataset (store valid mask per profile)
    ds = xr.Dataset(
        data_vars={
            "q": (("profile", "plev"), q_ut),
            "lnq": (("profile", "plev"), lnq_ut),
            "precision": (("profile", "plev"), (prec_ut if prec_ut is not None else np.full_like(q_ut, np.nan))),
            "valid": (("profile", "plev"), valid_ut),
            "keep_hard": (("profile",), keep_hard),
            "keep_cov": (("profile",), keep_cov),
            "keep_final": (("profile",), keep_final),
            "uut_frac_valid": (("profile",), uut_frac),
            "lut_frac_valid": (("profile",), lut_frac),
            "n_valid_ut": (("profile",), n_valid_ut),
            "n_valid_uut": (("profile",), n_valid_uut),
            "n_valid_lut": (("profile",), n_valid_lut),
        },
        coords={
            "profile": np.arange(q_ut.shape[0]),
            "plev": p_ut,
        },
        attrs={
            "product": "ML2H2O v005",
            "hard_fail": "Quality>0.7, Convergence<2.0, Status even",
            "ut_range_hPa": f"{P_UT_MIN}-{P_UT_MAX}",
            "uut_range_hPa": f"{P_UUT_MIN}-{P_UUT_MAX}",
            "lut_range_hPa": f"{P_LUT_MIN}-{P_LUT_MAX}",
        }
    )

    # attach geolocation if present
    if raw.get("lat") is not None: ds["lat"] = (("profile",), raw["lat"])
    if raw.get("lon") is not None: ds["lon"] = (("profile",), raw["lon"])
    if raw.get("time") is not None: ds["time"] = (("profile",), raw["time"])

    ds_clean = ds.isel(profile=np.where(ds["keep_final"].values)[0])

    return ds_clean


#3.11 Process many MLS files (daily) and concatenate
from glob import glob

files = sorted(glob(r"data/Aura MLS H2O Data/ML2H2O_005-20260216_223159 (2004 Data)/*.he5"))

dsets = []
for fp in files[:10]:  # start small
    ds = process_ml2h2o_file_step3(fp)
    dsets.append(ds)

ds_all = xr.concat(dsets, dim="profile")


import os

out_dir = "data/Aura MLS H2O Data/QC Processed L2 H2O Data/"
os.makedirs(out_dir, exist_ok=True)

for fp in files:
    ds = process_ml2h2o_file_step3(fp)
    out_name = os.path.basename(fp).replace(".he5", "_UT.nc")
    ds.to_netcdf(os.path.join(out_dir, out_name))
