from cython_vinth2p import vinth2p_ecmwf_fast
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar

# do some atmosphere eddy and mean meridional ciculations.
def vinth2p_ecmwf_wrap(
    var3d, hbcoefa, hbcoefb, p0, varps, plevo, phis, tbot, varint, intyp, extrapp, spval
):
    """
    This is a static type wrapper for Deepak's cythonized version of the vinth2p_ecmwf.
    Some numpy operations on the model variables will change the static type.
    """
    var3d = np.array(var3d).astype(np.float32)
    hbcoefa = np.array(hbcoefa).astype(np.float64)
    hbcoefb = np.array(hbcoefb).astype(np.float64)
    p0 = float(p0)
    varps = np.array(varps).astype(np.float32)
    plevo = np.array(plevo).astype(np.float64)
    phis = np.array(phis).astype(np.float32)
    tbot = np.array(tbot).astype(np.float32)
    varint = int(varint)
    intyp = int(intyp)
    extrapp = int(extrapp)
    spval = float(spval)

    var_p = vinth2p_ecmwf_fast(
        var3d,
        hbcoefa,
        hbcoefb,
        p0,
        varps,
        plevo,
        phis,
        tbot,
        varint,
        intyp,
        extrapp,
        spval,
    )
    return var_p


def h2pvar(cesmxr, var, plevs, intyp="lin"):
    """
    Convert hybrid coordinate level CESM variable to pressure levels using
    vinth2p_ecmwf_fast converted from ncl and cythonized by Deepak Chandan.
    This function expects the cesm xarray of monthly data or a minimum
    number of a set of cesm variables in an xarray. This function can
    loop over the time variable.

    Args:
        cesmxr : input data in a xarray format
        var : the name of the 3-D variable which you want to trasform
        plevs : output pressure levels (units: hPa)
        intyp : specify the interpolation type. Either of: "lin" for linear, "log"
                for log or "loglog" for log-log interpolation (default = "lin")

    Returns:
        3D array of shape (nplevo, nlat, nlon) or 4D array of shape (time nplevo, nlat, nlon)
        where nplevo is the length of the input pressure levels variable plevo

    Default Settings:
        phis : surface geopotential height (units: m2/s2)
        tbot : temperature at level closest to ground (units: K). Only used
               if interpolating geopotential height
        varint : one of: "T"=1, "Z"=2 or None=3 to indicate whether interpolating temperature,
              geopotential height, or some other variable
        intyp : specify the interpolation type. Either of: "lin"=1 for linear, "log"=2
                for log or "loglog"=3 for log-log interpolation
        extrapp 0 = no extrapolation when the pressure level is outside of the range of psfc.


    """
    try:
        hbcoefa = cesmxr["hyam"].data
    except:
        raise Exception("No hyam varibale in xarray")
    try:
        hbcoefb = cesmxr["hybm"].data
    except:
        raise Exception("No hybm varibale in xarray")

    p0 = 1000.0  # hPa
    plevo = plevs
    spval = 9.96921e36
    extrapp = 1

    ndim = cesmxr[var].ndim
    if (ndim < 3) and (ndim > 4):
        raise ValueError(
            "Error: Working with CESM variable array that is not"
            "of dimension (time,ilev,lat,lon)"
            "or of dimension (ilev,lat,lon)"
        )
    try:
        ntime = cesmxr["time"].data.shape[0]
    except:
        ntime = None
    try:
        varint = ["T", "Z3"].index(var) + 1
    except:
        varint = 3

    if ntime is None:
        # Working with no time dimension
        try:
            varps = cesmxr["PS"].data / 100.0  # hPa
        except:
            raise Exception("No PS varibale in xarray")
        try:
            phis = cesmxr["PHIS"].data
        except:
            raise Exception("No PHIS varibale in xarray")
        try:
            tbot = cesmxr["T"].data[25, :, :]
        except:
            raise Exception("No T varibale in xarray")
        try:
            var3d = np.ma.masked_invalid(cesmxr[var].data)
        except:
            raise Exception("No {} varibale in xarray".format(var))
        try:
            intyp = ["lin", "log"].index(var) + 1
        except:
            intyp = 3
        pvar = vinth2p_ecmwf_wrap(
            var3d,
            hbcoefa,
            hbcoefb,
            p0,
            varps,
            plevo,
            phis,
            tbot,
            varint,
            intyp,
            extrapp,
            spval,
        )
        xrda = xr.DataArray(
            pvar,
            coords=[
                ("plev", plevs),
                ("lat", cesmxr.coords["lat"].data),
                ("lon", cesmxr.coords["lon"].data),
            ],
            dims=["plev", "lat", "lon"],
            attrs=cesmxr[var].attrs,
            name=var,
        )
        timeattrs = None
    elif ntime == 1:
        # cesm file will have dims(time=1,height,lat,lon)
        try:
            varps = cesmxr["PS"].data.squeeze() / 100.0  # hPa
        except:
            raise Exception("No PS varibale in xarray")
        try:
            phis = cesmxr["PHIS"].data.squeeze()
        except:
            raise Exception("No PHIS varibale in xarray")
        try:
            tbot = cesmxr["T"].data[0, 25, :, :]
        except:
            raise Exception("No T varibale in xarray")
        try:
            var3d = np.ma.masked_invalid(cesmxr[var].data.squeeze())
        except:
            raise Exception("No {} varibale in xarray".format(var))
        try:
            intyp = ["lin", "log"].index(var) + 1
        except:
            intyp = 3
        pvar = vinth2p_ecmwf_wrap(
            var3d,
            hbcoefa,
            hbcoefb,
            p0,
            varps,
            plevo,
            phis,
            tbot,
            varint,
            intyp,
            extrapp,
            spval,
        )
        # pvar will have dims(pressure,lat,lon): return xarray.dataset
        # reconstruct a xarray Dataset form DataArray
        xrda = xr.DataArray(
            pvar,
            coords=[
                ("plev", plevs),
                ("lat", cesmxr.coords["lat"].data),
                ("lon", cesmxr.coords["lon"].data),
            ],
            dims=["plev", "lat", "lon"],
            attrs=cesmxr[var].attrs,
            name=var,
        )

        timeattrs = None
    else:
        try:
            varps = cesmxr["PS"].data / 100.0  # hPa
        except:
            raise Exception("No PS varibale in xarray")
        try:
            phis = cesmxr["PHIS"].data
        except:
            raise Exception("No PHIS varibale in xarray")
        try:
            tbot = cesmxr["T"].data[:, 25, :, :]
        except:
            raise Exception("No T varibale in xarray")
        try:
            var4d = np.ma.masked_invalid(cesmxr[var].data)
        except:
            raise Exception("No {} varibale in xarray".format(var))
        try:
            intyp = ["lin", "log"].index(var) + 1
        except:
            intyp = 3

        pvar = np.array(
            [
                vinth2p_ecmwf_wrap(
                    var4d[t],
                    hbcoefa,
                    hbcoefb,
                    p0,
                    varps[t],
                    plevo,
                    phis[t],
                    tbot[t],
                    varint,
                    intyp,
                    extrapp,
                    spval,
                )
                for t in tqdm(range(0, ntime), "converting...")
            ]
        )
        xrda = xr.DataArray(
            pvar,
            coords=[
                ("time", cesmxr["time"].data),
                ("plev", plevs),
                ("lat", cesmxr.coords["lat"].data),
                ("lon", cesmxr.coords["lon"].data),
            ],
            dims=["time", "plev", "lat", "lon"],
            attrs=cesmxr[var].attrs,
            name=var,
        )
        timeattrs = cesmxr.time.attrs

    # convert DataArray to DataSet before return
    xrds = xrda.to_dataset(name=xrda.name)
    lattrs = {
        "long_name": "pressure",
        "units": "hPa",
        "positive": "down",
        "standard_name": "atmosphere__pressure_coordinate",
    }
    xrds.attrs = cesmxr[var].attrs
    if timeattrs is not None:
        xrds.time.attrs = timeattrs
    xrds.plev.attrs = lattrs
    xrds.lat.attrs = cesmxr[var].lat.attrs
    xrds.lon.attrs = cesmxr[var].lon.attrs
    return xrds


if __name__ == "__main__":

    # use this to test out the functions
    in_file = "cesmpirblall01.cam2.h0.Z3_T_PS_PHIS_hyb.nc"
    out_file = "cesmpirblall01.cam2.h0.Z3_p.nc"
    z3_on_hyb = xr.open_dataset(in_file, engine="netcdf4")
    # print(z3_on_hyb)
    plvl = np.array(
        [
            30.0,
            50.0,
            70.0,
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            775.0,
            850.0,
            925.0,
            1000.0,
        ]
    )
    z3_on_p = h2pvar(z3_on_hyb, "Z3", plvl)
    # there is a problem with time being written as int64
    try:
        z3_on_p.time.encoding["dtype"] = "float64"
    except:
        pass
    write_job = z3_on_p.to_netcdf(
        out_file, unlimited_dims="time", engine="netcdf4", compute=False
    )
    with ProgressBar():
        print(f"Writing to {out_file}")
        write_job.compute()
