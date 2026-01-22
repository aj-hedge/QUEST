import numpy as np
from scipy import integrate, interpolate
from astropy.io import fits
import re as regex
import photutils as photutils
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, Angle, CartesianRepresentation, ICRS, FK5
from reproject import reproject_interp
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip
import astropy.units as u
from astropy.visualization import wcsaxes as wcsaxes
from typing import Callable


def weighted_average_skycoord(coords: list[SkyCoord], weights: np.ndarray):
    """
    Computes the weighted average of a list of SkyCoord objects.

    The coordinates are converted to Cartesian, the weighted average of
    the Cartesian (x, y, z) components is computed, and then converted
    back to a SkyCoord.

    All input SkyCoord objects are transformed to the frame of the
    first coordinate in the list.

    Parameters
    ----------
    coords : list of astropy.coordinates.SkyCoord
        The list of SkyCoord objects to average.
    weights : array-like
        A list or array of weights corresponding to each coordinate.
        Must have the same length as `coords`.

    Returns
    -------
    astropy.coordinates.SkyCoord
        The weighted average coordinate.

    Raises
    ------
    ValueError
        If `coords` is empty, or `coords` and `weights` have different lengths,
        or sum of weights is zero.
    TypeError
        If `coords` is not a list of SkyCoord objects.
    """
    if not isinstance(coords, list) or not all(isinstance(c, SkyCoord) for c in coords):
        raise TypeError("Input 'coords' must be a list of SkyCoord objects.")
    if not coords:
        raise ValueError("Input coordinate list 'coords' cannot be empty.")

    weights = np.atleast_1d(np.squeeze(np.asarray(weights, dtype=float)))
    if len(coords) != len(weights):
        raise ValueError("Number of coordinates and weights must be the same.")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    sum_weights = np.sum(weights)
    if sum_weights == 0:
        raise ValueError("Sum of weights cannot be zero if there are coordinates and all weights are non-negative.")

    # Use the frame of the first coordinate as the common frame.
    common_frame = coords[0].frame

    # Convert all coordinates to this common frame and get their Cartesian representation.
    # If SkyCoords have no distance, .cartesian gives unit vectors (x,y,z on unit sphere).
    # If they have distances, .cartesian uses these distances.
    xyz_coords = []
    for c in coords:
        transformed_c = c.transform_to(common_frame)
        xyz_coords.append(transformed_c.cartesian.xyz) # This is a Quantity array [x,y,z]

    xyz_array = u.Quantity(xyz_coords)

    # Compute the weighted average of the Cartesian components.
    avg_xyz = np.average(xyz_array, axis=0, weights=weights)

    # Create a new CartesianRepresentation from the averaged components
    avg_cartesian_rep = CartesianRepresentation(avg_xyz)

    # Convert the averaged Cartesian representation back to a SkyCoord in the common frame
    avg_skycoord = SkyCoord(avg_cartesian_rep, frame=common_frame)

    return avg_skycoord

def needs_realignment(header: fits.Header, tolerance: float = 1e-6) -> list:
    """
    Programmatically checks a FITS header to see if the WCS needs realignment.

    This function checks for:
    1. Rotation of the pixel grid relative to the celestial axes.
    2. Use of an outdated coordinate reference system (e.g., FK5).

    Parameters
    ----------
    header : astropy.io.fits.Header
        The FITS header object to check.
    tolerance : float, optional
        The tolerance for checking if off-diagonal elements of the PC matrix
        are effectively zero. Default is 1e-6.

    Returns
    -------
    list
        A list of string descriptions of the reasons realignment is needed.
        Returns an empty list if the WCS is already north-aligned and in ICRS.
    """
    reasons = []
    
    try:
        wcs = WCS(header)
        if not wcs.is_celestial:
            # Not a celestial image, no need to realign in this context
            return []
    except Exception as e:
        # Header does not contain a valid WCS
        reasons.append(f"[ERROR]\tneeds_realignment: Invalid or missing WCS information: {e}")
        return reasons

    # --- 1. Check for Rotation ---
    # Astropy's WCS object standardizes rotation info (from CD or CROTA keywords)
    # into a PC matrix. A north-aligned WCS has zero off-diagonal elements.
    if 'pc' in wcs.wcs.to_header():
        pc_matrix = wcs.wcs.pc
        # Check the off-diagonal (rotation/shear) terms
        if abs(pc_matrix[0, 1]) > tolerance or abs(pc_matrix[1, 0]) > tolerance:
            reasons.append("Rotation detected in WCS (non-zero off-diagonal PC matrix elements).")

    # --- 2. Check the Coordinate Frame ---
    # The modern standard is ICRS. FK5 is common but older.
    # Anything pre-J2000 (like B1950/FK4) definitely needs transforming.
    radesys = header.get('RADESYS')
    if not radesys:
        radesys = header.get('RADECSYS')
    if not radesys:
        raise KeyError("No RA-Dec system defined in header.")
    else:
        radesys = radesys.upper()
    if radesys == 'FK5':
        reasons.append("Outdated coordinate frame 'FK5' detected (ICRS is preferred).")
    elif radesys == 'FK4':
        reasons.append("Outdated coordinate frame 'FK4' detected (ICRS is preferred).")
    
    # Also check the equinox, as this is a strong indicator of older frames
    equinox = header.get('EQUINOX', 2000.0)
    if equinox < 2000.0:
        reasons.append(f"Pre-J2000 equinox ({equinox}) detected.")
    
    return reasons

def generate_realigned_hdu(original_hdu: fits.ImageHDU, center_coord: SkyCoord) -> fits.ImageHDU:
    """
    Takes an original (potentially rotated and off-center) HDU and returns a new, 
    realigned HDU that is guaranteed to be centered on the given coordinate.

    This is the robust solution to the NoOverlapError.

    Parameters
    ----------
    original_hdu : fits.ImageHDU
        The input HDU containing the data and original WCS.
    center_coord : SkyCoord
        The celestial coordinate that should be the center of the new, realigned image.

    Returns
    -------
    fits.ImageHDU
        A new HDU with north-aligned data and a header centered on center_coord.
    """
    original_wcs = WCS(original_hdu.header)
    original_data = original_hdu.data

    # Determine the pixel scale from the original WCS. This preserves the resolution.
    try:
        pixel_scale_matrix = original_wcs.pixel_scale_matrix
        pixel_scale = np.sqrt(np.abs(np.linalg.det(pixel_scale_matrix))) * u.deg
    except (AttributeError, TypeError):
        pixel_scale = np.abs(original_wcs.wcs.cdelt[0]) * u.deg

    # --- Create the target header, centered ON THE DESIRED CUTOUT COORDINATE ---
    # This completely ignores the original CRPIX/CRVAL, which might be off-image.
    target_header = fits.Header()
    target_header['NAXIS1'] = original_data.shape[1]
    target_header['NAXIS2'] = original_data.shape[0]
    target_header['CTYPE1'] = 'RA---TAN'
    target_header['CTYPE2'] = 'DEC--TAN'
    
    # Define the reference pixel as the center of the DATA ARRAY.
    # This makes the new header simple and intuitive.
    target_header['CRPIX1'] = original_data.shape[1] / 2.0 + 0.5
    target_header['CRPIX2'] = original_data.shape[0] / 2.0 + 0.5
    
    target_header['CRVAL1'] = center_coord.icrs.ra.degree
    target_header['CRVAL2'] = center_coord.icrs.dec.degree
    
    # Set the pixel scale to be north-up (no rotation).
    # A negative CDELT1 makes RA increase to the left.
    target_header['CDELT1'] = -pixel_scale.to(u.deg).value
    target_header['CDELT2'] = pixel_scale.to(u.deg).value
    target_header['RADESYS'] = 'ICRS'
    
    target_wcs = WCS(target_header)
    
    # --- Perform the reprojection onto this new, ideal grid ---
    reprojected_data, footprint = reproject_interp(
        (original_data, original_wcs),
        target_wcs,
        shape_out=original_data.shape
    )
    
    # Create and return a new HDU object. It is now safe for Cutout2D.
    realigned_hdu = fits.ImageHDU(data=reprojected_data, header=target_header)
    return realigned_hdu

def get_mean_local_bkg(data: np.ndarray, wcs: WCS, coords: SkyCoord,
                    annu_in: u.Quantity, annu_out: u.Quantity, bkg_sigma_clip: float=3) -> np.ndarray:
    '''
    Get the mean local background value within annuli extending from `annu_in` to `annu_out` at positions `coords`.
    '''
    annuli = photutils.aperture.SkyCircularAnnulus(coords,annu_in,annu_out)
    sigma_clipper = SigmaClip(sigma=bkg_sigma_clip,maxiters=10)
    annuli_stats = photutils.aperture.ApertureStats(data,annuli,wcs=wcs,sigma_clip=sigma_clipper)
    bkg_means = annuli_stats.mean
    return np.atleast_1d(bkg_means) # ensure we can index/loop through the result even if it is for one coord

def calc_ap_corr(hdul: fits.HDUList, ap_diameter: u.Quantity=2*u.arcsec, ap_corrections: np.ndarray=None) -> float:
    '''
    See if we can get a PSF_FWHM or equivalent from the header, assume gaussian PSF profile and
    get the ratio of the total flux to the aperture integrated flux of the PSF.

    NOTE: The proper way to do this is empirically and tricky to automate. Need to find a bright,
    compact, unsaturated star and sample the "growth" curve of flux in increasingly larger apertures,
    stopping at ~95% of the curve as the "total" flux. Taking the ratio of total/aperture flux
    for this star, mapped to their respective aperture sizes, as the aperture correction to apply
    for any objects we meaure the flux of with an aperture if the aperture doesn't fully contain
    the object convolved with the PSF.
    
    If the aperture is large enough relative to the PSF and the source,
    then the aperture correction should be approximately 1, with 5-10% flux uncertainty being a
    reasonable estimate in place of an aperture correction.

    TODO: REPLACE MOST OF THIS WITH SOME LOGIC TO DETERMINE IF APERTURE CORRECTION IS REQUIRED AND IF THE IMAGE IS
        LARGE ENOUGH AND CONTAINS A BRIGHT, COMPACT, ISOLATED STAR TO DO SO!!!
        First check if a curve-of-growth already exists for the image in question (pre-computed and saved previously).
        If not and we can do the aperture correction, we use a few (5) stars and get the average curve-of-growth,
        saving it as the image's filepath + _curve_of_growth.csv, containing aperture radii and associated correction factors.
    '''
    ap_corr = 1
    if ap_diameter == None:
        return ap_corr
    if ap_corrections:
        spline = interpolate.make_interp_spline(ap_corrections[:,0],
                                                ap_corrections[:,1])
        ap_corr = spline(ap_diameter.to(u.arcsec).value/2)
        return ap_corr
    if get_hdu_with(hdul, 'PSF_FWHM'):
        fwhm = get_hdu_with(hdul, 'PSF_FWHM').header['PSF_FWHM']  # assumed in units of arcsec
        if ap_diameter.to(u.arcsec).value < 3*fwhm: # Only if aperture is relatively small to the FWHM
            print(f"[INFO]\tAperture correction is likely required.")
            # s = fwhm / np.sqrt(8*np.log(2))
            # spatial_vec = np.linspace(-50,50,int(1e6))
            # psf_gauss = 1/np.sqrt(2*np.pi*s**2) * np.exp(-(spatial_vec - 0)**2/(2*s**2))
            # ap_slice = np.argwhere(np.logical_and(spatial_vec > -ap_diameter.to(u.arcsec).value/2, spatial_vec < ap_diameter.to(u.arcsec).value/2)).T
            # ap_psf_flux = integrate.trapezoid(psf_gauss[ap_slice],spatial_vec[ap_slice])
            # ap_corr = 1/ap_psf_flux
    elif get_hdu_with(hdul, 'CDELT1'):
        pass
        # pixscale = abs(get_hdu_with(hdul,'CDELT1').header['CDELT1']) * 3600  # arcsec
        # if ap_diameter.to(u.arcsec).value < 6*pixscale:
        #     # assume FWHM is 5*pixscale
        #     s = 5*pixscale / np.sqrt(8*np.log(2))
        #     spatial_vec = np.linspace(-50,50,int(1e6))
        #     psf_gauss = 1/np.sqrt(2*np.pi*s**2) * np.exp(-(spatial_vec - 0)**2/(2*s**2))
        #     ap_slice = np.argwhere(np.logical_and(spatial_vec > -ap_diameter.to(u.arcsec).value/2, spatial_vec < ap_diameter.to(u.arcsec).value/2)).T
        #     ap_psf_flux = integrate.trapezoid(psf_gauss[ap_slice],spatial_vec[ap_slice])
        #     ap_corr = 1/ap_psf_flux
    
    if get_hdu_with(hdul, 'ORIGIN').header.get('ORIGIN') in ['Spitzer', 'SEIP']:
        # Interpolate from warm corrections with small annulus when possible using:
        # https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/17/
        ap_radii = np.array([2,3,4,5,6,8,10]) * 1.2
        match get_hdu_with(hdul, 'BAND').header['BAND']:
            case 'IRAC1':
                ap_corrs = np.array([1.2132, 1.1233, 1.0736, 1.0588, 1.0315, 1.0112, 1.0])
            case 'IRAC2':
                ap_corrs = np.array([1.2322, 1.1336, 1.0809, 1.0625, 1.0349, 1.0125, 1.0])
            case 'IRAC3':
                ap_corrs = np.array([1.366, 1.135, 1.076, 1.063, 1.030, 1.011, 1.0])
            case 'IRAC4':
                ap_corrs = np.array([1.568, 1.221, 1.087, 1.084, 1.051, 1.017, 1.0])
            case 'MIPS24':
                ap_radii = np.array([3.5, 7.0, 13.0, 20.0, 35.0]) * 2.5
                ap_corrs = np.array([2.78, 2.05, 1.17, 1.13, 1.06]) # diagonal of Table 4.13
            case _:
                return ap_corr
        spline = interpolate.make_interp_spline(ap_radii,ap_corrs)
        ap_corr = spline(ap_diameter.to(u.arcsec).value/2)

    return ap_corr

def flux2mag(flux: u.Quantity) -> float:
    '''
    Convert a flux density (`Quantity`) to AB magnitude.
    '''
    return -2.5*np.log10(flux.to(u.Jy).value) + 8.90


def calculate_rms(hdul: fits.HDUList) -> float:
    """
    Calculates the sigma-clipped RMS of the background in an image HDU,
    after converting the data to Jy for consistent comparison.
    """
    im_hdu = get_image_hdu(hdul)
    if im_hdu is None or im_hdu.data is None:
        return None

    # 1. Convert the entire image array to Jy
    band = get_band(hdul)
    data_jy = convert_flux(im_hdu.data, get_astronomy_method(hdul), hdul, ap_diameter=None, band=band)
    if data_jy is None:
        return None

    # 2. Perform sigma clipping on the Jy-scaled data
    #    Use the data part of the masked array, ignoring the mask itself
    #    Clip to 5-sigma to include more non-linear noise effects in RMS estimate (more realistic)
    clipped_data = sigma_clip(data_jy.value, sigma=5.0, maxiters=10, masked=False)

    # 3. Calculate RMS using the root of the mean of the squares
    #    Use nanmean to be robust against any remaining NaNs
    rms_jy = np.sqrt(np.nanmean(clipped_data**2))
    
    return rms_jy if np.isfinite(rms_jy) else None


def get_pix_res(hdul: fits.HDUList) -> u.Quantity:
    '''
    Returns the average pixel resolution in degrees.
    '''
    wcs = WCS(get_image_hdu(hdul).header, naxis=2)
    return (np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * wcs.wcs.cunit[0]).to(u.deg)


def get_image_hdu(hdul: fits.HDUList, return_idx: bool=False) -> fits.ImageHDU:
    '''
    Iterate through a HDUList and return the first HDU that appears to contain image/cube data.
    TODO: Find a better check, maybe first checking for IMAGE extension namecard.
    '''
    im_hdu = None
    for i, hdu in enumerate(hdul):
        if isinstance(hdu, fits.ImageHDU) or hdu.header.get('XTENSION','') in ['IMAGE', 'SCI'] \
        or hdu.header.get('EXTNAME','') in ['IMAGE', 'SCI'] or hdu.header.get('EXTTYPE','') in ['IMAGE', 'SCI'] \
        or hdu.header.get('IMAGETYP','') in ['IMAGE', 'SCI'] or hdu.header.get('TYPE','') in ['IMAGE', 'SCI'] \
        or hdu.header.get('PRODCATG','') in ['SCIENCE.IMAGE'] or \
        (hdu.header.get('NAXIS',0) >=2 and hdu.header.get('NAXIS1',0) > 10 and hdu.header.get('NAXIS2',0) > 10):
            im_hdu = hdu
            break
        # if isinstance(np.asarray(hdu.data), np.ndarray):  # check array of data
        #     if len(np.asarray(hdu.data).shape) > 1: # check at least 2D
        #         if max(np.asarray(hdu.data).shape) > 30: # check reasonable size
        #             im_hdu = hdu
        #             break
    if return_idx:
        return i
    return im_hdu


def get_zp(hdul: fits.HDUList, tmass_corr=None, band='') -> float:
    '''
    Get the photometric zero-point (AB magnitude).
    When the PHOTSYS is VEGA and there is a conversion available for the given BAND, apply it.
    (NOTE: not accounting for 2MASS correction)
    '''
    if get_hdu_with(hdul,'BUNIT'):
        # If units are already in some form of Jy, skip (8.9 will make ZP correction in Jy just in case)
        if 'JY' in get_hdu_with(hdul,'BUNIT').header['BUNIT'].strip(' ').upper(): zp = 8.9
    if get_hdu_with(hdul,'PHOTZP'): zp = get_hdu_with(hdul,'PHOTZP').header['PHOTZP']
    elif get_hdu_with(hdul,'MAGZERO'): zp = get_hdu_with(hdul,'MAGZERO').header['MAGZERO']
    elif get_hdu_with(hdul,'MAGZPT'): zp = get_hdu_with(hdul,'MAGZPT').header['MAGZPT']
    elif get_hdu_with(hdul,'MAGZP'): zp = get_hdu_with(hdul,'MAGZP').header['MAGZP']
    elif get_hdu_with(hdul,'FLUXMAG0'): zp = 2.5*np.log10(get_hdu_with(hdul,'FLUXMAG0').header['FLUXMAG0'])
    # HSC above BUNIT that makes this equation hold for stars: (AB magnitude) = -2.5 log(flux / FLUXMAG0)
    elif get_hdu_with(hdul,'ZPAB'): zp = np.log10(get_hdu_with(hdul,'ZPAB').header['ZPAB'])
    else: raise KeyError("FITS header did not contain a photometric zero-point.")
    if get_hdu_with(hdul,'PHOTSYS') and get_hdu_with(hdul,'PHOTSYS').header['PHOTSYS'].strip(' ').upper() == 'VEGA':
        if band == 'Ks' or band == '2.15um':
            zp += 1.83  # VEGA CONVERSION CURRENTLY ONLY FOR HAWK-I K band
    # zp = zp + tmass_corr
    return zp


def validate_image(im_hdu: fits.ImageHDU, min_fill_factor: float=0.4) -> bool:
    '''
    From the image HDU, get the data, check for a fill-value (using the mode of the data distribution), else use NaNs, then
    ensure the fill factor of valid data is at least min_fill_factor.
    '''
    data = im_hdu.data
    mode = np.unique(data, return_counts=True)[0]
    if len(mode) > 1: mode = mode[0]
    data = np.where(data == mode, np.nan, data)
    if np.sum(~np.isnan(data)) / len(data.ravel()) < min_fill_factor:
        return False
    else:
        return True


def skycoord_in_image(fits_file: str, coord: SkyCoord) -> bool:
    '''
    Checks if a coord is present in a FITS image (using the header).
    TODO: DataTool should ingest local data, validate and store set of DataEntry instances for
    later lookup/loops (such as utilising this function).
    '''
    # Open the FITS file and extract the WCS information
    with fits.open(fits_file) as hdul:
        header = get_image_hdu(hdul).header
        wcs = WCS(header, naxis=2)

        # Get the dimensions of the image
        n_x = header['NAXIS1']
        n_y = header['NAXIS2']

    # Get the pixel coordinates of the corners of the image
    corners_pix = np.array([[0, 0], [n_x - 1, 0], [0, n_y - 1], [n_x - 1, n_y - 1]])

    # Convert the pixel coordinates of the corners to world coordinates
    corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])

    # Convert to SkyCoord for easier comparisons
    # corners_skycoord = SkyCoord(corners_world, unit='deg')
    corners_skycoord = corners_world
    
    # Create a bounding box of RA and Dec
    min_ra, max_ra = corners_skycoord.ra.deg.min(), corners_skycoord.ra.deg.max()
    min_dec, max_dec = corners_skycoord.dec.deg.min(), corners_skycoord.dec.deg.max()

    # Check if the SkyCoord is within the bounding box
    is_within_ra = min_ra <= coord.ra.deg <= max_ra
    is_within_dec = min_dec <= coord.dec.deg <= max_dec

    return is_within_ra and is_within_dec


def get_astronomy_method(hdul: fits.HDUList) -> str:
    '''
    Identify the type of astronomy method the data was observed with (using the units keyword in header).
    Assumes interferometric for Jy / beam, radio single-dish for Jy and photometric for ADU or e-/s.
    '''
    brightness_unit = None
    hdu = get_hdu_with(hdul,'BUNIT')
    if hdu: brightness_unit = hdu.header['BUNIT'].strip(' ').upper()
    
    match brightness_unit:
        case 'JY/BEAM':
            method = 'interferometric'
        case 'JY':
            method = 'single-dish'  # could also be photometric...
        case 'COUNTS/S':
            method = 'photometric'
        case 'ADU':
            method = 'photometric'
        case 'ADU/S':
            method = 'photometric'
        case '10*NJY':
            # Why on Earth is the JADES output FITS this way???!
            method = 'photometric'
        case 'DN':
            # This is found in WISE image headers "Data Numbers"
            method = 'photometric'
        case 'MJY/SR':
            # This is found in Spitzer image headers
            method = 'photometric'
        case None:
            # OK, so the header did not have enough information? Assume photometric then...
            method = 'photometric'
        case _:
            raise ValueError(f"BUNIT={brightness_unit} does not have a matched astronomy method.")

    return method


def get_hdu_with(hdul: fits.HDUList, keyword: str) -> fits.FitsHDU:
    '''
    Iterate through a HDUList and return the first HDU that contains the required CARD via keyword.
    '''
    hdu_out = None
    
    for hdu in hdul:
        if keyword in hdu.header:
            hdu_out = hdu
            break
    return hdu_out


def get_instrument(hdul: fits.HDUList) -> str:
    '''
    Attempt to resolve the instrument the data came from using the 'INSTRUME', 'TELESCOP' or 'TILENAME' keywords.
    '''
    instr = ''
    if get_hdu_with(hdul,'INSTRUME'):
        instr = get_hdu_with(hdul,'INSTRUME').header['INSTRUME']
    elif get_hdu_with(hdul,'TELESCOP'):
        instr = get_hdu_with(hdul,'TELESCOP').header['TELESCOP']
    elif get_hdu_with(hdul,'TILENAME'):
        instr = regex.sub('[^a-zA-Z]','',get_hdu_with(hdul,'TILENAME').header['TILENAME'])
    elif get_hdu_with(hdul,'SURVEY'):
        instr = get_hdu_with(hdul,'SURVEY').header['SURVEY']
    elif get_hdu_with(hdul,'ORIGIN'):
        instr = get_hdu_with(hdul,'ORIGIN').header['ORIGIN']
    return instr


def get_wavelength(hdul: fits.HDUList) -> u.Quantity:
    '''
    Identify the wavelength of observation, matching with common filter name lambda_c lookups
    if needed (only approximating to parent wavelength).
    TODO: Refactor to use get_hdu_with() function
    '''
    lambda_c = 1 * u.um
    for hdu in hdul:
        if 'WAVELEN' in hdu.header:
            lambda_c = hdu.header['WAVELEN'] * u.um
            break
        if 'FILTER' in hdu.header or 'BAND' in hdu.header or 'BANDS' in hdu.header or 'HIERARCH ESO INS FILT1 NAME' in hdu.header:
            if 'BAND' in hdu.header: filter_name = str(hdu.header['BAND']).strip(' ')
            elif 'FILTER' in hdu.header: filter_name = str(hdu.header['FILTER']).strip(' ')
            elif 'HIERARCH ESO INS FILT1 NAME' in hdu.header: filter_name = str(hdu.header['HIERARCH ESO INS FILT1 NAME']).strip(' ')
            else: filter_name = str(hdu.header['BANDS']).strip(' ')
            # filter_name = filter_name.split()
            match filter_name:
                case 'U' | 'u' | "u'" | 'u*' | 'u_SDSS':
                    lambda_c = 0.365
                case 'B' | 'b':
                    lambda_c = 0.445
                case 'G' | "g'" | 'g2' | 'G2' | 'g_SDSS':
                    lambda_c = 0.464
                case 'g' | 'DECAM_g':   # DES / DECam g
                    lambda_c = 0.472
                    if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'HSC': lambda_c = 0.464
                case 'V' | 'v' | "v'":
                    lambda_c = 0.551
                case 'R' | "r'" | "R'" | 'Rc' | 'Re' | 'Rj' | 'r2' | 'R2' | 'r_SDSS':
                    lambda_c = 0.658
                case 'r' | 'DECAM_r':   # DES / DECam r
                    lambda_c = 0.6415
                    if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'HSC': lambda_c = 0.658
                case 'VIS':
                    lambda_c = 0.725
                case 'I' | "i'" | 'Ic' | 'Ie' | 'Ij' | 'i2' | 'I2' | 'i_SDSS':
                    lambda_c = 0.806
                case 'i' | 'DECAM_i':   # DES / DECam i
                    lambda_c = 0.7835
                    if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'HSC': lambda_c = 0.806
                case 'Z' | "z'" | 'z2' | 'Z2' | 'z_SDSS':
                    lambda_c = 0.900
                case 'z' | 'DECAM_z':   # DES / DECam z
                    lambda_c = 0.9260
                    if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'HSC': lambda_c = 0.900
                case 'Y' | 'Y2' | 'y2':
                    lambda_c = 1.020
                case 'y' | 'DECAM_y':   # DES / DECam y
                    lambda_c = 1.0095
                    if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'HSC': lambda_c = 1.020
                case 'NIR_Y':   # assumed Euclid
                    lambda_c = 1.033
                case 'J' | "J'" | 'Js' | 'j':
                    lambda_c = 1.220
                case 'NIR_J':   # assumed Euclid
                    lambda_c = 1.259
                case 'H' | 'h':
                    lambda_c = 1.630
                case 'NIR_H':   # assumed Euclid
                    lambda_c = 1.686
                case 'K' | "K'" | 'Klong' | 'K8' | 'nbK':
                    lambda_c = 2.190
                case 'Ks' | 'k':  # HAWK-I/VIRCAM Ks
                    lambda_c = 2.146
                case 'L' | "L'" | "nbL'":
                    lambda_c = 3.450
                # case '1':
                #     if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'WISE': lambda_c = 3.4 # approx
                # case '2':
                #     if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'WISE': lambda_c = 4.6 # approx
                # case '3':
                #     if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'WISE': lambda_c = 12.0 # approx
                # case '4':
                #     if 'TELESCOP' in hdu.header and hdu.header['TELESCOP'].strip(' ') == 'WISE': lambda_c = 22.0 # approx
            lambda_c = lambda_c * u.um
            break
        elif 'CTYPE3' in hdu.header and (hdu.header['CTYPE3'].strip(' ') in ['FREQ', 'FREQ-LSR']):    # radio image format
            if 'CUNIT3' not in hdu.header:
                freq_unit = u.Hz
            else:
                freq_unit = u(hdu.header['CUNIT3'].strip(' '))
            freq_c = hdu.header['CRVAL3'] * freq_unit
            lambda_c = freq_c.to(u.um)
            break
        elif 'RESTFRQ' in hdu.header:
            if 'CUNIT3' not in hdu.header:
                freq_unit = u.Hz
            else:
                freq_unit = u(hdu.header['CUNIT3'].strip(' '))
            freq_c = hdu.header['RESTFRQ'] * freq_unit
            lambda_c = freq_c.to(u.um)
            break

    return lambda_c


def get_band(hdul: fits.HDUList) -> str:
    '''
    See if any of the HDU have the BAND recorded in their headers.
    '''
    band_str = ''
    if get_hdu_with(hdul, 'BAND'):
        band_str = get_hdu_with(hdul, 'BAND').header['BAND']
    # Band string coercing
    match band_str.lower():
        case 'u':
            band_str = 'u'
        case 'g':
            band_str = 'g'
        case 'r' | 'r2':
            band_str = 'r'
        case 'i' | 'i2':
            band_str = 'i'
        case 'z' | 'z2':
            band_str = 'z'
        case 'y':
            band_str = 'Y'
        case 'j':
            band_str = 'J'
        case 'h':
            band_str = 'H'
        case 'ks':
            band_str = 'Ks'
        case 'k':
            band_str = 'K'
        case _:
            pass

    return band_str


def convert_flux(flux_sum: np.ndarray, astronomy_method: str, hdul: fits.HDUList,
                 ap_diameter: u.Quantity=2*u.arcsec, band='',
                 known_origins: list[str]=None,
                 origin_conversions: dict[str, Callable]=None,
                 ap_corrections: np.ndarray=None) -> u.Quantity:
    '''
    Convert flux from Jy / beam, ADU or COUNTS/S to Jy (assuming background subtraction already applied).
    First must identify the type and any additional factors such as zero-point.
    Require certain header keyword(s) that unamiguously define the conversion to use (so no mis-matches),
    per-instrument or per-survey, i.e. ORIGIN (to organise methods of converting flux), USEABZP (where
    the zero-point has been altered to account for corrections and is in AB magnitudes), BAND (wavelength
    band), WAVELEN (effective/central wavelength of the band), etc.
    '''
    if not known_origins: known_origins = list(default_known_origins)
    if not origin_conversions: origin_conversions = default_conversions
    match astronomy_method:
        case 'interferometric':
            hdu = get_image_hdu(hdul)
            bmaj = hdu.header['BMAJ']
            bmin = hdu.header['BMIN']
            pixres_square = np.sqrt(abs(hdu.header['CDELT1']*hdu.header['CDELT2']))
            pix_per_beam = 1.133 * hdu.header['BMAJ'] * hdu.header['BMIN'] / (hdu.header['CDELT1'])**2
            flux_out = flux_sum / pix_per_beam * u(hdu.header['BUNIT'].strip(' ')) * u.beam
        case 'single-dish':
            hdu = get_image_hdu(hdul)
            flux_out = flux_sum * u(hdu.header['BUNIT'].strip(' '))
        case 'photometric':
            ap_corr = calc_ap_corr(hdul,ap_diameter,ap_corrections)
            im_hdu = get_image_hdu(hdul)
            pixscale = im_hdu.header.get('PIXSCALE')
            if pixscale:
                pixscale = pixscale** 2 * u.arcsec**2
            elif im_hdu.header.get('CD1_1'):
                pixscale = abs(im_hdu.header['CD1_1'] * abs(im_hdu.header['CD2_2'])) * u.deg**2
            else:
                pixscale = get_pix_res(hdul)
            if get_hdu_with(hdul,'ORIGIN'):
                origin = get_hdu_with(hdul,'ORIGIN').header.get('ORIGIN')
            else:
                origin = ''
            match origin:
                # NOTE/TODO: Requires verifying: 2MASS, unWISE, AllWISE, VIDEO, SHARKS, UHS, VHS, LAS and new additions
                case o if o in known_origins:
                    if o == 'SHARKS':
                        # Double check if this SHARKS data has been manually calibrated in lieu of fix_SHARKS
                        if get_hdu_with(hdul,'USEABZP'):
                            pass
                        else:
                            raise NotImplementedError("SHARKS data requires manual calibration; no automatic conversion available.")
                    flux_out = origin_conversions[o](flux_sum=flux_sum, ap_corr=ap_corr, im_hdu=im_hdu,
                                                        band=band, pixscale=pixscale)
                case 'VIDEO' | 'UltraVISTA' | 'DXS' | 'UDS':   # ************************TODO************************
                    raise NotImplementedError(f"Photometric flux conversion for ORIGIN={origin} not yet implemented.")
                case _:
                    print(f'[INFO]\tNo preset conversion for ORIGIN={origin}, attempting auto conversion...')
                    # Check BUNIT to try and distinguish unrecognised ORIGIN
                    bunit = get_hdu_with(hdul,'BUNIT').header.get('BUNIT').strip(' ').upper()
                    match bunit:
                        case 'COUNTS/S':
                            flux_out = ap_corr * flux_sum*10**(3.56 - 0.4*get_zp(hdul, band=band)) * u.Jy
                        case 'ADU' | 'ADU/s':
                            # # convert ADU to e- to e-/second [NOTE: HAWK-I data suggests this is not needed?? Just use ADU??]
                            # flux_sum *= hdul[0].header['GAIN']
                            # if 'EFF_EXPT' in hdul[0].header:
                            #     flux_sum /= hdul[0].header['EFF_EXPT']
                            # else:
                            #     flux_sum /= hdul[0].header['EXPTIME']
                            flux_out = ap_corr * flux_sum*10**(3.56 - 0.4*get_zp(hdul, band=band)) * u.Jy
                        case '10*NJY':
                            flux_out = flux_sum * u.Jy
                        case 'DN': # WISE "Data Number" units apply same as ADU
                            flux_out = ap_corr * flux_sum*10**(3.56 - 0.4*get_zp(hdul, band=band)) * u.Jy
                        case 'MJY/SR':  # convert to Jy / sr and muliply by sr / pix
                            pixscale = im_hdu.header.get('PIXSCALE')
                            if pixscale:
                                pixscale = pixscale** 2 * u.arcsec**2
                            else:
                                pixscale = abs(im_hdu.header['CD1_1'] * abs(im_hdu.header['CD2_2'])) * u.deg**2
                            flux_out = ap_corr * flux_sum*10**(3.56 - 0.4*get_zp(hdul, band=band)) * 1e6 \
                                * pixscale.to(u.steradian).value * u.Jy
                        case _:
                            print("[WARNING]\tHeader did not contain a brightness unit? Assuming Jy and checking for a zero-point...")
                            flux_out = ap_corr * flux_sum*10**(3.56 - 0.4*get_zp(hdul, band=band)) * u.Jy
    return flux_out

# Survey configs when retrieving cutouts, saving, applying known fixes.
# Since this config is accessible via the QueryTool class, it can be updated with custom survey definitions as needed, simply
# linking to custom get_<survey> and fix_<survey> functions which can be imported from a library or defined in a script/notebook.
default_known_origins = ['2MASS', 'unWISE', 'AllWISE', 'EuclidMER', 'HSC', 'DECaLS', 'DES', 'HAWK-I', \
                        'UHS', 'VHS', 'LAS', 'VIKING', 'SEIP', 'Spitzer', 'SHARKS']

# TEMPLATE CONVERSION FUNCTION
# def _convert_<usecase>(flux_sum=flux_sum, ap_corr=ap_corr, im_hdu=im_hdu,
#                           band=band, pixscale=pixscale):
#    return converted_expression

def _convert_useabzp(flux_sum: np.ndarray[float], ap_corr: float, im_hdu: fits.ImageHDU, **kwargs):
    return ap_corr * flux_sum * 3631 * 10 ** (-im_hdu.header['USEABZP'] / 2.5) * u.Jy

def _convert_MJy_per_steradian(flux_sum: np.ndarray[float], ap_corr: float, pixscale: u.Quantity, **kwargs):
    return ap_corr * flux_sum * 1e6 * pixscale.to(u.steradian).value * u.Jy   # MJy/sr to Jy/pix correction

# TODO: finish populating this
default_conversions = {
    '2MASS': _convert_useabzp,
    'unWISE': _convert_useabzp,
    'AllWISE': _convert_useabzp,
    'Euclid': _convert_useabzp,
    'HSC': _convert_useabzp,
    'DECaLS': _convert_useabzp,
    'DES': _convert_useabzp,
    'DECam': _convert_useabzp,
    'HAWK-I': _convert_useabzp,
    'UHS': _convert_useabzp,
    'VHS': _convert_useabzp,
    'LAS': _convert_useabzp,
    'VIKING': _convert_useabzp,
    'SEIP': _convert_MJy_per_steradian,
    'Spitzer': _convert_MJy_per_steradian,
    'SHARKS': _convert_useabzp
}