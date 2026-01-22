import numpy as np
import requests
from astropy.io import fits
import photutils as photutils
from astropy.wcs import WCS, FITSFixedWarning
from astropy.utils.data import download_file
from astropy.coordinates import SkyCoord, Angle, CartesianRepresentation, ICRS, FK5
from astropy.nddata import utils as nddata_utils
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip
import astropy.units as u
from astropy.table import Table
from astropy.visualization import wcsaxes as wcsaxes
import glob, os, io, shutil, gzip
import pyvo as vo
from astroquery import hips2fits
from astroquery.casda import Casda      # CSIRO ASKAP Science Data Archive (e.g. RACS, EMU)
from astroquery.eso import Eso          # European Southern Observatory (e.g. VLT survey products)
from astroquery.cadc import Cadc        # Canadian Astronomy Data Centre (Large variety including radio)
from astroquery.skyview import SkyView
# from astroquery.alma import Alma        # Atacama Large Millimetre/sub-millimetre Array
from astroquery.vsa import Vsa          # VISTA Science Archive (e.g. VHS, VIDEO, VIKING)
from astroquery.ukidss import Ukidss    # UKIRT Infrared Deep Sky Surveys (not including UHS???)
from astroquery.sdss import SDSS        # SLOAN Digital Sky Survey
# from astroquery.mast import Mast        # For STSci mission products (e.g. from JWST, HST, TESS, ...)
from astroquery.ipac.irsa import Irsa   # IRSA contains a variety (e.g. Euclid, 2MASS, Spitzer, with submodules for irsa_dust and ibe)
from astroquery.ipac.irsa.irsa_dust import IrsaDust
# from astroquery.ipac.irsa.ibe import Ibe
# from astroquery.ipac.ned import Ned     # NED (only really useful for looking up well-known object image/spectra or object aliases)
# from astroquery.esa.euclid import Euclid# Euclid via ESA
# from astroquery.esa.hsa import HSA      # Herschel Science Archive
# from astroquery.esa.hubble import ESAHubble # Hubble via ESA
# from astroquery.esa.integral import Integral# INTernational Gamma-Ray Astrophysics Laboratory
# from astroquery.esa.iso import ISO      # Infrared Space Observatory
# from astroquery.esa.jwst import Jwst    # JWST via ESA
# from astroquery.esa.xmm_newton import XMMNewton # XMM-Newton via ESA
# from astroquery.esasky import ESASky    # Broad access but not consistent with the rest of ESA
# from astroquery.svo_fps import SvoFps   # Access to the Spanish Virtual Observatory Filter Profile Service (get filter curves for various instruments)
from .lib_util import get_hdu_with, get_image_hdu, get_pix_res, skycoord_in_image

# TODO: Could try downloading all matching images and letting PhotometryTool pick the deepest one,
#       with future possibility of the tool coadding them.
# TODO: See if we can replace uses of astropy.Table with directly interfacing with SIA2Result instead.
# NOTE: List of SIA2Service accessible data collections via IRSA here: https://irsa.ipac.caltech.edu/ibe/sia.html
# NOTE: List of SIAService accessible data collections via NoirLAB Science Archive (NSA):
#       https://datalab.noirlab.edu/docs/manual/UsingAstroDataLab/DataAccessInterfaces/SimpleImageAccessSIA/SimpleImageAccessSIA.html


def update_header(file: str, keyword: str, value: str, mode: str, clobber: bool=False):
    '''
    Update keyword-value card in FITS header of `file`, either in the 'image' HDU or 'primary' HDU based on `mode`.
    '''
    with fits.open(file, mode='update') as hdul:
        if mode == 'image':
            # Find the HDU with image data
            for hdu in hdul:
                if isinstance(hdu, fits.ImageHDU) or len(hdul) == 1:
                    header = hdu.header
                    if keyword not in header or clobber:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        header[keyword] = value
                        print(f"Added {keyword}={value} to image HDU in {file}")
                    break
        elif mode == 'primary':
            # Update the primary HDU header
            primary_header = hdul[0].header
            if keyword not in primary_header or clobber:
                try:
                    value = float(value)
                except ValueError:
                    pass
                primary_header[keyword] = value
                print(f"Added {keyword}={value} to primary HDU in {file}")

def save_image_from_SIARecord(deepest_image: vo.dal.SIARecord, outname: str, coord: SkyCoord, cutout_size: u.Quantity, keep_full_image: bool,
            **kwargs):
    '''
    Helper to save images from SIARecord URLs.
    '''
    debug = kwargs.get('debug', False)
    with fits.open(deepest_image.getdataurl(), use_fsspec=False) as hdul:
        # Apply Cutout2D to reduce size of saved file
        if debug: print(f"[DEBUG]\tOpened URL: {deepest_image.getdataurl()}")
        im_hdu_idx = get_image_hdu(hdul, return_idx=True)
        wcs = WCS(hdul[im_hdu_idx].header)
        if debug: print(f"[DEBUG]\tPreparing cutout at {coord.to_string(style='hmsdms', precision=2)} of size {cutout_size}.")
        cutout = nddata_utils.Cutout2D(hdul[im_hdu_idx].section, position=coord, size=cutout_size, wcs=wcs)
        if debug: print(f"[DEBUG]\tCopying HDUList and updating relevant image data and header.")
        hdul_out = hdul.copy()
        hdul_out[im_hdu_idx].data = cutout.data
        hdul_out[im_hdu_idx].header.update(cutout.wcs.to_header())
    
        if debug: print(f"[DEBUG]\tSaving cutout to {outname}")
        hdul_out.writeto(outname, overwrite=True)
        hdul_out.close()

    if keep_full_image:
        fname = download_file(deepest_image.getdataurl())
        if kwargs.get('compression', None):
            match kwargs['compression']:
                case 'gzip':
                    with gzip.open(fname, 'rb') as archive:
                        with open(outname, 'wb') as save_file:
                            save_file.write(archive.read())
                    os.remove(fname)
                case _:
                    print(f"[WARNING]\tCompression type {kwargs['compression']} not recognized, saving raw file.")
                    shutil.move(fname, outname)
        else:
            shutil.move(fname, outname)

def get_image_centre_coord(filepath: str) -> SkyCoord:
    '''
    In case a coord is not on-hand, can quickly get one from a given FITS image filepath.
    '''
    hdul = fits.open(filepath)
    hdu = get_image_hdu(hdul)
    wcs = WCS(hdu.header, naxis=2)
    coord = SkyCoord.from_pixel(hdu.header['NAXIS1']/2, hdu.header['NAXIS2']/2, wcs)
    return coord

def get_extinction_table(coord: SkyCoord, **kwargs) -> Table:
    '''
    Retrieve extinction table for given `coord` from IRSA Dust Extinction Service.
    Also accepts `radius` in angular u.Quantity in the range [2 deg, 37 deg].
    From experience, the service appears to only return the following filter bandpasses:

        CTIO U, CTIO B, CTIO V, CTIO R, CTIO I
        DSS-II g, DSS-II r, DSS-II i
        SDSS u, SDSS g, SDSS r, SDSS i, SDSS z
        UKIRT J, UKIRT H, UKIRT K
        2MASS J, 2MASS H, 2MASS Ks
        IRAC-1, IRAC-2, IRAC-3, IRAC-4
        WISE-1, WISE-2
    '''
    radius = kwargs.get('radius', 2*u.deg)
    ext_table = IrsaDust.get_extinction_table(coord, show_progress=False, radius=radius)
    if len(ext_table) == 0:
        print(f"No extinction data found at {coord.to_string(style='hmsdms', precision=2)}.")
        return None
    return ext_table


class QueryTool:
    """
    TODO: Add QueryTool save-state to keep track of which SourceEntry source names have been (successfully)
    queried (found or none found) to which surveys/bands already, to avoid redundant queries.
    TODO: Add asynchronous querying to spread over multiple surveys/bands when querying one source.
    """
    def __init__(self, data_dir: str, online_archives_config: dict=None, **kwargs):
        """
        Initialize QueryTool with configuration for online archives.

        Parameters
        ----------
        `data_dir` : `str`
            Directory to store downloaded data.
        `online_archives_config` : `dict`, optional
            Configuration for online archives. If `None`, defaults to `default_online_archive_config`.
        """
        if online_archives_config:
            self.online_archives = online_archives_config
        else:
            self.online_archives = default_online_archive_config
        self.data_dir = data_dir
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(online_archives_config={self.online_archives})"

    def set_online_archives(self, online_archives_config: dict):
        self.online_archives = online_archives_config

    def query_online_archive(self, survey: str, coord: SkyCoord, bands: list[str]=None, query_radius: u.Quantity=10*u.arcsec,
                             cutout_size: u.Quantity=180*u.arcsec, keep_full_image: bool=False,
                             ignore_local_images: bool=False, **kwargs):
        """
        Abstracted wrapper to query online archives using astroquery/pyvo.
        
        Parameters
        ----------
        `survey` : `str`
            Name of survey to query from `self.online_archives`.
        `coord` : `SkyCoord`
            Target SkyCoord to retrieve cutouts for.
        `bands` : `list[str]`
            List of the bands to retrieve. If `None`, then all the available bands for the survey are retrieved.
        `cutout_size` : `u.Quantity`
            Angular size of the cutout (radius) that the cone search is performed with and trimmed to.
        """
        try:
            assert survey in self.online_archives.keys()
            query_config = self.online_archives[survey]

            if bands:
                bands_to_retrieve = list(set(bands) & set(query_config['available_bands']))
                if bands_to_retrieve == []:
                    raise ValueError(f"{bands} are not available from {survey}'s bands: {query_config['available_bands']}")
            else:
                bands = query_config['available_bands']
            
            for band in bands:
                # First check if the source is already contained in an equivalent survey-band combination image
                survey_band_filepaths = glob.glob(f"{self.data_dir}/*/{survey}/*_{band}_*.fits")
                if not ignore_local_images and any(skycoord_in_image(fp, coord) for fp in survey_band_filepaths):
                    print(f"Source already covered by local {survey} {band} image. Skipping download.")
                    continue
                print(f"Attempting to retrieve {survey} {band} data...")
                # TODO: Can we except a timeout error and retry? (DES appears to be a common trigger)
                query_config['download_function'](coord=coord, band=band, query_radius=query_radius,
                                                  cutout_size=cutout_size, keep_full_image=keep_full_image,
                                                  data_dir=self.data_dir, **kwargs)
        except Exception as e:
            print(f"[ERROR]\tquery_online_archive: Astroquery failed for {survey}: {e}")

def get_2MASS(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=20*u.arcsec,
            cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Automatically retrieve and sort requested 2MASS data containing `query_radius` around `coord`.
    TODO: CHANGE TO SIA2Service
    '''
    twomass_service = vo.dal.SIAService("https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?type=at&ds=asky&")
    im_table = twomass_service.search(pos=coord, size=query_radius)

    if len(im_table) == 0:
        print(f"No 2MASS {band} images found.")
    
    if not os.path.exists(f"{data_dir}/IR/2MASS"):
        os.mkdir(f"{data_dir}/IR/2MASS")

    tab = im_table.to_table()
    match_rows = np.argwhere(tab['band'] == band)
    
    if len(match_rows) > 1:
        print(f"Multiple 2MASS {band} images found, retrieving deepest image.")
        deepest_image = im_table.getrecord(np.argmax(np.asarray(tab[match_rows]['magzp'], dtype=float)))
    elif len(match_rows) == 1:
        print(f"2MASS {band} image found.")
        deepest_image = im_table.getrecord(match_rows[0])
    else:
        print(f"No 2MASS {band} images found.")
        return

    outname = f"{data_dir}/IR/2MASS/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    save_image_from_SIARecord(deepest_image, outname, coord, cutout_size, keep_full_image, compression='gzip', debug=kwargs.get('debug', False))

    # Attempt to apply any required fixes straight away
    fix_2MASS(outname, coord)

def fix_2MASS(filepath: str, coord: SkyCoord=None):
    '''
    2MASS requires adding the BUNIT keyword to the ImageHDU.
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    Ref: https://irsa.ipac.caltech.edu/data/2MASS/docs/releases/allsky/faq.html#conversion
        States that the data-number (DN) units are such that the MAGZP also corrects for the exposure time, i.e.
        mag = MAGZP - 2.5 log10 (Sum DN)
    '''
    update_header(filepath, 'BUNIT', 'DN', 'image', clobber=False)
    filt = fits.open(filepath)[0].header['FILTER']
    match filt:
        case 'j':
            wl = 1.235
            ext_filt = '2MASS J'
        case 'h':
            wl = 1.662
            ext_filt = '2MASS H'
        case 'k':
            wl = 2.159
            ext_filt = '2MASS Ks'
        case _:
            raise ValueError(f"Unrecognized 2MASS filter {filt} in {filepath}.")

    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = tab[tab['Filter_name'] == ext_filt]['A_SandF'][0]
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0
    
    update_header(filepath, 'WAVELEN', wl, 'image', clobber=False)
    update_header(filepath, 'ORIGIN', '2MASS', 'primary', clobber=True)
    update_header(filepath, 'USEABZP', fits.open(filepath)[0].header['MAGZP'] - ext_corr, 'image', clobber=True)
    update_header(filepath, 'BAND', filt.upper(), 'primary', clobber=True)

def get_SDSS():
    '''
    Automatically retrieve and sort requested SDSS data containing `query_radius` around `coord`.
    TODO ********************************************
    '''
    raise NotImplementedError()

def fix_SDSS():
    '''
    TODO ********************************************
    '''
    raise NotImplementedError()

def get_AllWISE(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=30*u.arcsec,
                cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Automatically retrieve and sort requested AllWISE data containing `query_radius` around `coord`.
    TODO: CHANGE TO SIA2Service
    '''
    allwise_service = vo.dal.SIAService("https://irsa.ipac.caltech.edu/ibe/sia/wise/allwise/p3am_cdd?")
    im_table = allwise_service.search(pos=coord, size=query_radius)

    if len(im_table) == 0:
        print(f"No AllWISE {band} images found.")

    if not os.path.exists(f"{data_dir}/IR/AllWISE"):
        os.mkdir(f"{data_dir}/IR/AllWISE")

    tab = im_table.to_table()
    match_rows = np.argwhere(tab['sia_bp_id'] == band)
    
    if len(match_rows) > 1:
        print(f"Multiple AllWISE {band} images found, retrieving deepest image.")
        deepest_image = im_table.getrecord(np.argmax(np.asarray(tab[match_rows]['magzp'], dtype=float)))
    elif len(match_rows) == 1:
        print(f"AllWISE {band} image found.")
        deepest_image = im_table.getrecord(match_rows[0][0])
    else:
        print(f"No AllWISE {band} images found.")
        return

    outname = f"{data_dir}/IR/AllWISE/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    save_image_from_SIARecord(deepest_image, outname, coord, cutout_size, keep_full_image, debug=kwargs.get('debug', False))

    # Attempt to apply any required fixes straight away
    fix_AllWISE(outname, coord)

def fix_AllWISE(filepath: str, coord: SkyCoord=None):
    '''
    AllWISE requires no known fixes at this time.
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    '''
    filt = fits.open(filepath)[0].header['BAND']
    match str(filt):
        case '1' | 'W1':
            wl = 3.368
            vega2ab = 2.699
            band = 'W1'
            ext_filt = 'WISE-1'
        case '2' | 'W2':
            wl = 4.618
            vega2ab = 3.339
            band = 'W2'
            ext_filt = 'WISE-1'
        case '3' | 'W3':
            wl = 12.082
            vega2ab = 5.174
            band = 'W3'
            ext_filt = None
        case '4' | 'W4':
            wl = 22.194
            vega2ab = 6.620
            band = 'W4'
            ext_filt = None
        case _:
            raise ValueError(f"Unrecognized AllWISE filter {filt} in {filepath}.")

    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = tab[tab['Filter_name'] == ext_filt]['A_SandF'][0]
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'WAVELEN', wl, 'image', clobber=False)
    update_header(filepath, 'ORIGIN', 'AllWISE', 'primary', clobber=True)
    update_header(filepath, 'USEABZP', 22.5 + vega2ab - ext_corr, 'image', clobber=True)
    update_header(filepath, 'PHOTSYS', 'AB', 'primary', clobber=True)
    update_header(filepath, 'BAND', band, 'primary', clobber=True)

def get_unWISE(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=30*u.arcsec,
            cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Automatically retrieve and sort requested unWISE data containing `query_radius` around `coord`.
    https://irsa.ipac.caltech.edu/data/WISE/unWISE/overview.html
    NOTE: we want the ones ending in 'img-m.fits' since these are with outliers removed and are deeper stacks.

    Extra args
    ----------
    outliers : `str`
        - Choose between 'removed' (default) or 'patched' outlier treatment in the unWISE coadds.
        'removed' gives slightly deeper images with more artifacts, while 'patched' gives
        cleaner images with less depth.
    '''
    IRSA_service = vo.dal.SIA2Service("https://irsa.ipac.caltech.edu/SIA")
    im_table = IRSA_service.search(pos=(coord, query_radius), collection='wise_unwise')

    if len(im_table) == 0:
        print(f"No unWISE {band} images found.")
        return

    if not os.path.exists(f"{data_dir}/IR/unWISE"):
        os.mkdir(f"{data_dir}/IR/unWISE")

    tab = im_table.to_table()
    science_rows = np.argwhere(tab['dataproduct_subtype'] == 'science')
    band_rows = np.argwhere(tab['energy_bandpassname'] == band)
    match_rows = np.intersect1d(science_rows, band_rows)

    outliers = kwargs.get('outliers', 'removed')
    assert outliers in ['removed', 'patched'], "Invalid option for 'outliers'. Must be 'removed' or 'patched'."

    if len(match_rows) >= 1:
        match_idx = 0
        for match_idx, row in enumerate(tab[match_rows]):
            if outliers == 'removed' and row['access_url'].endswith('img-m.fits'):
                break
            elif outliers == 'patched' and row['access_url'].endswith('img-u.fits'):
                break
        deepest_image = im_table.getrecord(match_rows[match_idx])
    else:
        print(f"No unWISE {band} images matching request found.")
        return

    print(f"unWISE {band} image found.")

    outname = f"{data_dir}/IR/unWISE/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    save_image_from_SIARecord(deepest_image, outname, coord, cutout_size, keep_full_image, debug=kwargs.get('debug', False))

    # Attempt to apply any required fixes straight away
    fix_unWISE(outname, coord)

def fix_unWISE(filepath: str, coord: SkyCoord=None):
    '''
    NOTE: unWISE IMAGES MAGZP ARE IN VEGA!! FIX MAGZP TO AB SCALE
    Also add header key for FILTER since WISE only provides BAND = 1, 2, 3, 4 and we look for FILTER = W1, W2, W3, W4.
    Also add BUNIT=DN to match AllWISE
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    '''
    update_header(filepath, 'BUNIT', 'DN', 'image', clobber=False)
    filt = fits.open(filepath)[0].header['BAND']
    match str(filt):
        case '1' | 'W1':
            wl = 3.368
            vega2ab = 2.699
            band = 'W1'
            ext_filt = 'WISE-1'
        case '2' | 'W2':
            wl = 4.618
            vega2ab = 3.339
            band = 'W2'
            ext_filt = 'WISE-2'
        case _:
            raise ValueError(f'Unmatched unWISE band for filter: {filt}')

    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = tab[tab['Filter_name'] == ext_filt]['A_SandF'][0]
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0
    
    update_header(filepath, 'WAVELEN', wl, 'image', clobber=False)
    update_header(filepath, 'ORIGIN', 'unWISE', 'primary', clobber=True)
    update_header(filepath, 'USEABZP', 22.5 + vega2ab - ext_corr, 'image', clobber=True)
    update_header(filepath, 'PHOTSYS', 'AB', 'primary', clobber=True)
    update_header(filepath, 'BAND', band, 'primary', clobber=True)

def get_SEIP(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=10*u.arcsec,
            cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Retrieve the Spitzer Enhanced Imaging Products (SEIP) from IRSA.
    https://irsa.ipac.caltech.edu/data/SPITZER/Enhanced/SEIP/overview.html
    NOTE: we want the ones ending in IRAC.{bandnum}.mosaic.fits WITHOUT "short"!
    '''
    IRSA_service = vo.dal.SIA2Service("https://irsa.ipac.caltech.edu/SIA")
    im_table = IRSA_service.search(pos=(coord, query_radius), collection='spitzer_seip')

    if len(im_table) == 0:
        print(f"No SEIP {band} images found.")
        return

    if not os.path.exists(f"{data_dir}/IR/SEIP"):
        os.mkdir(f"{data_dir}/IR/SEIP")

    tab = im_table.to_table()
    science_rows = np.argwhere(tab['dataproduct_subtype'] == 'science')
    band_rows = np.argwhere(tab['energy_bandpassname'] == band)
    match_rows = np.intersect1d(science_rows, band_rows)

    short_mask = np.array(['short' in row['access_url'] for row in tab[match_rows]])
    match_rows = match_rows[~short_mask]

    if len(match_rows) > 1:
        print(f"Multiple SEIP {band} images found, retrieving deepest image.")
        # We will get the headers of each access_url to find the one with the longest EXPTIME
        # TODO: alternate check for noise in region around target, in case the "deeper" image
        #       actually does not cover the target?
        exposure_times = []
        for _, row in enumerate(tab[match_rows]):
            with fits.open(row['access_url'], use_fsspec=True) as hdul:
                exposure_times.append(get_hdu_with(hdul,'EXPTIME').header['EXPTIME'])
        deepest_image = im_table.getrecord(match_rows[np.argmax(exposure_times)])
    elif len(match_rows) == 1:
        print(f"SEIP {band} image found.")
        deepest_image = im_table.getrecord(match_rows[0])
    else:
        print(f"No SEIP {band} images matching request found.")
        return

    outname = f"{data_dir}/IR/SEIP/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    save_image_from_SIARecord(deepest_image, outname, coord, cutout_size, keep_full_image, debug=kwargs.get('debug', False))

    # Attempt to apply any required fixes straight away
    fix_SEIP(outname, coord)

def fix_SEIP(filepath: str, coord: SkyCoord=None):
    '''
    SEIP requires no known fixes at this time.
    NOTE: could convert MJY/SR to JY and update BUNIT here instead of doing it during perform_photometry
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    '''
    wl = fits.open(filepath)[0].header['WAVELEN']
    match str(wl):
        case '3.6':
            band = 'IRAC1'
            ext_filt = 'IRAC-1'
        case '4.5':
            band = 'IRAC2'
            ext_filt = 'IRAC-2'
        case '5.8':
            band = 'IRAC3'
            ext_filt = 'IRAC-3'
        case '8.0':
            band = 'IRAC4'
            ext_filt = 'IRAC-4'
        case '23.68':
            band = 'MIPS24'
            ext_filt = None
        case _:
            raise ValueError(f'Unmatched Spitzer band for wavelength: {wl}')
    
    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = tab[tab['Filter_name'] == ext_filt]['A_SandF'][0]
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'ORIGIN', 'Spitzer', 'primary', clobber=True)
    update_header(filepath, 'USEABZP', fits.open(filepath)[0].header['ZPAB'] - ext_corr, 'image', clobber=True)
    update_header(filepath, 'BAND', band, 'primary', clobber=True)

def get_Euclid_MER(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=10*u.arcsec,
                cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Retrieve the Euclid MER image products from IRSA.
    '''
    IRSA_service = vo.dal.SIA2Service("https://irsa.ipac.caltech.edu/SIA")
    im_table = IRSA_service.search(pos=(coord, query_radius), collection='euclid_DpdMerBksMosaic')

    if len(im_table) == 0:
        print(f"No Euclid MER {band} images found.")
        return

    if not os.path.exists(f"{data_dir}/IR/EuclidMER"):
        os.mkdir(f"{data_dir}/IR/EuclidMER")
    if not os.path.exists(f"{data_dir}/Optical/EuclidMER"):
        os.mkdir(f"{data_dir}/Optical/EuclidMER")

    tab = im_table.to_table()
    science_rows = np.argwhere(tab['dataproduct_subtype'] == 'science')
    band_rows = np.argwhere(tab['energy_bandpassname'] == band)
    match_rows = np.intersect1d(science_rows, band_rows)

    if len(match_rows) > 1:
        print(f"Multiple Euclid MER {band} images found, retrieving deepest image.")
        # We will get the headers of each access_url to find the one with the longest EXPTIME
        exposure_times = []
        for _, row in enumerate(tab[match_rows]):
            with fits.open(row['access_url'], use_fsspec=True) as hdul:
                exposure_times.append(get_hdu_with(hdul,'EXPTIME').header['EXPTIME'])
        deepest_image = im_table.getrecord(match_rows[np.argmax(exposure_times)])
    elif len(match_rows) == 1:
        print(f"Euclid MER {band} image found.")
        deepest_image = im_table.getrecord(match_rows[0])
    else:
        print(f"No Euclid MER {band} images matching request found.")
        return

    if band in ['U','G','R','I','Z','VIS']:
        outname = f"{data_dir}/Optical/EuclidMER/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"
    else:
        outname = f"{data_dir}/IR/EuclidMER/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    save_image_from_SIARecord(deepest_image, outname, coord, cutout_size, keep_full_image, debug=kwargs.get('debug', False))

    # Attempt to apply any required fixes straight away
    fix_Euclid_MER(outname, coord)

def fix_Euclid_MER(filepath: str, coord: SkyCoord=None):
    '''
    Euclid MER requires:
    - addition of BUNIT keyword for ALL bands (ground-based optical and Euclid)... used ADU for now, but looks like there is missing information...
    - addition of INSTRUME keyword for ALL bands (ground-based can be DES/DECam, Pan-STARRS, HSC, CFIS/MegaCam, Euclid: NISP or VIS)
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    '''
    update_header(filepath, 'BUNIT', 'ADU', 'image', clobber=False)
    filt = get_hdu_with(fits.open(filepath),'FILTER').header['FILTER']
    # Match the instrument
    match filt:
        case 'DECAM_g' | 'DECAM_r' | 'DECAM_i' | 'DECAM_z':
            update_header(filepath, 'INSTRUME', 'DECam', 'primary', clobber=False)
        case 'PANSTARRS_i':
            update_header(filepath, 'INSTRUME', 'Pan-STARRS', 'primary', clobber=False)
        case 'HSC_g' | 'HSC_z':
            update_header(filepath, 'INSTRUME', 'HSC', 'primary', clobber=False)
        case 'MEGACAM_u' | 'MEGACAM_r':
            update_header(filepath, 'INSTRUME', 'MegaCam', 'primary', clobber=False)
        case 'VIS' | 'NIR_Y' | 'NIR_J' | 'NIR_H':
            update_header(filepath, 'INSTRUME', 'Euclid', 'primary', clobber=False)
        case _:
            print(f"[WARNING]\tCould not identify instrument for {filepath}")
    # Match the wavelength
    match filt:
        case 'MEGACAM_u':
            wl = 0.375
            ext_filt = 'SDSS u'
        case 'DECAM_g':
            wl = 0.473
            ext_filt = 'SDSS g'
        case 'HSC_g':
            wl = 0.474
            ext_filt = 'SDSS g'
        case 'MEGACAM_r':
            wl = 0.623
            ext_filt = 'SDSS r'
        case 'DECAM_r':
            wl = 0.642
            ext_filt = 'SDSS r'
        case 'VIS':
            wl = 0.725
            ext_filt = 'SDSS i'
        case 'PANSTARRS_i':
            wl = 0.7503
            ext_filt = 'SDSS i'
        case 'DECAM_i':
            wl = 0.784
            ext_filt = 'SDSS i'
        case 'HSC_z':
            wl = 0.889
            ext_filt = 'SDSS z'
        case 'DECAM_z':
            wl = 0.926
            ext_filt = 'SDSS z'
        case 'NIR_Y':
            wl = 1.033
            ext_filt = ['SDSS z', 'UKIRT J']
        case 'NIR_J':
            wl = 1.259
            ext_filt = 'UKIRT J'
        case 'NIR_H':
            wl = 1.686
            ext_filt = 'UKIRT H'
        case _:
            raise ValueError(f'Unmatched Euclid MER band for filter: {filt}')
        
    if filt == 'VIS':
        band = filt
    else:
        band = filt[-1]

    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = np.mean(tab[[filt in ext_filt for filt in tab['Filter_name']]]['A_SandF'])
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'WAVELEN', wl, 'image', clobber=False)
    update_header(filepath, 'ORIGIN', fits.open(filepath)[0].header['INSTRUME'], 'primary', clobber=True)
    update_header(filepath, 'USEABZP', fits.open(filepath)[0].header['MAGZERO'] - ext_corr, 'image', clobber=True)
    update_header(filepath, 'BAND', band, 'primary', clobber=True)

def get_PanSTARRS():
    '''
    Automatically retrieve and sort requested AllWISE data containing `query_radius` around `coord`.
    '''
    raise NotImplementedError()

def fix_PanSTARRS():
    '''
    TODO: ******************************
    '''
    raise NotImplementedError()

def get_HSC(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=10*u.arcsec,
                cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    This relies on code developed for PDR3 at:
    https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/blob/a83bcbd7993ae4936975a2b15f6c7041c9a7db60/pdr3/downloadCutout/downloadCutout.py
    We will assume the HSC data access tools are not installed and retrieve the required python script on-the-fly, saving to
    'external' and importing it from there.
    We will inform the user that they will need to provide their own login credentials, e.g. in a generated HSC.login file in the data directory.
    '''
    if not os.path.exists(f"{data_dir}/Optical/HSC"):
        os.mkdir(f"{data_dir}/Optical/HSC")

    if not os.path.exists(f"./external"):
        os.mkdir(f"./external")
    if not os.path.exists(f"./external/HSC_downloadCutout.py"):
        url = "https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-" \
            + "/raw/a83bcbd7993ae4936975a2b15f6c7041c9a7db60/pdr3/downloadCutout/downloadCutout.py?inline=False"
        print(f"Downloading HSC data access tool from {url} to ./external/HSC_downloadCutout.py")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(f"./external/HSC_downloadCutout.py", 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR]\tget_HSC: Error downloading file: {e}")
            return
    
    if not os.path.exists(f"{data_dir}/HSC.login") or os.stat(f"{data_dir}/HSC.login").st_size == 0:
        print(f"[WARNING]\tFirst time retrieving HSC-SSP data? You will need an account to access. Please enter your credentials in"
            +f" {data_dir}/HSC.login (first line: username, second line: password) so that future requests can be authenticated."
            +" See https://hsc-release.mtk.nao.ac.jp/datasearch/new_user/new to register.")
        if not os.path.exists(f"{data_dir}/HSC.login"):
            with open(f"{data_dir}/HSC.login", 'w') as f:
                f.write("")
        return
    else:
        with open(f"{data_dir}/HSC.login", 'r') as f:
            username = f.readline().strip('\n')
            password = f.readline().strip('\n')
    
    from external import HSC_downloadCutout as HSC_downloadCutout

    outname = f"{data_dir}/Optical/HSC/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    # NOTE: The cutout creation below does NOT use query_radius
    rect = HSC_downloadCutout.Rect.create(
        ra=str(coord.ra.value),
        dec=str(coord.dec.value),
        sw=f"{cutout_size.value}{cutout_size.unit}",
        sh=f"{cutout_size.value}{cutout_size.unit}",
        rerun='pdr3_wide',
        type='coadd',
        filter=f"HSC-{band.upper()}",
        tract=-1,
        image=True,
        mask=False,
        variance=False
    )

    images = HSC_downloadCutout.download(rect, user=username, password=password)

    if len(images) == 0:
        print(f"[WARNING]\tNo HSC cutouts returned for {coord.to_string(style='hmsdms', precision=2)} with size {cutout_size}.")
        return
    else:
        print(f"HSC {band} image found.")

    hdus: fits.HDUList = fits.open(io.BytesIO(images[0][1]))
    hdus.writeto(outname, overwrite=True)

    fix_HSC(outname, coord)

def fix_HSC(filepath: str, coord: SkyCoord=None):
    '''
    add BUNIT=ADU to image HDU, add INSTRUME=HSC to primary HDU
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    '''
    update_header(filepath, 'BUNIT', 'ADU', 'image')
    update_header(filepath, 'INSTRUME', 'HSC', 'primary')
    filt = fits.open(filepath)[0].header['FILTER']
    match filt:
        case 'g':
            wl = 0.474
            ext_filt = 'SDSS g'
        case 'r' | 'r2':
            wl = 0.617
            ext_filt = 'SDSS r'
        case 'i' | 'i2':
            wl = 0.765
            ext_filt = 'SDSS i'
        case 'z':
            wl = 0.889
            ext_filt = 'SDSS z'
        case 'y':
            wl = 0.976
            ext_filt = ['SDSS z', 'UKIRT J']
        case _:
            raise ValueError(f'Unmatched HSC band for filter: {filt}')

    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = np.mean(tab[[filt in ext_filt for filt in tab['Filter_name']]]['A_SandF'])
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'WAVELEN', wl, 'primary', clobber=False)
    update_header(filepath, 'ORIGIN', 'HSC', 'primary', clobber=True)
    # HSC makes this equation hold for stars: (AB magnitude) = -2.5 log(flux / FLUXMAG0), so the zero-point is
    # -2.5*log10(1/FLUXMAG0) = 2.5*log10(FLUXMAG0)
    update_header(filepath, 'USEABZP', 2.5*np.log10(fits.open(filepath)[0].header['FLUXMAG0']) - ext_corr, 'image', clobber=True)
    update_header(filepath, 'BAND', filt, 'primary', clobber=True)
    
def get_DECaLS(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=10*u.arcsec,
                cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Retrieve the DECaLS imaging products.
    NOTE: This requests pixscale=0.262 arcsec which is equivalent to the cutouts having data units of 1 nanomaggie / pixel
            (i.e. fluxes with ABZP = 22.5).
            https://www.legacysurvey.org/svtips/ : List item number 9.
    '''
    if not os.path.exists(f"{data_dir}/Optical/DECaLS"):
        os.mkdir(f"{data_dir}/Optical/DECaLS")

    outname = f"{data_dir}/Optical/DECaLS/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    size_pix = int(cutout_size.to(u.arcsec).value / 0.262)   # get approx cutout size in pixels by dividing by the URL pixscale requested

    # NOTE: We do not use query_radius since we are directly requesting a cutout at the specified size (in pixels, converted from cutout_size)
    decals_cutout_url = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={coord.ra.value}&dec={coord.dec.value}" \
                        +f"&layer=ls-dr10&pixscale=0.262&bands={band}&size={size_pix}"

    # Attempt to query the URL and make sure there is no Server Error (500) returned (indicating no cutout can be made within specified area)
    try:
        r = requests.get(decals_cutout_url, stream=True)
        r.raise_for_status()
        with open(outname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"[WARNING]\tLikely no DECaLS tiles at this position.\nError downloading file: {e}")
        pass

    fix_DECaLS(outname, coord)

def fix_DECaLS(filepath: str, coord: SkyCoord=None):
    '''
    add PHOTZP=22.5 to image HDU, add BUNIT=ADU to image HDU, add RADECSYS=ICRS
    can also overwrite SURVEY=DECaLS
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    '''
    update_header(filepath, 'PHOTZP', 22.5, 'image')
    update_header(filepath, 'BUNIT', 'ADU', 'image')
    update_header(filepath, 'RADECSYS', 'ICRS', 'image')
    update_header(filepath, 'SURVEY', 'DECaLS', 'primary', clobber=True)
    filt = fits.open(filepath)[0].header['BAND0']
    match filt:
        case 'g':
            wl = 0.473
            ext_filt = 'SDSS g'
        case 'r':
            wl = 0.642
            ext_filt = 'SDSS r'
        case 'i':
            wl = 0.784
            ext_filt = 'SDSS i'
        case 'z':
            wl = 0.926
            ext_filt = 'SDSS z'
        case 'Y':
            wl = 1.009
            ext_filt = ['SDSS z', 'UKIRT J']
        case _:
            raise ValueError(f'Unmatched DECaLS band for filter: {filt}')
    
    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = np.mean(tab[[filt in ext_filt for filt in tab['Filter_name']]]['A_SandF'])
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'WAVELEN', wl, 'image', clobber=False)
    update_header(filepath, 'ORIGIN', 'DECaLS', 'primary', clobber=True)
    # Default cutouts are 0.262 arcsec / pixel resolution, which matches the catalogues such that the data is in nanomaggies / pixel
    # 1 nanomaggy = 3631 * 10^-9 Jy = 22.5 AB mag, therefore the AB zero-point should be 22.5 to set the flux scale.
    # NOTE: From empirical testing and visual QA, it appears DECaLS over-estimates its sensitivity by about 0.1 mag.
    # NOTE: DECaLS supposedly has reprocessed DECam data including from DES, so should be similar or better sensitivity for the most part.
    update_header(filepath, 'USEABZP', 22.5 - ext_corr - 0.1, 'image', clobber=True)
    update_header(filepath, 'BAND', filt, 'primary', clobber=True)

def get_DES(data_dir: str, coord: SkyCoord, band: str, query_radius: u.Quantity=10*u.arcsec,
                cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Retrieve the DES imaging products.
    NOTE: We want to grab the nobkg images and specifically the 1st FITS extension image (science data)
        as the 2nd and 3rd contain background and variance (?) data.
    '''
    NSA_service = vo.dal.SIAService("https://datalab.noirlab.edu/sia/des_dr2")
    # NOTE: For whatever reason, the NSA SIAService treats the search size as the desired cutout size, hence DO NOT use query_radius here
    im_table = NSA_service.search(pos=coord, size=cutout_size)

    if len(im_table) == 0:
        print(f"No DES {band} images found.")
        return
    
    if not os.path.exists(f"{data_dir}/Optical/DES"):
        os.mkdir(f"{data_dir}/Optical/DES")

    tab = im_table.to_table()
    science_rows = np.argwhere((tab['proctype'] == 'Stack') & (tab['prodtype'] == 'image'))
    band_rows = np.argwhere(tab['obs_bandpass'] == band)
    match_rows = np.intersect1d(science_rows, band_rows)

    science_mask = np.logical_and(['_nobkg' in row['access_url'] for row in tab[match_rows]], ['&extn=1' in row['access_url'] for row in tab[match_rows]])
    match_rows = match_rows[science_mask]

    if len(match_rows) > 1:
        print(f"Multiple DES {band} images found, retrieving deepest image.")
        # np.argmax appears very broken on the DES tables (likely due to generic `object` dtype of column data), so force float dtype.
        deepest_image = im_table.getrecord(match_rows[np.argmax(np.asarray(tab[match_rows]['exptime'],dtype=float))])
    elif len(match_rows) == 1:
        print(f"DES {band} image found.")
        deepest_image = im_table.getrecord(match_rows[0])
    else:
        print(f"No DES {band} images found.")
        return

    outname = f"{data_dir}/Optical/DES/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    save_image_from_SIARecord(deepest_image, outname, coord, cutout_size, keep_full_image, debug=kwargs.get('debug', False))

    fix_DES(outname, coord)

def fix_DES(filepath: str, coord: SkyCoord=None):
    '''
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    '''
    filt = fits.open(filepath)[0].header['BAND']
    match filt:
        case 'g':
            wl = 0.473
            ext_filt = 'SDSS g'
        case 'r':
            wl = 0.642
            ext_filt = 'SDSS r'
        case 'i':
            wl = 0.784
            ext_filt = 'SDSS i'
        case 'z':
            wl = 0.926
            ext_filt = 'SDSS z'
        case 'Y':
            wl = 1.009
            ext_filt = ['SDSS z', 'UKIRT J']
        case _:
            raise ValueError(f'Unmatched DES band for filter: {filt}')

    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = np.mean(tab[[filt in ext_filt for filt in tab['Filter_name']]]['A_SandF'])
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'WAVELEN', wl, 'image', clobber=False)
    update_header(filepath, 'ORIGIN', 'DES', 'primary', clobber=True)
    # DES coadd tiles (from the multi-epoch pipeline) have fixed MAGZP=30 and data in units of picomaggies / pixel
    # 1 picomaggy = 3631 * 10^-12 Jy = 30 AB mag, therefore the AB zero-point should be 30 to set the flux scale to Jy.
    update_header(filepath, 'USEABZP', 30.0 - ext_corr, 'image', clobber=True)
    update_header(filepath, 'BAND', filt, 'primary', clobber=True)

def get_VISTA(data_dir: str, coord: SkyCoord, band: str, survey: str=None, query_radius: u.Quantity=10*u.arcsec,
                cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Retrieve the VISTA survey imaging products (VHS, VIKING, VIDEO, UltraVISTA).
    If `survey == None`, check the survey products, deepest first, and use the first
    `HDUList` that is returned. VIDEO and UltraVISTA, check for deep_stack first.
    '''
    allowed_surveys = ['UltraVISTA', 'VIDEO', 'VIKING', 'VHS']
    if not survey:
        survey = allowed_surveys
    else:
        survey = [str(survey)]
    
    for surv in survey:
        if surv not in allowed_surveys:
            raise ValueError(f'{surv} not in allowed surveys: {allowed_surveys}')

    # NOTE: DEFINE THE DATABASE TO SEARCH HERE (rather than wasting time searching 'all'). IMPORTANT
    #       TO UPDATE THE DATA RELEASE NUMBER WHEN A NEW ONE APPEARS!
    database = {
        'UltraVISTA': 'ULTRAVISTADR4',
        'VIDEO': 'VIDEODR6',
        'VIKING': 'VIKINGDR5',
        'VHS': 'VHSDR7'
    }

    # NOTE: We do not use radius=query_radius in the get_images() call since that would return multi-frame FITS HDULists
    for surv in survey:
        images = []
        outname = f"{data_dir}/IR/VISTA/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{surv}_{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"
        # Try deep_stack first if survey allows
        if surv in ['UltraVISTA', 'VIDEO']:
            images = Vsa.get_images(coord,waveband=band,frame_type='deep_stack',image_width=cutout_size,
                                    programme_id=surv,database=database[surv],verbose=False,show_progress=False)
        # If not deep_stack or not a 'deep' survey, try tilestack
        if len(images) == 0:
            images = Vsa.get_images(coord,waveband=band,frame_type='tilestack',image_width=cutout_size,
                                    programme_id=surv,database=database[surv],verbose=False,show_progress=False)
        # If get_images returned a list of HDULists, then save the first one (TODO: If not tiles, double check we aren't missing out on deepest?)
        if len(images) > 0:
            if not os.path.exists(outname.rsplit('/',1)[0]):
                os.mkdir(outname.rsplit('/',1)[0])
            images[0].writeto(outname, overwrite=True)
            break

    if len(images) == 0:
        print(f'No extragalactic VISTA {band} image found.')
        return

    try:
        fix_VISTA(outname, coord, surv)
    except Exception as e:
        print(f"[ERROR]\tCould not apply header fixes to {outname}: {e}")

def fix_VISTA(filepath: str, coord: SkyCoord=None, survey: str=None):
    '''
    add BUNIT=ADU to image HDU, some RADECSYS=ICRS which is NOT TRUE! Update to FK5 (should be true for all)
    Use standard header keywords for processing, overwriting with expected or found values:
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    
    NOTE: Native (co-added) data units are in counts, requires correcting by integration time per exposure to get ADU.
    NOTE: Vega to AB conversion obtained from http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/filter-set (assuming CASU v1.3)
    '''
    filt = fits.open(filepath)[1].header['HIERARCH ESO INS FILT1 NAME'].strip()
    match filt:
        case 'Z':
            wl = 0.877
            vega2ab = 0.524
            ext_filt = 'SDSS z'
        case 'Y':
            wl = 1.020
            vega2ab = 0.618
            ext_filt = ['SDSS z', '2MASS J']
        case 'J':
            wl = 1.252
            vega2ab = 0.937
            ext_filt = '2MASS J'
        case 'H':
            wl = 1.645
            vega2ab = 1.384
            ext_filt = '2MASS H'
        case 'Ks':
            wl = 2.147
            vega2ab = 1.839
            ext_filt = '2MASS Ks'
        case _:
            raise ValueError(f'Unmatched VISTA band for filter: {filt}')
    
    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = np.mean(tab[[filt in ext_filt for filt in tab['Filter_name']]]['A_SandF'])
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'WAVELEN', wl, 'image', clobber=True)
    update_header(filepath, 'BUNIT', 'ADU', 'image', clobber=True)
    update_header(filepath, 'RADECSYS', 'FK5', 'image', clobber=True)
    # update_header(filepath, 'EQUINOX', '2000', 'image', clobber=False)
    if survey: update_header(filepath, 'ORIGIN', survey, 'primary', clobber=True)   # Could instead check the headers to discern the survey?
    try:
        update_header(filepath, 'USEABZP', fits.open(filepath)[1].header['MAGZPT'] + vega2ab - ext_corr \
                                        + 2.5*np.log10(fits.open(filepath)[1].header['EXPTIME']), 'image', clobber=True)
        update_header(filepath, 'PHOTSYS', 'AB', 'image', clobber=True)
    except Exception as e:
        print(f'[ERROR]\tupdate_header: {e}')
    update_header(filepath, 'BAND', filt, 'primary', clobber=True)

def get_UKIRT(data_dir: str, coord: SkyCoord, band: str, survey: str=None, query_radius: u.Quantity=10*u.arcsec,
                cutout_size: u.Quantity=60*u.arcsec, keep_full_image: bool=False, **kwargs):
    '''
    Retrieve the UKIRT survey imaging products (UHS, LAS, DXS, UDS).
    If `survey == None`, check the survey products, deepest first, and use the first
    `HDUList` that is returned. DXS and UDS, check for deep_stack first.
    '''
    allowed_surveys = ['UDS', 'DXS', 'LAS', 'UHS']
    if not survey:
        survey = allowed_surveys
    else:
        survey = [str(survey)]
    
    for surv in survey:
        if surv not in allowed_surveys:
            raise ValueError(f'{surv} not in allowed surveys: {allowed_surveys}')

    # NOTE: DEFINE THE DATABASE TO SEARCH HERE (rather than wasting time searching 'all'). IMPORTANT
    #       TO UPDATE THE DATA RELEASE NUMBER WHEN A NEW ONE APPEARS!
    database = {
        'UDS': 'UKIDSSDR11PLUS',
        'DXS': 'UKIDSSDR11PLUS',
        'LAS': 'UKIDSSDR11PLUS',
        'UHS': 'UHSDR3'
    }

    # NOTE: We do not use radius=query_radius in the get_images() call since that would return multi-frame FITS HDULists
    for surv in survey:
        images = []
        outname = f"{data_dir}/IR/UKIRT/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{surv}_{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits"
        # Try deep_stack first if survey allows
        if surv in ['UDS', 'DXS']:
            images = Ukidss.get_images(coord,waveband=band,frame_type='deep_stack',image_width=cutout_size,
                                    programme_id=surv,database=database[surv],verbose=False,show_progress=False)
        # If not deep_stack or not a 'deep' survey, try leavstack
        if len(images) == 0:
            images = Ukidss.get_images(coord,waveband=band,frame_type='leavstack',image_width=cutout_size,
                                    programme_id=surv,database=database[surv],verbose=False,show_progress=False)
        # If get_images returned a list of HDULists, then save the first one (TODO: If not tiles, double check we aren't missing out on deepest?)
        if len(images) > 0:
            if not os.path.exists(outname.rsplit('/',1)[0]):
                os.mkdir(outname.rsplit('/',1)[0])
            images[0].writeto(outname, overwrite=True)
            break

    if len(images) == 0:
        print(f'No extragalactic UKIRT {band} image found.')
        return

    try:
        fix_UKIRT(outname, coord, surv)
    except Exception as e:
        print(f"[ERROR]\tCould not apply header fixes to {outname}: {e}")

def fix_UKIRT(filepath: str, coord: SkyCoord=None, survey: str=None):
    '''
    ****** TODO ******: Double check against UKIRT documentation of some sort (setting of metadata, default header keywords)
    add BUNIT=ADU to image HDU, some RADECSYS=ICRS which is NOT TRUE! Update to FK5 (should be true for all)
    Use standard header keywords for processing, overwriting with expected or found values:
        
        ORIGIN (set to either telescope or survey)
        USEABZP (set to the AB mag zero-point that will convert a pixel value from BUNIT to Jy, incorporating exposure time if necessary)
        BAND (for consistent naming of bands regardless of wavelength in the header)
    
    NOTE: Native (co-added) data units are in counts, requires correcting by integration time per exposure to get ADU.
    NOTE: Obtained central/effective wavelength and vega2ab conversions from: https://arxiv.org/pdf/astro-ph/0601592
    '''
    filt = fits.open(filepath)[0].header['FILTER'].strip()
    match filt:
        case 'Z':
            wl = 0.8817
            vega2ab = 0.528
            ext_filt = 'SDSS z'
        case 'Y':
            wl = 1.0305
            vega2ab = 0.634
            ext_filt = ['SDSS z', 'UKIRT J']
        case 'J':
            wl = 1.2483
            vega2ab = 0.938
            ext_filt = 'UKIRT J'
        case 'H':
            wl = 1.6313
            vega2ab = 1.379
            ext_filt = 'UKIRT H'
        case 'K':
            wl = 2.2010
            vega2ab = 1.900
            ext_filt = 'UKIRT K'
        case _:
            raise ValueError(f'Unmatched UKIRT band for filter: {filt}')
    
    if not coord:
        coord = get_image_centre_coord(filepath)
    tab = get_extinction_table(coord)
    if ext_filt:
        try:
            ext_corr = np.mean(tab[[filt in ext_filt for filt in tab['Filter_name']]]['A_SandF'])
        except Exception as e:
            ext_corr = 0
    else:
        ext_corr = 0

    update_header(filepath, 'WAVELEN', wl, 'image', clobber=True)
    update_header(filepath, 'BUNIT', 'ADU', 'image', clobber=True)
    update_header(filepath, 'RADECSYS', 'FK5', 'image', clobber=True)
    # update_header(filepath, 'EQUINOX', '2000', 'image', clobber=False)
    if survey: update_header(filepath, 'ORIGIN', survey, 'primary', clobber=True)   # Could instead check the headers to discern the survey?
    try:
        update_header(filepath, 'USEABZP', fits.open(filepath)[1].header['MAGZPT'] + vega2ab - ext_corr \
                                        + 2.5*np.log10(fits.open(filepath)[0].header['EXP_TIME']), 'image', clobber=True)
        update_header(filepath, 'PHOTSYS', 'AB', 'image', clobber=True)
    except Exception as e:
        print(f'[ERROR]\tupdate_header: {e}')
    update_header(filepath, 'BAND', filt, 'primary', clobber=True)

def get_KiDS():
    raise NotImplementedError

def fix_KiDS():
    raise NotImplementedError

def get_VLASS(data_dir: str, coord: SkyCoord, band: str='3GHz', query_radius: u.Quantity=10*u.arcsec,
            cutout_size: u.Quantity=1*u.arcmin, keep_full_image: bool=False):
    '''
    
    '''
    cadc = Cadc()
    result = cadc.query_region(coord, radius=query_radius, collection='VLASS')
    result.sort('time_exposure', reverse=True)
    result.sort('position_sampleSize')
    image_list = cadc.get_image_list(result, coordinates=coord, radius=cutout_size/2)
    image_list = [image_list[i] for i in np.argwhere(result['position_sampleSize'] < 1.1*np.min(result['position_sampleSize'])).ravel()]

    if len(image_list) == 0:
        print(f'[WARNING]\tNo VLASS image found.')

    if len(image_list) > 1:
        print('Multiple VLASS images found, selecting lowest RMS...')
        best_rms = None
        best_url = None
        for url in image_list:
            with fits.open(url, use_fsspec=False) as hdul:
                # Can iterate through the images in memory and get the one with the lowest RMS, sending it to download
                rms = np.sqrt(np.nanmean(sigma_clip(hdul[0].data, sigma=5, maxiters=10)**2))
                if not best_rms:
                    best_rms = rms
                    best_url = url
                else:
                    if rms < best_rms:
                        best_rms = rms
                        best_url = url
    else:
        best_url = image_list[0]
    
    if not os.path.exists(f'{data_dir}/Radio/VLASS'): os.mkdir(f'{data_dir}/Radio/VLASS')

    with fits.open(best_url, usefsspec=False) as hdul:
        hdul.writeto(f"{data_dir}/Radio/VLASS/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}" \
                    + f"_{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits")

def get_EMU(data_dir: str, coord: SkyCoord, username: str, band: str='888MHz', query_radius: u.Quantity=3*u.arcmin,
            cutout_size: u.Quantity=3*u.arcmin, keep_full_image: bool=False):
    '''
    
    '''
    casda = Casda()
    casda.login(username=username)
    try:
        result = Casda.query_region(coord, radius=query_radius)
    except Exception as e:
        print(f'\n[ERROR]\tNo query results returned: {e}')

    public_data = Casda.filter_out_unreleased(result)
    subset: Table = public_data[((public_data['obs_collection'] == 'EMU') & (public_data['dataproduct_type'] == 'cube') \
                        & (public_data['dataproduct_subtype'] == 'cont.restored.t0') \
                            & (np.char.endswith(public_data['filename'], 'cont.taylor.0.restored.conv.fits')) \
                            & (public_data['o_ucd'] == 'phot.flux.density') \
                                & np.char.startswith(public_data['filename'], 'image.i'))]
    
    if len(subset) == 0:
        print(f'[WARNING]\tNo EMU image found.')
        return

    # Grab row from subset with min distance (if subset not empty), also EMU only has 1 epoch per tile, so top row is most likely the best
    row = subset[[np.argmin(subset['distance'])]]
    # Get image cutout
    cutout_url = casda.cutout(row, coordinates=coord, radius=cutout_size/2)
    if not os.path.exists(f'{data_dir}/Radio/EMU'): os.mkdir(f'{data_dir}/Radio/EMU')
    cutout_file = casda.download_files(cutout_url, savedir=f'{data_dir}/Radio/EMU')
    for f in cutout_file:
        if 'checksum' not in f:
            os.rename(f, f"{data_dir}/Radio/EMU/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}" \
                + f"_{band}_{cutout_size.value}{cutout_size.unit}_cutout.fits")
        else:
            os.remove(f)

def get_TGSS():
    '''
    TODO
    https://hips.astron.nl/ASTRON/P/tgssadr
    '''
    raise NotImplementedError

def get_LoTSS(data_dir: str, coord: SkyCoord, query_radius: u.Quantity=3*u.arcmin,
              cutout_size: u.Quantity=180*u.arcsec, low: bool=False):
    '''
    See https://lofar-surveys.org/cutout_api_details.html
    adapted from
    https://github.com/mhardcastle/lotss-cutout-api/blob/main/cutout.py
    Alternatively, the worse(?) HiPS: https://hips.astron.nl/ASTRON/P/lotss_dr2_high could be
    used with astroquery.hips2fits to get cutouts.
    '''
    # Martin Hardcastle's cutout code
    base = 'dr2'
    url = 'https://lofar-surveys.org/'
    if low:
        page = base + '-low-cutout.fits'
    else:
        page = base + '-cutout.fits'

    if not os.path.exists(f'{data_dir}/Radio/LoTSS'): os.mkdir(f'{data_dir}/Radio/LoTSS')
    outname = f"{data_dir}/Radio/LoTSS/{coord.to_string(style='hmsdms', precision=2).replace(' ','')}_" \
                + f"{'low_' if low else ''}LoTSS_{cutout_size.value}{cutout_size.unit}_cutout.fits"

    r = requests.get(url + page, params={'pos': coord.to_string(style='hmsdms', precision=5, sep=':'),
                                       'size': cutout_size.to(u.arcmin).value}, stream=True)
    if r.status_code != 200:
        raise RuntimeError('Status code %i returned' % r.status_code)
    if r.headers['content-type'] != 'application/fits':
        raise RuntimeError('Server did not return FITS file, probably no coverage of this area')

    with open(outname,'wb') as f:
        f.write(r.content)
        r.close()

def get_LoLSS(coord: SkyCoord, cutout_size: u.Quantity=180*u.arcsec):
    '''
    HiPS previewer of DR1 available at https://lofar-surveys.org/public_hips/lolss
    Try to use astroquery.hips2fits to get cutouts? LoLSS HiPS is not in the default hips2fits list at CDS...
    '''
    raise NotImplementedError

    # Example
    hips_url = "https://lofar-surveys.org/public_hips/lolss"

    # Generate the FITS cutout (returns the FITS data directly)
    fits_data = hips2fits.query(hips_url, ra=coord.ra, dec=coord.dec, fov=cutout_size, format='fits')

    # Save the FITS data to a file (using astropy, if needed)
    fits.writeto('m101_cutout.fits', fits_data[0].data, fits_data[0].header)



default_online_archive_config = {
    '2MASS':
    {
        'survey_name': '2MASS',
        'available_bands': ['J','H','Ks'],
        'how_to_cite': 'https://www.ipac.caltech.edu/2mass/releases/allsky/',
        'notebook_example': 'https://caltech-ipac.github.io/irsa-tutorials/tutorials/irsa-sia-examples/sia_2mass_allsky.html',
        'download_function': get_2MASS,
        'apply_fixes_function': fix_2MASS
    },
    'SDSS':
    {
        'survey_name': 'SDSS-DR13',
        'available_bands': ['u','g','r','i','z'],
        'how_to_cite': 'https://www.sdss.org/collaboration/citing-sdss/',
        'download_function': get_SDSS,
        'apply_fixes_function': fix_SDSS
    },
    'AllWISE':
    {
        'survey_name': 'AllWISE',
        'available_bands': ['W1','W2','W3','W4'],
        'how_to_cite': 'https://caltech-ipac.github.io/irsa-tutorials/tutorials/irsa-sia-examples/sia_allwise_atlas.html#citations',
        'notebook_example': 'https://caltech-ipac.github.io/irsa-tutorials/tutorials/irsa-sia-examples/sia_allwise_atlas.html',
        'download_function': get_AllWISE,
        'apply_fixes_function': fix_AllWISE
    },
    'unWISE':
    {
        'survey_name': 'unWISE',
        'available_bands': ['W1','W2'],
        'how_to_cite': 'https://irsa.ipac.caltech.edu/data/WISE/unWISE/overview.html',
        'download_function': get_unWISE,
        'apply_fixes_function': fix_unWISE
    },
    'SEIP':
    {
        'survey_name': 'Spitzer Enhanced Imaging Products (SEIP)',
        'available_bands': ['IRAC1','IRAC2','IRAC3','IRAC4','MIPS24'],
        'how_to_cite': 'https://irsa.ipac.caltech.edu/data/SPITZER/Enhanced/SEIP/overview.html',
        'notebook_example': 'https://caltech-ipac.github.io/irsa-tutorials/tutorials/irsa-sia-examples/sia_seip.html',
        'download_function': get_SEIP,
        'apply_fixes_function': fix_SEIP
    },
    'EuclidMER':
    {
        'survey_name': 'Euclid Q1 MER mosaics',
        'available_bands': ['U','G','R','I','Z','VIS','Y','J','H'],
        'how_to_cite': 'https://irsa.ipac.caltech.edu/data/Euclid/docs/overview_ero.html',
        'notebook_example': 'https://caltech-ipac.github.io/irsa-tutorials/tutorials/euclid_access/1_Euclid_intro_MER_images.html',
        'download_function': get_Euclid_MER,
        'apply_fixes_function': fix_Euclid_MER
    },
    'Pan-STARRS1':
    {
        'survey_name': '',
        'available_bands': ['g','r','i','z','y'],
        'how_to_cite': '',
        'notebook_example': 'https://ps1images.stsci.edu/ps1image.html',
        'download_function': get_PanSTARRS,
        'apply_fixes_function': fix_PanSTARRS
    },
    'HSC':
    {
        'survey_name': 'HSC-SSP',
        'available_bands': ['g','r','i','z','y'],
        'how_to_cite': 'https://hsc-release.mtk.nao.ac.jp/doc/index.php/acknowledging-hsc__pdr3/',
        'download_function': get_HSC,
        'apply_fixes_function': fix_HSC,
        'login_file': f"/path/to/data/HSC.login"
    },
    'DECaLS':
    {
        'survey_name': 'DECam Legacy Survey',
        'available_bands': ['g','r','i','z'],
        'how_to_cite': 'https://www.legacysurvey.org/acknowledgment/',
        'download_function': get_DECaLS,
        'apply_fixes_function': fix_DECaLS
    },
    'DES':
    {
        'survey_name': 'Dark Energy Survey',
        'available_bands': ['g','r','i','z','Y'],
        'how_to_cite': 'https://des.ncsa.illinois.edu/thanks',
        'download_function': get_DES,
        'apply_fixes_function': fix_DES
    },
    'VISTA':
    {
        'survey_name': 'VISTA surveys',
        'available_bands': ['Z','Y','J','H','Ks'],
        'how_to_cite': 'https://ui.adsabs.harvard.edu/abs/2013Msngr.154...32E/abstract',
        'download_function': get_VISTA,
        'apply_fixes_function': fix_VISTA
    },
    'UKIRT':
    {
        'survey_name': 'UKIDSS surveys',
        'available_bands': ['Z','Y','J','H','K'],
        'how_to_cite': 'http://www.ukidss.org/archive/archive.html',
        'download_function': get_UKIRT,
        'apply_fixes_function': fix_UKIRT
    },
    'KiDS':
    {
        'survey_name': 'Kilo-Degree Survey',
        'available_bands': ['u','g','r','i',],
        'how_to_cite': '',
        'download_function': get_KiDS,
        'apply_fixes_function': fix_KiDS
    },
    'VLASS':
    {
        'survey_name': 'VLA Sky Survey',
        'available_bands': ['3GHz'],
        'how_to_cite': '',
        'download_function': get_VLASS
    },
    'EMU':
    {
        'survey_name': 'Evolutionary Map of the Universe',
        'available_bands': ['943MHz'],
        'how_to_cite': '',
        'download_function': get_EMU
    },
    'TGSS':
    {
        'survey_name': 'TIFR GMRT Sky Survey',
        'available_bands': ['150MHz'],
        'how_to_cite': '',
        'download_function': get_TGSS
    },
    'RACS-high':
    {

    },
    'LoTSS':
    {
        'survey_name': 'LOFAR Two-metre Sky Survey',
        'available_bands': ['144MHz'],
        'how_to_cite': '',
        'download_function': get_LoTSS,
    },
    'LoLSS':
    {
        'survey_name': 'LOFAR LBA Sky Survey',
        'available_bands': ['54MHz'],
        'how_to_cite': '',
        'download_function': get_LoLSS,
    }
}

