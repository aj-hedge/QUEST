import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from astropy.io import fits
import photutils as photutils
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, Angle, CartesianRepresentation, ICRS, FK5
from astropy.nddata import utils as nddata_utils
from reproject import reproject_interp
import reproject.mosaicking as mosaicking
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip
import astropy.units as u
from astropy.visualization import wcsaxes as wcsaxes
from astropy.visualization.mpl_normalize import simple_norm
import glob
from .lib_data import SourceEntry
from .lib_util import get_image_hdu, needs_realignment, generate_realigned_hdu, get_hdu_with, \
                        get_instrument, skycoord_in_image, validate_image
from .lib_photometry import PhotometryTool


def plot_stamp(in_ax: wcsaxes.WCSAxes, hdul: fits.HDUList, coord: SkyCoord, cutout_size: u.Quantity=5*u.arcsec,
            cmap: str='Greys', vmin: float=0.05, vmax: float=0.95, sigma_clip_norm: float|None=None, **kwargs) -> wcsaxes.WCSAxes:
    '''
    Create 2D cutout/stamp with field-of-view `cutout_size` about a target `coord`.
    '''
    im_hdu = get_image_hdu(hdul)
    im_wcs = WCS(im_hdu.header,fobj=hdul)
    # check if RA is for some reason on the y-axis and correct if so
    align_checks = needs_realignment(im_hdu.header)
    if align_checks != []:
        print(f"Re-aligning WCS to celestial north and target coord due to: {align_checks}")
        im_hdu = generate_realigned_hdu(im_hdu, coord)
        im_wcs = WCS(im_hdu.header, fobj=None)
    stamp = nddata_utils.Cutout2D(im_hdu.data,coord,(cutout_size,cutout_size),im_wcs)
    ax = plt.subplot(in_ax.get_subplotspec(),projection=stamp.wcs)
    del in_ax
    if sigma_clip_norm is not None:
        sigma_cut = sigma_clip(im_hdu.data,sigma=sigma_clip_norm, maxiters=10)
        ax.imshow(stamp.data,cmap=cmap,norm=simple_norm(im_hdu.data,vmin=np.nanmin(sigma_cut),vmax=np.nanmax(sigma_cut)))
    else:
        ax.imshow(stamp.data,cmap=cmap,norm=simple_norm(im_hdu.data,min_percent=vmin*100,max_percent=vmax*100))
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('Dec. (J2000)')
    # Flip RA to be increasing to the left if it is not
    ra_left = ax.wcs.pixel_to_world(0,0)
    ra_right = ax.wcs.pixel_to_world(10,0)
    if ra_left.ra < ra_right.ra:
        ax.invert_xaxis()
    ax.coords[0].set_ticklabel(exclude_overlapping=True)
    ax.coords[1].set_ticklabel(exclude_overlapping=True)
    return ax, stamp

def overlay_aperture(ax: wcsaxes.WCSAxes, ap_pos: SkyCoord, ap_diameter: u.Quantity, **plot_args) -> wcsaxes.WCSAxes:
    '''
    Overlay a `SkyCircularAperture` onto the provided axes using position (`SkyCoord`) and diameter (`Quantity`) specifications.
    '''
    ap = photutils.aperture.SkyCircularAperture(ap_pos,ap_diameter/2)
    ap = ap.to_pixel(ax.wcs)
    ap.plot(ax,**plot_args)
    return ax
    
def plot_overlays(phot_tool: PhotometryTool, save_path: str=None, stamp_radius: u.Quantity=15*u.arcsec,
                  optical_bands: list[str]=['g', 'r', 'i', 'z'], ir_bands: list[str]=['J', 'H', 'K', 'Ks'],
                  ignore_negative: bool=False, data_dir_radio: list[str]=[],
                  ignored_filepaths: set[str]=set(), **kwargs):
    """
    Creates optical/IR stacks and overlays radio continuum contours.
    
    Extra kwargs
    ----------
    sigma_clip_norm :   Sigma to clip to when determining `vmin`/`vmax` for image stretch.
    cmap    :   Colourmap to use in `imshow`. Default is 'Greys'.
    add_beam    :   Whether to add beam ellipse to radio contours. Default: `False`.
    contour_step    :   Overlaid contour step size (geometric progression of contour_step ** (0,1,2)), default: `np.sqrt(2)`.
                        First contour is at 3*RMS.
    """
    print(f"--- Generating overlay plots for {phot_tool.source.source_name} ---")
    
    optical_hdus, available_optical_bands = _get_hdus_for_bands(phot_tool=phot_tool,
                                                                bands=optical_bands,
                                                                cutout_size=stamp_radius*3,
                                                                ignore_negative=ignore_negative)
    ir_hdus, available_ir_bands = _get_hdus_for_bands(phot_tool=phot_tool,
                                                      bands=ir_bands,
                                                      cutout_size=stamp_radius*3,
                                                      ignore_negative=ignore_negative)

    if optical_hdus:
        _generate_stack_and_plot(se=phot_tool.source, hdus=optical_hdus, stack_name=available_optical_bands,
                                 title_prefix="Optical", save_path=save_path, cutout_size=stamp_radius*2,
                                 data_dir_radio=data_dir_radio, ignored_filepaths=ignored_filepaths, **kwargs)
    if ir_hdus:
        _generate_stack_and_plot(se=phot_tool.source, hdus=ir_hdus, stack_name=available_ir_bands,
                                 title_prefix="IR", save_path=save_path, cutout_size=stamp_radius*2,
                                 data_dir_radio=data_dir_radio, ignored_filepaths=ignored_filepaths, **kwargs)
    print("-" * 30)

def _get_hdus_for_bands(phot_tool: PhotometryTool, bands: list[str],
                        cutout_size: u.Quantity, ignore_negative: bool):
    hdus = []
    available_bands = []
    for band in bands:
        phot_key = f"{phot_tool.source.source_name}_{band}"
        summary = phot_tool.photometry_data.get(phot_key)
        if not summary:
            continue
        if summary['flux'] + summary['flux_err'] < 0 and ignore_negative == True:
            continue
        entry = phot_tool.source.best_data.get(band)
        if entry:
            hdul = fits.open(entry.filepath)
            hdu = get_image_hdu(hdul)
            wcs = WCS(hdu.header)
            cutout = nddata_utils.Cutout2D(hdu.data, phot_tool.source.host_coord, cutout_size, wcs=wcs)
            hdu.data = cutout.data
            hdu.header.update(cutout.wcs.to_header())
            hdus.append(hdu)
            available_bands.append(band)
    return hdus, ''.join(available_bands)

def _generate_stack_and_plot(se: SourceEntry, hdus: list[fits.ImageHDU], stack_name: str, title_prefix: str,
                                save_path: str, cutout_size: u.Quantity, data_dir_radio: list[str],
                                ignored_filepaths: set[str], **kwargs):
    wcs_stack, shape_stack = mosaicking.find_optimal_celestial_wcs(hdus)
    data_stacked, _ = mosaicking.reproject_and_coadd(hdus, wcs_stack, shape_out=shape_stack, reproject_function=reproject_interp)

    cutout = nddata_utils.Cutout2D(data_stacked, se.host_coord, cutout_size, wcs=wcs_stack)
    
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={'projection': cutout.wcs})
    
    sigma_clipped = sigma_clip(cutout.data, sigma=kwargs.get('sigma_clip_norm', 5))

    ax.imshow(cutout.data, origin='lower', cmap=kwargs.get('cmap', 'Greys'),
                norm=simple_norm(cutout.data, vmin=np.nanmin(sigma_clipped), vmax=np.nanmax(sigma_clipped)))
    
    ax.set_title(f"{title_prefix} ({stack_name}) Stack", fontsize='x-large')
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('Dec. (J2000)')
    # Retain imshow axes lims after adding contours, in case the optical/IR cutout is trimmed and the radio cutout is larger.
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Overlay radio contours
    # TODO: Update SourceEntry to also keep track of available (deepest) radio images for each source and iterate over them here + label
    contour_colour_iter = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    beam_pos_iter = iter(['bottom left', 'bottom right', 'top left' ,'top right', 'bottom', 'top', 'left', 'right'])
    handles, labels = ax.get_legend_handles_labels()
    
    if len(data_dir_radio) > 1:
        radio_paths = set()
        for radio_dir in data_dir_radio:
            radio_paths.update(glob.glob(f'{radio_dir}/*.fits'))
            radio_paths.update(glob.glob(f'{radio_dir}/**/*.fits'))
    elif len(data_dir_radio) == 1:
        radio_paths = set()
        radio_paths.update(glob.glob(f'{data_dir_radio[0]}/*.fits'))
        radio_paths.update(glob.glob(f'{data_dir_radio[0]}/**/*.fits'))
    else:
        radio_paths = set()
    radio_cutout = None

    for radio_path in radio_paths:
        if not any([fp in radio_path for fp in ignored_filepaths]):
            try:
                if skycoord_in_image(radio_path, se.host_coord):
                    with fits.open(radio_path) as radio_hdul:
                        if not validate_image(get_image_hdu(radio_hdul), min_fill_factor=0.2):
                            print(f'[DEBUG] {radio_path} could not be validated.')
                            continue
                else:
                    continue
            except Exception as e:
                print(f"Warning: Could not process radio image {radio_path} for overlays: {e}")
                continue

            radio_hdul = fits.open(radio_path)
            radio_hdu = get_image_hdu(radio_hdul)
            instrument = get_instrument(radio_hdul)
            # If the radio image does not have good enough beam resolution relative to the stamp size, continue
            bmaj = radio_hdu.header.get('BMAJ')
            bmin = radio_hdu.header.get('BMIN')
            if not bmaj or not bmin:
                continue
            if max(bmaj, bmin) > cutout_size.to(u.deg).value / 4:
                continue
            radio_wcs = WCS(radio_hdu.header, naxis=2)
            # Ensure we deal with stokes I images, take first element of last axis
            radio_data = radio_hdu.data
            if radio_data.ndim > 2:
                radio_data = radio_data[0,...]
            # No support for cubes yet, so hopefully taking the first element of the frequency axis is OK
            if radio_data.ndim > 2:
                radio_data = radio_data[0,...]

            radio_cutout = nddata_utils.Cutout2D(np.squeeze(radio_data), se.host_coord, cutout_size, wcs=radio_wcs)
            mask = ~np.isnan(radio_cutout.data)
            if np.all(mask == False):
                continue
            rms = np.sqrt(np.nanmean(sigma_clip(radio_cutout.data[~np.isnan(radio_cutout.data)], sigma=3, maxiters=10)**2))
            levels = rms * 3 * kwargs.get('contour_step', np.sqrt(2))**np.arange(0, 3, 1)
            contour_colour = next(contour_colour_iter)
            ax.contour(radio_cutout.data, levels=levels, transform=ax.get_transform(radio_cutout.wcs),
                        colors=contour_colour, alpha=0.6)
            if kwargs.get('add_beam', False) == True:
                wcsaxes.add_beam(ax,header=radio_hdu.header,corner=next(beam_pos_iter),frame=False,
                                    edgecolor=contour_colour,hatch='////',fill=True,facecolor='white')
            if radio_cutout:    # QuadContourSet (from ax.contour call) does not support labels directly
                handles.append(patches.Patch(color=contour_colour,label=f"{instrument}"))

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    if radio_cutout:
        ax.legend(handles=handles, loc='upper left', fontsize='large', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    if save_path:
        plt.savefig(f"{save_path}/{se.short_label}_{stack_name}_overlay.png", bbox_inches='tight', dpi=200)
    plt.show()

def plot_photometry(phot: PhotometryTool, save_path: str=None, ignore_negative: bool=False, yscale: str='symlog',
                    xscale: str='log'):
    """
    Generates and optionally saves the SED plot.
    """
    if not phot.photometry_data:
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for data in phot.photometry_data.values():
        if data['flux'] + data['flux_err'] < 0:
            if ignore_negative == True:
                continue
            else:
                ax.errorbar(data['wavelength'].to(u.um).value, data['flux'].to(u.uJy).value, 
                    yerr=data['flux_err'].to(u.uJy).value, ls=' ', marker='.', c='red',
                    mec=None, mfc='red', ms=10, capsize=2)
        else:
            ax.errorbar(data['wavelength'].to(u.um).value, data['flux'].to(u.uJy).value, 
                        yerr=data['flux_err'].to(u.uJy).value, ls=' ', marker='.', c='dodgerblue',
                        mec=None, mfc='dodgerblue', ms=10, capsize=2)
    display_name = data['display_name']
        
    ax.set_xlabel(r'$\lambda$ / $\mu$m', fontsize='xx-large')
    ax.set_ylabel(r'$F_\nu$ / $\mu$Jy', fontsize='xx-large')
    ax.text(0.05, 0.90, f"{display_name}", fontsize='x-large', fontweight='bold', color='black', transform=ax.transAxes)
    ax.axhline(0, color='grey', linestyle='dashed', alpha=0.7)
    
    assert xscale in ['linear', 'log'], "xscale must be 'linear' or 'log'"
    assert yscale in ['linear', 'log', 'symlog'], "yscale must be 'linear', 'log' or 'symlog'"

    ax.set_xscale(xscale)
    if yscale == 'symlog':
        ax.set_yscale(yscale, linthresh=1e0)
    else:
        ax.set_yscale(yscale)
    if xscale != 'linear' and yscale != 'linear':
        ax.minorticks_on()

    # ax.grid(True,'both','both')
    fluxes = [data['flux'][0].to(u.uJy).value for data in phot.photometry_data.values() \
                if data['flux'] + data['flux_err'] > 0 or ignore_negative == False]
    ax.set_ylim(min(np.floor(min(fluxes)*0.9**np.sign(min(fluxes))), 0.2*min(fluxes)), max(np.ceil(max(fluxes)*1.1), 0))

    # Get the major locator object matplotlib is using for the y-axis
    y_major_locator = ax.yaxis.get_major_locator()

    # Get the tick locations that the locator has automatically determined
    major_ticks = y_major_locator.tick_values(*ax.get_ybound())

    # Add the zero tick to the list of major ticks if it's not already there
    if 0 not in major_ticks:
        major_ticks = np.append(major_ticks, 0)
        major_ticks.sort()

    # Set a new locator for the y-axis that places ticks at our desired locations
    ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))

    if yscale == 'symlog':
        # Create a MinorSymLogLocator instance, passing the same linthresh
        minor_locator = ticker.SymmetricalLogLocator(ax.yaxis.get_transform(), subs=np.arange(2,10,1))

        # Set the minor locator for the y-axis
        minor_ticks = minor_locator.tick_values(*ax.get_ybound())
        minor_ticks = np.append(minor_ticks, np.arange(-0.9, 1.0, 0.1))
        minor_ticks.sort()

        # ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    
    if save_path:
        plt.savefig(f"{save_path}/{display_name}_fluxes.pdf")
    plt.show()

def plot_multiwl_stamps(phot: PhotometryTool, cutout_size: u.Quantity=10*u.arcsec, save_path: str=None, ax_ticks_off: str='xy',
                        ignore_negative: bool=False, **kwargs):
    """
    Generates and optionally saves multi-wavelength stamp plots.
    """
    if not phot.photometry_data: return
    
    # Sort by wavelength
    sorted_phot = dict(sorted(phot.photometry_data.items(), key=lambda item: item[1]['wavelength']))
    
    n_data = len(sorted_phot)
    for summary in sorted_phot.values():
        if summary['flux'] + summary['flux_err'] < 0 and ignore_negative == True:
            n_data -= 1
    fig, axs = plt.subplots(1, n_data, figsize=(4*n_data, 3))
    axs = np.atleast_1d(axs)

    ax_idx = 0
    for _, (source_name, summary) in enumerate(sorted_phot.items()):
        if summary['flux'] + summary['flux_err'] < 0 and ignore_negative == True:
            continue

        hdul = fits.open(summary['filename'])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax, stamp = plot_stamp(axs[ax_idx], hdul, summary['skycoord'], cutout_size=cutout_size, **kwargs)
        
        if ax_idx == 0:
            ax.text(0.05, 0.90, summary['display_name'], color='white', fontsize='x-large', fontweight='heavy',
                    bbox=dict(boxstyle="round", ec='white', fc='black'), transform=ax.transAxes)
        
        overlay_aperture(ax, ap_pos=summary['skycoord'], ap_diameter=kwargs.get('ap_diameter', 2*u.arcsec), color='red', lw=1.5)
        
        ax.plot(*stamp.wcs.world_to_pixel(summary['radio_coord']), marker='x', color='yellow', ms=6, transform=ax.get_transform(stamp.wcs))

        ax.set_title(rf"{summary['origin']} ${summary['band']}$, $\lambda_c$={summary['wavelength']}", fontsize='large')

        if 'y' in ax_ticks_off:
            ax.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False)
            plt.subplots_adjust(wspace=0)
            fig.set_size_inches(3*n_data,3)
        if 'x' in ax_ticks_off:
            ax.tick_params(axis='x',which='both',left=False,right=False,labelleft=False)
        
        hdul.close()

        ax_idx += 1

    if save_path:
        plt.savefig(f"{save_path}/{summary['display_name']}_stamps.png", bbox_inches='tight', dpi=200)
    plt.show()

def plot_all_stamps(se: SourceEntry, cutout_size: u.Quantity=10*u.arcsec, save_path: str=None,
                    ax_ticks_off: str='xy', **kwargs):
    """
    Generates and optionally saves ALL stamp plots containing the source (in its register).
    """
    if len(se.containing_images) == 0: return
    
    valid_images = list(se.containing_images)
    wavelengths = []

    for f in se.containing_images:
        hdul = fits.open(f)
        try:
            wl = get_hdu_with(hdul, 'WAVELEN').header['WAVELEN']
        except Exception as e:
            valid_images.remove(f)
            hdul.close()
            continue
        wavelengths.append(wl)
        hdul.close()

    # Sort by wavelength
    sorted_data = sorted(zip(valid_images, wavelengths), key=lambda item: item[1])
    sorted_images = [x for x, _ in sorted_data]
    wavelengths = [x for _, x in sorted_data]
    n_data = len(sorted_images)
    rows = (n_data - 1)// 6 + 1
    if rows == 0: rows = 1
    cols = min([n_data, 6])

    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    fig.suptitle(se.radio_coord.to_string(style='hmsdms'), fontsize='xx-large', fontweight='heavy')
    axs = np.atleast_1d(axs)
    axs = axs.ravel()

    for i, f in enumerate(sorted_images):
        hdul = fits.open(f)
        origin = get_hdu_with(hdul, 'ORIGIN').header.get('ORIGIN', '')
        band = get_hdu_with(hdul, 'BAND').header.get('BAND', '')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax, stamp = plot_stamp(axs[i], hdul, se.host_coord, cutout_size=cutout_size, **kwargs)
        
        ax.plot(*stamp.wcs.world_to_pixel(se.radio_coord), marker='x', color='yellow', ms=6, transform=ax.get_transform(stamp.wcs))
        ax.plot(*stamp.wcs.world_to_pixel(se.host_coord), marker='o', ms=14, mfc='none', mec='blue', mew=0.5, transform=ax.get_transform(stamp.wcs))

        ax.set_title(rf"{origin} ${band}$, $\lambda_c$={wavelengths[i]} um", fontsize='medium')

        if 'y' in ax_ticks_off:
            ax.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False)
            plt.subplots_adjust(wspace=0)
        if 'x' in ax_ticks_off:
            ax.tick_params(axis='x',which='both',left=False,right=False,labelleft=False)
            plt.subplots_adjust(hspace=0.1)
        
        hdul.close()
    
    for i in range(len(axs),n_data,-1):
        axs[i-1].axis('off')

    if save_path:
        plt.savefig(f"{save_path}/debug_{se.source_name}_all_stamps.pdf", bbox_inches='tight')
    plt.show()