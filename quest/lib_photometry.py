import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import photutils as photutils
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, Angle, CartesianRepresentation, ICRS, FK5
from astropy.nddata import utils as nddata_utils
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip
import astropy.units as u
from astropy.visualization import wcsaxes as wcsaxes
from astropy.visualization.mpl_normalize import simple_norm
from .lib_data import SourceEntry
from .lib_util import convert_flux, get_astronomy_method, get_wavelength, get_image_hdu, get_hdu_with, \
                        needs_realignment, generate_realigned_hdu, get_mean_local_bkg



class PhotometryTool:
    """
    A self-contained tool to perform photometry, plotting, and data export for a *single* source.
    Optionally also provide output_fig_directory: str='./relative/path/to/figures' to control
    where (debug) figures are saved. Default is './figures'.
    """
    def __init__(self, source_entry: SourceEntry, data_dir_radio: list[str]=[], output_fig_dir: str='./figures'):
        self.source = source_entry
        self.photometry_data = {}
        self.data_dir_radio = data_dir_radio
        self.output_fig_dir = output_fig_dir

    def perform_photometry(self, ap_diameter: u.Quantity=2*u.arcsec, n_rand_aps: int=1000, ap_max_dist: u.Quantity=60*u.arcsec,
                           keep_negative: bool=True, rand_aper_flux_sigma_clip: float=10, debug: bool=False):
        print(f"--- Performing photometry for {self.source.source_name} ---")
        for band, data_entry in self.source.best_data.items():
            try:
                hdul = fits.open(data_entry.filepath)
                
                im_hdu = get_image_hdu(hdul)
                if not im_hdu: continue

                wcs = WCS(im_hdu.header, naxis=2)
                astronomy_method = get_astronomy_method(hdul)
                wl = get_wavelength(hdul)

                flux, local_bkg = self.get_phot(im_hdu.data, wcs, self.source.host_coord, ap_diameter)
                flux_jy = convert_flux(flux, astronomy_method, hdul, ap_diameter, band=band,
                                       ap_corrections=data_entry.aperture_corrections)
                # Clip the random aperture fluxes at 10-sigma to safeguard against clipping mildly strong artefacts/spurious pixels
                # which would then lower the uncertainty estimate unreasonably.
                flux_std, flux_bkg = self.rand_apertures(im_hdu.data, hdul, wcs, [self.source.host_coord], n_rand_aps,
                                                  ap_max_dist, ap_diameter, sigma_clipping=rand_aper_flux_sigma_clip,
                                                  debug=debug, debug_figure_dir=self.output_fig_dir)#, self.get_zp(hdul, band=band))
                flux_err_jy = convert_flux(flux_std, astronomy_method, hdul, ap_diameter, band=band,
                                       ap_corrections=data_entry.aperture_corrections)
                # flux_bkg = convert_flux(flux_bkg, astronomy_method, hdul, ap_diameter, band=band,
                #                       ap_corrections=data_entry.aperture_corrections)

                if flux_jy + flux_err_jy < 0 and not keep_negative:
                    hdul.close()
                    continue
                
                # TODO: record aperture size/shape information as well. Could expand in future
                #       to have per-aperture flux and flux_err for a list of input apertures.
                self.photometry_data[f"{self.source.source_name}_{band}"] = {
                    'filename': data_entry.filepath,
                    'display_name': self.source.short_label,
                    'astronomy_method': astronomy_method,
                    'instrument': data_entry.instrument,
                    'origin': data_entry.origin,
                    'skycoord': self.source.host_coord,
                    'radio_coord': self.source.radio_coord,
                    'wavelength': wl,
                    'band': band,
                    'flux': flux_jy,
                    'flux_err': flux_err_jy
                }
                
                hdul.close()

            except Exception as e:
                print(f"[ERROR]\tperform_photometry: Could not perform photometry on {data_entry.filepath}: {e}")
        print("-" * 30)

    def print_photometry(self, out_type: str='str', out_file: str=None):
        '''
        Decorated printing of all photometry dictionaries in list.
        `phot` dictionary structure is `source_name_N` : {`filename`, `display_name`, `astronomy_method`, `instrument`,
                                                            `skycoord`, `wavelength`, `band`, `flux`, `flux_err`}
        
        Parameters
        ----------
        - out_type : `str`
            Printing style. Can be 'str', 'latex' or 'beagle'.
        - out_file : `str`
            Path to output file, otherwise output is to stdout.
        '''

        if not self.photometry_data:
            return

        photometries = [self.photometry_data] # Adapt to original function's list input
        
        header_str = ''
        data_str = ''

        if out_type == 'beagle':
            header_str += '#band,wavelength,rms_uJy,flux_uJy\n'
        elif out_type == 'latex':
            header_str += "Source "
            # Before adding the bands, sort by wavelength
            photometries[0] = dict(sorted(photometries[0].items(), key=lambda item: item[1]['wavelength']))
            for key in photometries[0].keys():
                header_str += rf"& ${photometries[0][key]['band']}$ "
            header_str += '\\\\\n'
            for _ in photometries[0].keys():
                header_str += r"& [\textmu{}Jy] "
            header_str += '\\\\\n'
        elif out_type == 'str':
            header_str += "Source"
            for key in photometries[0].keys():
                header_str += f", {photometries[0][key]['wavelength']}"    # Assumes homogeneous photometry for all sources
            header_str += '\n'
        else:
            raise ValueError(f"{out_type} not supported.")

        file = None
        if out_file != None:
            file = open(out_file,'a')

        if out_type == 'beagle':
            if out_file != None:
                file.close()
            for phot in photometries:
                data_str = ''
                for key in phot.keys():
                    data_str = data_str + f"{phot[key]['instrument']}_{phot[key]['band']},{phot[key]['wavelength'].to(u.um).value}," \
                                        + f"{phot[key]['flux_err'][0].to(u.uJy).value},{phot[key]['flux'][0].to(u.uJy).value}\n"
                
                file = None
                if out_file != None:
                    file = open(f"{out_file}_{phot[next(iter(phot))]['display_name']}.asc",'w')
                print(phot[key]['display_name'])
                print(header_str+data_str, file=file)
                if out_file != None:
                    file.close()
        elif out_type == 'latex':
            for phot in photometries:
                data_str += f"{phot[next(iter(phot))]['display_name']}"
                # before adding the fluxes, sort by wavelength
                phot = dict(sorted(phot.items(), key=lambda item: item[1]['wavelength']))
                for key in phot.keys():
                    data_str += f" & ${phot[key]['flux'][0].to(u.uJy).value:.2f}\\pm{phot[key]['flux_err'][0].to(u.uJy).value:.2f}$"
                data_str += '\\\\\n'
            print(header_str + data_str, file=file)
            if out_file != None:
                file.close()
        elif out_type == 'str':
            for phot in photometries:
                data_str += f"{phot[next(iter(phot))]['display_name']}"
                for key in phot.keys():
                    data_str += f", {phot[key]['flux'][0]:.3e}+-{phot[key]['flux_err'][0]:.3e}"
                data_str += '\n'
            print(header_str + data_str, file=file)
            if out_file != None:
                file.close()
        return
    
    @staticmethod
    def print_all_photometry(photometries: list[dict], out_type: str='str', out_file: str=None):
        '''
        Decorated printing of all photometry dictionaries in list.
        `phot` dictionary structure is `sourcename_BAND` : {`filename`, `display_name`, `astronomy_method`, `instrument`,
                                                            `skycoord`, `wavelength`, `band`, `flux`, `flux_err`}
        I.e. a dict of { `sourcename_BAND`: ... } for each BAND of a source, all appended to a list of dicts for each source.
        
        Parameters
        ----------
        - out_type : `str`
            Printing style. Can be 'str', 'latex' or 'beagle'.
        - out_file : `str`
            Path to output file, otherwise output is to stdout.
        '''
        header_str = ''
        data_str = ''

        if out_type == 'beagle':
            header_str += '#band,wavelength,rms_uJy,flux_uJy\n'
        elif out_type == 'latex':
            header_str += "Source "
            
            # Firstly, get out all the unique band and wavelengths (roughly)
            bands = {}
            for phot_source in photometries:
                for phot_band in phot_source.values():
                    bands[phot_band['band']] = phot_band['wavelength']

            # # Secondly, sort by wavelength
            band_order = sorted(bands, key=bands.get)

            # photometries[0] = dict(sorted(photometries[0].items(), key=lambda item: item[1]['wavelength']))
            for band in band_order:
                header_str += rf"& ${band}$ "
            header_str += '\\\\\n'
            for _ in band_order:
                header_str += r"& [\textmu{}Jy] "
            header_str += '\\\\\n'
        elif out_type == 'str':
            header_str += "Source"

            # Firstly, get out all the unique band and wavelengths (roughly)
            bands = {}
            for phot_source in photometries:
                for phot_band in phot_source.values():
                    bands[phot_band['band']] = phot_band['wavelength']

            # # Secondly, sort by wavelength
            band_order = sorted(bands, key=bands.get)

            for band in band_order:
                header_str += f",{bands[band]: >19}"    # Assumes homogeneous photometry for all sources
            header_str += '\n'
        else:
            raise ValueError(f"{out_type} not supported.")

        if out_type == 'beagle':
            for phot in photometries:
                data_str = ''
                for key in phot.keys():
                    data_str = data_str + f"{phot[key]['instrument']}_{phot[key]['band']},{phot[key]['wavelength'].to(u.um).value}," \
                                        + f"{phot[key]['flux_err'][0].to(u.uJy).value},{phot[key]['flux'][0].to(u.uJy).value}\n"
                
                file = None
                if out_file != None:
                    file = open(f"{out_file}_{phot[next(iter(phot))]['display_name']}.asc",'w')
                # print(phot[key]['display_name'])
                print(header_str+data_str, file=file)
                if out_file != None:
                    file.close()
        elif out_type == 'latex':
            file = None
            if out_file != None:
                file = open(out_file,'a')
            for phot_source in photometries:
                source_display = phot_source[next(iter(phot_source))]['display_name']
                source_name = '_'.join(next(iter(phot_source)).split('_')[:-1])
                data_str += f"{source_display}"
                # No need to sort if we are using .get() from the dict
                for band in band_order:
                    if phot_source.get(f"{source_name}_{band}"):
                        phot = phot_source[f"{source_name}_{band}"]
                        data_str += f" & ${phot.get('flux')[0].to(u.uJy).value:.2f}\\pm{phot.get('flux_err')[0].to(u.uJy).value:.2f}$"
                    else:
                        data_str += f" &"
                # phot = dict(sorted(phot.items(), key=lambda item: item[1]['wavelength']))
                # for key in phot.keys():
                #     data_str += f" & ${phot[key]['flux'][0].to(u.uJy).value:.2f}\\pm{phot[key]['flux_err'][0].to(u.uJy).value:.2f}$"
                data_str += '\\\\\n'
            print(header_str + data_str, file=file)
            if out_file != None:
                file.close()
        elif out_type == 'str':
            file = None
            if out_file != None:
                file = open(out_file,'a')
            for phot_source in photometries:
                source_display = phot_source[next(iter(phot_source))]['display_name']
                source_name = next(iter(phot_source)).split('_')[0]
                data_str += f"{source_display}"
                # No need to sort if we are using .get() from the dict
                for band in band_order:
                    if phot_source.get(f"{source_name}_{band}"):
                        phot = phot_source[f"{source_name}_{band}"]
                        flux_str = f"{phot.get('flux')[0].value:.3e}+-{phot.get('flux_err')[0].value:.3e}"
                        data_str += f",{flux_str: >22}"
                    else:
                        data_str += f",{'--': >22}"
                # for key in phot.keys():
                #     data_str += f", {phot[key]['flux'][0]:.3e}+-{phot[key]['flux_err'][0]:.3e}"
                data_str += '\n'
            print(header_str + data_str, file=file)
            if out_file != None:
                file.close()
        return

    @staticmethod
    def rand_xy(dist: u.Quantity, npos: int) -> np.ndarray:
        '''
        Generate `npos` random offsets up to `dist` away in both x/y.
        `dist` must be an angular quantity, and the output offsets will be in degrees.
        NOTE: Beware of RA wrapping/illegal values when combining offsets with positions near the celestial extrema.
        '''
        dx = np.random.uniform(-dist.to(u.deg).value,dist.to(u.deg).value,npos) * u.deg
        dy = np.random.uniform(-dist.to(u.deg).value,dist.to(u.deg).value,npos) * u.deg
        return dx, dy

    @staticmethod
    def rand_apertures(data: np.ndarray, hdul: fits.HDUList, wcs: WCS, coords: SkyCoord,
                    npos: int, rand_dist: u.Quantity, ap_diameter: u.Quantity,
                    sigma_clipping: int=5, debug: bool=False, debug_figure_dir: str='./figures') -> tuple[np.ndarray,np.ndarray]:
        '''
        Place down random apertures (annuli) relative to target coordinates to sample the statistical
        mean and standard deviation of fluxes (background).
        '''
        dx, dy = PhotometryTool.rand_xy(rand_dist,npos)

        flux_std = []
        bkg_mean = []
        for coord in coords:
            ypos = coord.dec + dy
            xpos = coord.ra + (dx*np.cos(ypos.to(u.rad).value))
            rad_sys = 'icrs'    # Default
            if get_hdu_with(hdul,'RADECSYS'): rad_sys = get_hdu_with(hdul,'RADECSYS').header['RADECSYS'].strip(' ').lower()
            elif get_hdu_with(hdul,'RADESYS'): rad_sys = get_hdu_with(hdul,'RADESYS').header['RADESYS'].strip(' ').lower()
            n_coords = SkyCoord(xpos,ypos,frame=rad_sys)
            if debug:
                hdu = get_image_hdu(hdul)
                if needs_realignment(hdu.header):
                    hdu = generate_realigned_hdu(hdu, coord)
                wcs = WCS(hdu.header, naxis=2)
                cutout = nddata_utils.Cutout2D(hdu.data, coord, rand_dist*2+ap_diameter, wcs)
                rand_flux, rand_bkg, apertures = PhotometryTool.get_phot(cutout.data,cutout.wcs,n_coords,ap_diameter,debug=debug)
            else:
                rand_flux, rand_bkg = PhotometryTool.get_phot(data,wcs,n_coords,ap_diameter,debug=debug)

            flux_clipped: np.ma.MaskedArray = sigma_clip(rand_flux,sigma=sigma_clipping,maxiters=10)

            if debug:
                origin = PhotometryTool.get_hdu_with(hdul, 'ORIGIN').header.get('ORIGIN', '')
                band = PhotometryTool.get_hdu_with(hdul, 'BAND').header.get('BAND', '')
                print(f"[DEBUG]\t{origin} {band}: Within {rand_dist} about {coord.to_string(style='hmsdms')}," \
                      +f"\n {npos} random {ap_diameter} apertures have flux stdv of {np.std(flux_clipped, ddof=1):.2e}" \
                      +f"\nand mean {sigma_clipping}-sigma clipped annuli background of {np.mean(rand_bkg):.2e} subtracted (their unique values).")
                # Simple plot of image overlaid with the apertures, use masked array to determine aperture colour (i.e. if it was clipped)
                plt.clf()
                ax = plt.subplot(1,1,1,projection=cutout.wcs)
                ax.imshow(cutout.data, cmap='Greys', norm=simple_norm(cutout.data, percent=99.5))
                good_aps = apertures[flux_clipped.mask == False]
                clipped_aps = apertures[flux_clipped.mask == True]
                good_aps.plot(color='green',lw=1,alpha=0.6,ax=ax)
                clipped_aps.plot(color='red',lw=1,alpha=0.6,ax=ax)
                ax.scatter_coord(coord, marker='x', color='blue', s=20, label='Target')
                ax.set_title(f"{origin} {band}: {npos} random apertures within {rand_dist},\nclipped at {sigma_clipping} sigma")
                plt.savefig(f"{debug_figure_dir}/debug_randapertures_{coord.to_string(style='hmsdms')}_{origin}_{band}_{rand_dist}.png")
                ax.remove()
            # add 10% to flux uncertainty (equivalent to 0.1 mag)
            flux_std.append(np.std(flux_clipped,ddof=1) * 1.1)
            bkg_mean.append(np.mean(rand_bkg))

        return u.Quantity(flux_std), u.Quantity(bkg_mean)

    @staticmethod
    def get_phot(data: np.ndarray, wcs: WCS, coords: SkyCoord, ap_diameter: u.Quantity, debug: bool=False) -> tuple[float,float]:
        '''
        Extract the apertue flux and annulus background for given coordinates,
        subtracting the local background from the aperture flux. (NOTE: No flux conversion here)
        '''
        # x, y = wcs.world_to_pixel(coords)
        ap = photutils.aperture.SkyCircularAperture(coords,r=ap_diameter/2)
        pix_ap = ap.to_pixel(wcs)
        phot_table = photutils.aperture.aperture_photometry(data,pix_ap)
        local_bkg = get_mean_local_bkg(data,wcs,coords,1/2*ap_diameter,3/2*ap_diameter)
        flux = phot_table['aperture_sum'] - pix_ap.area*local_bkg
        if debug:
            return flux, local_bkg, pix_ap
        return flux, local_bkg

