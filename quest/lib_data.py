import numpy as np
import warnings
import pickle
from contextlib import contextmanager, redirect_stdout
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
import photutils as photutils
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, Angle, CartesianRepresentation, ICRS, FK5
from astropy.nddata import utils as nddata_utils
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip
import astropy.units as u
import astropy.convolution as convolution
from astropy.visualization import wcsaxes as wcsaxes
from astropy.visualization.mpl_normalize import simple_norm
from sys import stdout
import glob
from .lib_util import *



class DataEntry:
    """
    Represents a single best-data file.
    Providing aperture radii and associated aperture corrections will let them be applied whenever flux
    is extracted from the DataEntry (in the form aperture_corrections=np.array([[radii], [corrections]])).
    This can also be set after the DataTool has created the DataEntry instances.
    """
    def __init__(self, filepath: str, astronomy_method: str, instrument: str, origin: str, band: str,
                 wavelength: u.Quantity, rms: float, pix_res: float, aperture_corrections: np.ndarray=None):
        self.filepath = filepath
        self.astronomy_method = astronomy_method
        self.instrument = instrument
        self.origin = origin
        self.band = band
        self.wavelength = wavelength
        self.rms = rms
        self.pix_res = pix_res
        if aperture_corrections:
            self.aperture_corrections = aperture_corrections
        else:
            self.aperture_corrections = None

    def __repr__(self) -> str:
        return f"""{type(self).__name__}({self.filepath=!r}
        \t\t{self.astronomy_method=!r}
        \t\t{self.instrument=!r}
        \t\t{self.band=!r}
        \t\t{self.wavelength=!r}
        \t\t{self.rms=!r}
        \t\t{self.pix_res=!r})\n\t\t\t"""
    
    def set_aperture_corrections(self, aperture_corrections: dict):
        self.aperture_corrections = aperture_corrections
    
    def calculate_rms(self) -> float:
        return calculate_rms(hdul=fits.open(self.filepath))
    
    def get_pix_res(self) -> u.Quantity:
        return get_pix_res(hdul=fits.open(self.filepath))
    
    def get_astronomy_method(self) -> str:
        return get_astronomy_method(hdul=fits.open(self.filepath))
    
    def get_instrument(self) -> str:
        return get_instrument(hdul=fits.open(self.filepath))

    def get_wavelength(self) -> u.Quantity:
        return get_wavelength(hdul=fits.open(self.filepath))

    def get_band(self) -> str:
        return get_band(hdul=fits.open(self.filepath))
    
    def get_zp(self, tmass_corr=None, band='') -> float:
        return get_zp(hdul=fits.open(self.filepath), tmass_corr=tmass_corr, band=band)

    def get_hdu_with(self, keyword: str) -> fits.FitsHDU:
        '''
        Iterate through a HDUList and return the first HDU that contains the required CARD via keyword.
        '''
        return get_hdu_with(hdul=fits.open(self.filepath), keyword=keyword)
    
    def get_image_hdu(self, return_idx: bool=False) -> fits.ImageHDU:
        return get_image_hdu(hdul=fits.open(self.filepath), return_idx=return_idx)

    def validate_image(self, min_fill_factor: float=0.4) -> bool:
        return validate_image(im_hdu=self.get_image_hdu(), min_fill_factor=min_fill_factor)

    def skycoord_in_image(self, coord: SkyCoord) -> bool:
        return skycoord_in_image(fits_file=self.filepath, coord=coord)

    def needs_realignment(self, tolerance: float = 1e-6) -> list:
        return needs_realignment(header=self.get_image_hdu().header, tolerance=tolerance)
    
    def generate_realigned_hdu(self, center_coord: SkyCoord) -> fits.ImageHDU:
        return generate_realigned_hdu(original_hdu=self.get_image_hdu(), center_coord=center_coord)

class SourceEntry:
    """
    Represents a single source and its associated data.
    """
    def __init__(self, source_name: str, short_label: str, radio_coord: SkyCoord):
        self.source_name = source_name
        self.short_label = short_label
        self.radio_coord = radio_coord
        self.best_data = {}
        self.host_coord = radio_coord
        self.checked_images = set()
        self.containing_images = set()

    def __repr__(self) -> str:
        return f"""{type(self).__name__}({self.source_name=!r}
        \t{self.short_label=!r}
        \t{self.radio_coord=!r}
        \t{self.host_coord=!r}
        \t{self.best_data=!r})\n\t"""

    def update_best_data(self, new_data: DataEntry):
        """
        Checks a FITS file and updates the best data for a given band if the new
        data is of higher quality (lower RMS and similar or better resolution).
        """
        try:
            if new_data.rms is None: return

            # TODO: add validation check here that region around source (~30"x30") is <30% NaNs or fill_factor > 0.7, otherwise return
            
            if not self.has_band(new_data.band):
                print(f"\n  -> Found new '{new_data.band}' data for {self.source_name} in {'/'.join(new_data.filepath.split('/')[-2:])}"
                            +f" (RMS: {new_data.rms:.2e})")
                self.best_data[new_data.band] = new_data
            elif new_data.rms < self.best_data[new_data.band].rms or new_data.pix_res < self.best_data[new_data.band].pix_res:
                # Even though the RMS or resolution is better, ensure the other metric is not drastically worse
                # In this case, accept higher-res if not 1.5x worse depth (or twice as high res), accept deeper if not 2x worse resolution
                if new_data.rms < self.best_data[new_data.band].rms and new_data.pix_res <= 2*self.best_data[new_data.band].pix_res:
                    print(f"\n  -> Found deeper '{new_data.band}' data for {self.source_name} in {'/'.join(new_data.filepath.split('/')[-2:])}"
                            +f" (New RMS: {new_data.rms:.2e} < Old RMS: {self.best_data[new_data.band].rms:.2e})")
                    self.best_data[new_data.band] = new_data
                # < 0.95 *, to account for rounding innaccuracies
                elif new_data.pix_res < 0.95*self.best_data[new_data.band].pix_res and new_data.rms <= 3*self.best_data[new_data.band].rms:
                    if new_data.pix_res < 0.5*self.best_data[new_data.band].pix_res or new_data.rms <= 1.5*self.best_data[new_data.band].rms:
                        print(f"\n  -> Found higher-res '{new_data.band}' data for {self.source_name} in {'/'.join(new_data.filepath.split('/')[-2:])}"
                                +f" (New Pix Res: {new_data.pix_res.to(u.arcsec):.2f} < Old Pix Res: {self.best_data[new_data.band].pix_res.to(u.arcsec):.2f})")
                        self.best_data[new_data.band] = new_data

        except Exception as e:
            print(f"\n[ERROR]\tupdate_best_data: Failed to update best data for {self.source_name} with {new_data.filepath}: {e}")

    @staticmethod
    def get_coord_to_closest_source(fits_file: str, coord: SkyCoord, cutout_len: u.Quantity=1*u.arcmin,
                                    max_separation: u.Quantity=1*u.arcsec, debug_image: bool=False) -> list[SkyCoord, u.Quantity]:
        '''
        Return the coordinates to the closest source in the image, if within `max_separation`.
        '''
        # NOTE: Seems broken?

        if not skycoord_in_image(fits_file, coord):
            print(f"[WARNING]\tCoordinates {coord.to_string('hmsdms')} not contained in image WCS.")
            return coord

        with fits.open(fits_file) as hdul:
            hdu = get_image_hdu(hdul)

            # Make a medium sized cutout around the coord
            stamp = nddata_utils.Cutout2D(hdu.data,coord,cutout_len,WCS(hdu.header,fobj=hdul))
            wcs = stamp.wcs
            hdu.data = stamp.data
            hdu.header = stamp.wcs.to_header()
            if 'RADESYS' in hdu.header: frame = hdu.header['RADESYS'].strip(' ').lower()
            elif 'RADECSYS' in hdu.header: frame = hdu.header['RADECSYS'].strip(' ').lower()
            else: frame = 'icrs'

            # First estimate the background using sigma-clipped stats
            mean, _, std = sigma_clipped_stats(hdu.data, sigma=3.0, maxiters=15)
            data_const_bkg_subbed = hdu.data - mean
            threshold = 3.0 * std
        
        # Run photutils.segmentation and deblend the sources, returning the coordinates of the closest source if within `max_separation`
        finder = photutils.segmentation.SourceFinder(npixels=5, deblend=True, nlevels=32, contrast=0.001, progress_bar=False)
        segment_map = finder(data_const_bkg_subbed, threshold=threshold)

        # Pick closest segment to centroid from table (without looping through)
        closest_source = None
        new_coord = coord
        sep2d_match = 0 * u.arcsec

        if segment_map:
            # Make table from the deblended image
            cat = photutils.segmentation.SourceCatalog(data_const_bkg_subbed, segment_map, wcs=wcs)
            tab = cat.to_table()
            source_spread = SkyCoord(tab['sky_centroid'], frame=frame)

            idx_match, sep2d_match, dist3d_match = coord.match_to_catalog_sky(source_spread)
            if sep2d_match < max_separation:
                closest_source = tab[idx_match]
                new_coord = SkyCoord(closest_source['sky_centroid'], frame=frame)

        if closest_source is None:
            # Smooth the image using a Gaussian kernel if no deblended source is found within the max_separation and repeat
            kernel = photutils.segmentation.make_2dgaussian_kernel(5.0, size=11)
            convolved_data = convolution.convolve(data_const_bkg_subbed, kernel)

            finder = photutils.segmentation.SourceFinder(npixels=5, deblend=True, nlevels=32, contrast=0.001, progress_bar=False)
            segment_map = finder(convolved_data, threshold=threshold)

            if segment_map:
                # Make table from the deblended image
                cat = photutils.segmentation.SourceCatalog(convolved_data, segment_map, wcs=wcs)
                tab = cat.to_table()
                source_spread = SkyCoord(tab['sky_centroid'], frame=frame)

                idx_match, sep2d_match, dist3d_match = coord.match_to_catalog_sky(source_spread)
                if sep2d_match < max_separation:
                    closest_source = tab[idx_match]
                    new_coord = SkyCoord(closest_source['sky_centroid'], frame=frame)
            else:
                cat = None
                pass    # Rather than raise a warning, just accept that no source was found

        if closest_source:
            print(f"[INFO]\t{fits_file}:\n"
                  +f"Closest source to {coord.to_string('hmsdms', precision=4)}"
                  +f" is {new_coord.to_string('hmsdms', precision=4)}"
                  +f" with separation {sep2d_match.to(u.arcsec)[0]:.3f}.")
        
        if debug_image:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': wcs})
            ax.imshow(stamp.data, origin='lower', cmap='Greys', norm=simple_norm(stamp.data,percent=99.5)) #vmin=-threshold, vmax=threshold)#np.nanmax(stamp.data))
            if cat: cat.plot_kron_apertures(ax=ax, color='cyan', lw=1)
            if closest_source: ax.plot(closest_source['xcentroid'], closest_source['ycentroid'], marker='x',
                                       color='red', ms=10, label='Closest source')
            ax.plot(*stamp.wcs.world_to_pixel(coord), marker='x', color='yellow',ms=6,
                    transform=ax.get_transform(stamp.wcs), label='Radio coord')
            handles, _ = ax.get_legend_handles_labels()
            if cat: handles.append(patches.Patch(edgecolor='cyan', facecolor='none', lw=1, label='Detected sources'))
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            ax.set_title(f"{fits_file[:int(len(fits_file)/2)]}\n{fits_file[int(len(fits_file)/2):]}")
            plt.show()

        return new_coord, sep2d_match
    
    def find_host(self, max_separation: u.Quantity, **kwargs):
        """
        Iterates through optical/IR images to find the best host position. Ideal frame is ICRS (convert others to this).
        """
        debug = kwargs.get('debug', False)

        host_coords = []
        host_seps = []
        host_interseps = 0*u.deg
        
        optical_ir_bands = ['u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'K', 'Ks']
        image_paths = [de.filepath for band, de in self.best_data.items() if band in optical_ir_bands]

        for im_file in image_paths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                host_coord, host_sep = SourceEntry.get_coord_to_closest_source(im_file, self.radio_coord,
                                                    cutout_len=20*u.arcsec, max_separation=max_separation, debug_image=debug)
            if host_coord is not self.radio_coord:
                if isinstance(host_coord.frame, FK5):
                    # Re-construct in ICRS to avoid issues of missing frame attributes such as proper motions
                    host_coord = SkyCoord(ra=host_coord.ra, dec=host_coord.dec, frame='icrs')
                host_coords.append(host_coord)
                host_seps.append(host_sep.value)
        if len(host_coords) > 0:
            _, _, host_interseps, _ = SkyCoord(host_coords).search_around_sky(SkyCoord(host_coords), max_separation*3)
            print(f"[INFO]\tMean interseparation of all host coords with themselves: {np.mean(host_interseps.to(u.arcsec)):.3f}")
        
        if len(host_coords) > 0 and np.mean(host_interseps) < max_separation/3:
            weights = 1 / np.array(host_seps)**3
            new_coord = weighted_average_skycoord(host_coords, weights)
            offset = self.radio_coord.separation(new_coord)
            print(f"[INFO]\tUpdated coordinates for {self.source_name} to {new_coord.to_string('hmsdms', precision=4)}"
                +f" (offset: {offset.to(u.arcsec):.3f}).")
            self.host_coord = new_coord
        elif np.mean(host_interseps) > max_separation/3:
            print(f"[WARNING]\tPossibly degenerate hosts for {self.source_name}, using initial coordinates.")
            self.host_coord = self.radio_coord
        else:
            print(f"[INFO]\tNo host found for {self.source_name}, using initial coordinates.")
            self.host_coord = self.radio_coord


    def has_band(self, band_name: str) -> bool:
        return any(band_name.lower() == band.lower() for band in self.best_data)

class DataTool:
    """
    Class to discover and hold the best available imaging data for a list of sources.
    It intelligently updates its database as new sources or images are added.
    """
    def __init__(self, search_directories: list[str] = None):
        self.search_directories = search_directories if search_directories else []
        self.sources: list[SourceEntry] = []
        self.all_data: list[DataEntry] = []   # Mostly for debug purposes
        self.checked_filepaths = set() # Master set of all file paths ever seen
        self.ignored_filepaths = set() # File paths to ignore, can be added to if images are deemed invalid or not useful

    def __repr__(self) -> str:
        return f"""{type(self).__name__}({self.search_directories=!r})"""

    def load_from_file(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data.__dict__)
        print(f"Database loaded from {filepath}. Contains {len(self.sources)} sources and {len(self.checked_filepaths)} checked images.")

    def save_to_file(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Database saved to {filepath}.")

    def add_source(self, source_name: str, short_label: str, radio_coord: SkyCoord):
        """
        Adds a new source and checks it against all previously known images.
        """
        if any(s.source_name == source_name for s in self.sources):
            print(f"Source {source_name} already in database. Skipping.")
            return

        print(f"Adding new source: {source_name}. Checking against {len(self.checked_filepaths)} known images.")
        source = SourceEntry(source_name, short_label, radio_coord)
        self.sources.append(source)

        # Check the new source against all previously found images
        for de in self.all_data:
            try:
                if skycoord_in_image(de.filepath, source.radio_coord):
                    source.update_best_data(de)
                    source.containing_images.add(de.filepath)
            except Exception as e:
                print(f"[ERROR]\tadd_source: Could not process {de.filepath} for new source {source_name}: {e}")
            source.checked_images.add(de.filepath) # Mark as checked for this source

    def add_data(self, filepath: str) -> DataEntry:
        """
        Adds new data if not already indexed. Returns reference to the new DataEntry.
        """
        if any(de.filepath == filepath for de in self.all_data):
            print(f"Data in {filepath} already indexed in database. Skipping.")
            return None
        
        try:
            with fits.open(filepath) as hdul:
                if not validate_image(get_image_hdu(hdul)):
                    raise ValueError("Image is mostly NaNs!")
            
                astronomy_method = get_astronomy_method(hdul)
                wavelength = get_wavelength(hdul)
                band = get_band(hdul)
                if band == '':
                    band = f"{wavelength.to(u.um).value:.2f}um"
                instrument = get_instrument(hdul)
                origin = get_hdu_with(hdul, 'ORIGIN').header.get('ORIGIN', instrument)
                rms = calculate_rms(hdul)
                pix_res = get_pix_res(hdul)

                new_entry = DataEntry(filepath=filepath,
                                    astronomy_method=astronomy_method,
                                    instrument=instrument,
                                    origin=origin,
                                    band=band,
                                    wavelength=wavelength,
                                    rms=rms,
                                    pix_res=pix_res)
                self.all_data.append(new_entry)
                return new_entry
            
        except Exception as e:
            print(f"\n[ERROR]\tadd_data: Could not add data in {filepath}: {e}")
            return None

    def scan_local_directories(self, verbose: bool=True):
        """
        Scans local directories for new images and checks them against all sources.
        """
        print("Scanning local directories for FITS files...")
        all_fits_files = []
        for data_dir in self.search_directories:
            found_files = glob.glob(f'{data_dir}/**/*.fits', recursive=True)
            if len(self.ignored_filepaths) > 0:
                found_files = [s for s in found_files if all(ignored_path not in s for ignored_path in self.ignored_filepaths)]
            all_fits_files.extend(found_files)

        new_files = set(all_fits_files) - self.checked_filepaths
        if not new_files:
            print("No new images found.")
            return
        
        print(f"Found {len(new_files)} new images. Checking against {len(self.sources)} sources.")

        if verbose:
            stdout_destination = stdout
        else:
            stdout_destination = None
        # Check each new image against all existing sources
        for i, new_filepath in enumerate(new_files):
            print(f"\rAdding {new_filepath}\t({i+1}/{len(new_files)})" + " "*50, end='')
            stdout.flush()

            with redirect_stdout(stdout_destination):
                new_data = self.add_data(new_filepath)

                for source in self.sources:
                    if new_filepath in source.checked_images:
                        continue # This source has already been checked against this file for some reason
                    try:
                        if new_data and new_data.skycoord_in_image(source.radio_coord):
                            source.update_best_data(new_data)
                            source.containing_images.add(new_filepath)
                        source.checked_images.add(new_filepath) # Mark as checked for this source
                    except Exception as e:
                        print(f"[ERROR]\tscan_local_directories: Could not process {new_filepath} for source {source.source_name}: {e}")

        self.checked_filepaths.update(new_files)
        print("Finished scanning local directories.")

    def set_ignored_filepaths(self, ignore_paths: list[str]):
        self.ignored_filepaths = set(ignore_paths)
    
    def add_ignored_filepaths(self, ignore_paths: list[str]):
        self.ignored_filepaths.update(set(ignore_paths))

    def find_best_host_coordinates(self, max_separation=2.5 * u.arcsec, **kwargs):
        print("Finding best host coordinates...")
        for source in self.sources:
            source.find_host(max_separation, **kwargs)
        print("Finished host coordinate refinement.")
