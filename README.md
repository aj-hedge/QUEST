## QUEST: QUerying, Extraction and Survey Tool

A Python-based alternative to setups like using `Aladin` in conjunction with its plugins (`AperturePhotometryPlugin` or `PhotControlPlug`), whilst allowing more head-room for custom user scripting. Currently designed for retrieving the deepest images in a given band from a given archive, and prioritising the deepest images in a given band (with compromise for better resolution) from the ones indexed on-disk.

### Future development goals
- parallelise both querying (asynchronous within each coord's query) and checking if local images contain `SourceEntry` coord (when it comes to `update_best_data`, will have to either parallelise the `SourceEntry` instances being looped over, or parallelise different wavelength band images being checked for a `SourceEntry` -- or both)
- separate deep-prioritised mode for continuum and a cadence-prioritised mode for transient lightcurves
- refactor implementation of survey query/fix functions & functions to convert data units per-pixel to flux densities, into an expandable registry (with template examples) in the plugins module (where users can contribute their own into the module source directly, or utilise the registry decorator on their own functions)
- GUI plugin to preview an image with matplotlib (interactive) and use a cursor widget to select SkyCoord(s) to send to the source input list