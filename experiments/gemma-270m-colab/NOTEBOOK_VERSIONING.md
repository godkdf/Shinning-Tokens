# Notebook version policy

The user requires every revision to create a new notebook version and filename.
Never overwrite, regenerate in place, or replace a previously delivered notebook.
Preserve earlier notebooks as historical artifacts, including failed versions.
Use exclusive file creation for new notebooks so an existing filename cannot be
silently replaced. Shared runtime/data caches may retain their names for reuse.

## User-requested cleanup — 23 October 2026

The user explicitly requested removal of all pre-V4 notebooks in this Colab
training series. That request overrides preservation in the current checkout.
Only V4 and newer notebooks remain here. The 18 removed notebooks are preserved
at Git commit `d1143a10cf5ecef2ed4b8eeb8cdfe5f2edc74a81`; source/helper directories
remain historical material. This does not authorize overwriting V4+ notebooks.
The V4 builder reads its historical base from that commit when absent locally.
