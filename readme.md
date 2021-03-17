# Zone of Influence

Use these tools to help process data from the model output files to rasterize surfaces.

While the tool is designed for the zone of influence, you can also utilize it to create surfaces of the head/pressure output of a model.

Tool needs to be run in both arcgis pro and arcgis desktop. There is some improvements made in pro that significantly simplify the processing code, so this is why its split. We don't have a 3D analyst permission in pro, so this then needs to be run in desktop. If we eventually get 3D in pro, we would run both in pro.

## Process Model - Pro

This loads the data from the model into a geodatabase for processing into rasters. This is the conversion and merging of the junctions and model results to a single point feature class

## Generate Rasters- Desktop

This processes the results from the processed geodatabase into the final raster files.