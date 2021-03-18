# Zone of Influence

Use these tools to help process data from the model output files to rasterize surfaces.

While the tool is designed for the zone of influence, you can also utilize it to create surfaces of the head/pressure output of a model.

Tool needs to be run in both arcgis pro and arcgis desktop. There is some improvements made in pro that significantly simplify the processing code, so this is why its split. We don't have a 3D analyst permission in pro, so this then needs to be run in desktop. If we eventually get 3D in pro, we would run both in pro.

## Process Model - Pro

This loads the data from the model into a geodatabase for processing into rasters. This is the conversion and merging of the junctions and model results to a single point feature class.

Tool will process either a daily average feature OR a complete daily feature with timestamp. The complete feature is really only useful if visualizing time based results and likely isn't going to be the default. As such, it is set to export the average feature only unless otherwise checked.

Unless you need the full results, running the average only will really reduce the processing time. In testing, it seems that 22 scenarios (full ZOI) took around an hour processing the full dataset. Average only took around 10 min.

## Generate Rasters- Desktop

This processes the results from the processed geodatabase into the final raster files.

With a cell size of 20, it takes approx. 5 min for 22 scenarios.

## Future changes

### **Process model**

1. add option for average/max day scenarios
3. manually type out scenarios? This could be good to manually load a single scenario in
4. support for pipe features or valve features?
