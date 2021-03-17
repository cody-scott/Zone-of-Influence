# -*- coding: utf-8 -*-
# check if python 3 and enable pro tools
import sys
pro_tools = False
if sys.version_info[0] > 2:
    # can't import or run these if not in arcgis pro
    pro_tools = True
    import pandas as pd
    from arcgis.features import GeoAccessor, GeoSeriesAccessor
    from pathlib import Path
else:
    import arcview

import arcpy
import datetime
import os


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = ""

        # pro tools: load dbf and average feature
        # desk tools: generate raster results
        # List of tool classes associated with this toolbox
        if pro_tools:
            self.tools = [ProcessModelData]
        else:
            self.tools = [GenerateRasters]

class ProcessModelData(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Process Model Data"
        self.description = "process and convert model quality results to gis for use in generating rasters"
        self.canRunInBackground = False

        self.ignore_scenarios = ["AVE_DAY_CURRENT", "MAX_DAY_CURRENT", "BASE"]

    def getParameterInfo(self):
        """Define parameter definitions"""
        working_folder = arcpy.Parameter(
            displayName="Working Folder",
            name="out_folder",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        model_output_directory = arcpy.Parameter(
            displayName="Model Output Folder",
            name="mdl_out_fld",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        junctions_feature = arcpy.Parameter(
            displayName="Junctions Feature",
            name="jct_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        junctions_feature.filter.list = ["Point"]

        params = [
            working_folder,
            model_output_directory,
            junctions_feature
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        _fld, _mdl_out, _jct = [_.valueAsText for _ in parameters]
        self.do_work(_fld, _mdl_out, _jct)
        return

    def do_work(self, _folder, _model_output_directory, _junctions_feature):
        _folder = Path(_folder)
        _model_output_directory = Path(_model_output_directory)
        _junctions_feature = Path(_junctions_feature)
        
        self.start_time = datetime.datetime.now()
        arcpy.AddMessage("Start: {}".format(self.start_time.strftime("%Y-%m-%d %H:%M:%S")))

        working_folder = self.create_run_folder(_folder)
        gdb = self.create_run_gdb(working_folder)

        dbf_files = self.get_dbf_dict(_model_output_directory)

        junction_dataframe = self.load_junctions(_junctions_feature)

        results = []
        for i, _name in enumerate(dbf_files, 1):
            arcpy.AddMessage("Loading DBF: {} of {}".format(i, len(dbf_files)))
            _dbf_path = dbf_files[_name]
            results += self.process_dbf(gdb, junction_dataframe, _name, _dbf_path)
        return results

    def create_run_folder(self, _fld):
        dt = self.start_time.strftime("%Y%m%d_%H%M")
        work_dir = _fld/"results_{}".format(dt)
        run_fld = work_dir.mkdir()
        arcpy.AddMessage('Working folder: {}'.format(work_dir))
        return work_dir

    def create_run_gdb(self, _fld):
        _gdb = arcpy.CreateFileGDB_management(str(_fld), "Results")[0]
        arcpy.AddMessage('Working GDB: {}'.format(_gdb))
        return Path(_gdb)

    def get_dbf_dict(self, model_out_dir):
        model_out_dir = model_out_dir/'SCENARIO'

        arcpy.AddMessage("Getting scenarios from {}".format(model_out_dir))
        _scenarios = self._check_base_dir(model_out_dir)
        arcpy.AddMessage("{} scenarios located".format(len(_scenarios)))
        return _scenarios

    def _check_base_dir(self, _dir):
        results = {}
        for i, d in enumerate(_dir.iterdir(),1):
            if d.is_dir():
                dir_result = self._check_scenario_dir(d)
                if dir_result is not None:
                    results[d.name] = dir_result
        return results
                    
    def _check_scenario_dir(self, _dir):
        if _dir.name in self.ignore_scenarios: return
        for _ in _dir.iterdir():
            if _.name == "JunctOut.dbf":
                return _

    def process_dbf(self, _gdb, _junctions, _name, _dbf_path):
        scen_df = self.load_dbf(_dbf_path)
        scen_avg_df = self.average_scenario(scen_df)
        df = self.join_junction_scenario(
            _junctions,
            scen_df
        )
        avg_df = self.join_junction_scenario(
            _junctions,
            scen_avg_df
        )
        _all = self.save_to_gdb(df, _gdb, "{}".format(_name))
        _avg = self.save_to_gdb(avg_df, _gdb, "{}_AVG".format(_name))
        return [_all, _avg]

    def load_dbf(self, _dbf):
        arcpy.AddMessage("Loading DBF {} to Dataframe".format(_dbf))
        _df = pd.DataFrame.spatial.from_table(_dbf)
        _df = _df.set_index('ID')
        _df = _df[['TIME_STEP', 'TIME', 'DEMAND', 'HEAD', 'PRESSURE', 'QUALITY']]
        _df['TIME'] = _df['TIME'].apply(lambda x: self.time_to_datetime(x, datetime.datetime(2021, 1, 1)))
        return _df

    def time_to_datetime(self, value, start_time):
        value = value.split(" hrs")[0].split(":")
        value = start_time + datetime.timedelta(hours=int(value[0]), minutes=int(value[1]))
        value = value.strftime("%Y-%m-%d %H:%M:%S")
        return value

    def load_junctions(self, _jct):
        arcpy.AddMessage("Loading Junctions {} to DataFrame".format(_jct))
        _df = pd.DataFrame.spatial.from_featureclass(_jct)
        _df = _df.set_index('ID')
        _df = _df[['ZONE', 'ELEVATION', 'WATERTYPE', 'CLASS', 'DMDNODE', 'SHAPE']]
        return _df

    def join_junction_scenario(self, junctions_df, scenario_df):
        """Combine"""
        arcpy.AddMessage("Performing Join")
        _df = junctions_df.join(scenario_df, how="inner")
        return _df

    def average_scenario(self, scenario_df):
        """Generates the averaged scenrio to plot"""
        arcpy.AddMessage("Calculating Average Results")
        return scenario_df.groupby(scenario_df.index).mean()

    def save_to_gdb(self, _df, _gdb, _name):
        """Save dataframe back to a geodatabase"""
        arcpy.AddMessage("Saving to {}".format(_gdb/_name))
        _df = _df.reset_index() if _df.index.name == "ID" else _df
        _df.spatial.to_featureclass(_gdb/_name)
        return _gdb/_name

class GenerateRasters(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Generate Rasters"
        self.description = "Create ZOI Rasters from Input GDB"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        working_folder = arcpy.Parameter(
            displayName="Working Folder",
            name="out_folder",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        scenario_gdb = arcpy.Parameter(
            displayName="Scenario GDB",
            name="mdl_out_fld",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        clip_feature = arcpy.Parameter(
            displayName="Clip Feature",
            name="clip_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        clip_feature.filter.list = ["Polygon"]

        cell_size = arcpy.Parameter(
            displayName="Cell Size",
            name='cell_size',
            datatype="GPSACellSize",
            parameterType='Required',
            direction='Input'
        )

        target_field = arcpy.Parameter(
            displayName="Processing Field",
            name='target_field',
            datatype="GPString",
            parameterType='Required',
            direction='Input'
        )
        target_field.filter.type = "ValueList"
        target_field.values = "QUALITY"
        target_field.filter.list = ["QUALITY", "PRESSURE", "HEAD"]

        output_suffix = arcpy.Parameter(
            displayName="Raster Suffix",
            name='output_suffix',
            datatype="GPString",
            parameterType='Optional',
            direction='Input'
        )

        params = [
            working_folder,
            scenario_gdb,
            clip_feature,
            cell_size,
            target_field,
            output_suffix,
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        working_folder = parameters[0].valueAsText
        scenario_gdb = parameters[1].valueAsText
        clip_feature = parameters[2].valueAsText
        cell_size = parameters[3].valueAsText
        target_field = parameters[4].valueAsText
        output_suffix = parameters[5].valueAsText
    
        output_suffix = output_suffix if output_suffix != '' else ''

        self.do_work(
            working_folder, scenario_gdb, clip_feature,
            target_field, cell_size, output_suffix
        )

        return

    def do_work(self, working_folder, input_gdb, clip_feature=None, processing_field=None, cell_size=None, suffix=None):
        assert arcpy.CheckExtension("3D") == "Available"
        self.start_time = datetime.datetime.now()
        arcpy.AddMessage("Start: {}".format(self.start_time.strftime("%Y-%m-%s %H:%M:%S")))

        scenarios = self.get_scenarios(input_gdb)
        temp_dir = self.create_temp_folder(working_folder)
        scenario_folders = self.generate_scenarios(scenarios, temp_dir, clip_feature)

        out_gdb = self.create_output_gdb("Rasters", working_folder)

        rasters = [RasterProcessing(_) for _ in scenario_folders]
        for raster in rasters:
            raster.process_raster(
                scenario_name=raster.scenario_name,
                processing_field=processing_field,
                cell_size=cell_size,
                suffix=suffix,
                target_gdb=out_gdb
            )
        
    def get_scenarios(self, gdb):
        """Identifies average scenarios in a geodatabase"""
        arcpy.env.workspace = gdb
        arcpy.AddMessage("Getting Scenarios from {}".format(gdb))
        features = [_ for _ in arcpy.ListFeatureClasses()]
        features = [_ for _ in features if "AVG" in _.upper()]
        features = {_.split("_")[0]: os.path.join(gdb,_) for _ in features}
        return features

    def generate_scenarios(self, scenarios, _folder, _clip_feature=None):
        """Loop provided scenarios and create individual folders"""
        arcpy.AddMessage("Generating Scenario folders at {}".format(_folder))
        arcpy.AddMessage("{} scenarios located".format(len(scenarios)))
        scenario_folders = []
        for scenarios_name in scenarios:
            scenario = self.generate_scenario(
                scenarios_name,
                scenarios[scenarios_name],
                _folder,
                _clip_feature
            )
            scenario_folders.append(scenario)
        return scenario_folders

    def generate_scenario(self, _name, _scenario_data, _folder, _clip_feature=None):
        """
        Create a standard processing folder for the scenario
        This is more work then required for single thread, but it is essential to multithread
        You need to have each process holding its own lock per-folder

        _name: name of scenario
        _scenario_data: path to scenario data, will be from the parent gdb
        _folder: folder to create scenario folder in
        _clip_feature: feature to clip rasters to afterwards 
        """
        arcpy.AddMessage("Creating scenario: {}".format(_name))
        _fld = self.create_working_folder(_name, _folder)
        _gdb = self.create_working_gdb(_name, _fld)
        new_scenario = self.copy_scenario_data(_gdb, _scenario_data)
        new_clip = None if _clip_feature is None else self.copy_clip_feature(_gdb, _clip_feature)
        return _fld

    def create_working_folder(self, feature, _folder):
        """Create a working folder for the scenario"""
        arcpy.AddMessage("Creating working folder: {}\\{}".format(_folder, feature))
        new_dir = os.path.join(_folder, feature)
        if not os.path.isdir(new_dir): os.mkdir(new_dir)
        return new_dir

    def create_working_gdb(self, feature, _folder):
        """Create a working gdb for the scenario"""
        arcpy.AddMessage("Creating working geodatabse: {}\\{}".format(_folder, feature))
        return arcpy.CreateFileGDB_management(_folder, feature)[0]

    def copy_scenario_data(self, _gdb, _scenario_data):
        """Copy scenario data to working gdb"""
        arcpy.AddMessage("Copying scenario data")
        _scenario_name = _scenario_data.split("\\")[-1]
        _source = _scenario_data
        _target = os.path.join(_gdb, _scenario_name)
        return arcpy.Copy_management(_source, _target)[0]

    def copy_clip_feature(self, _gdb, _feature):
        """Copy clip feature into geodatabase"""
        arcpy.AddMessage("Copying Clip feature")
        _source = _feature
        _target = os.path.join(_gdb, "CLIP_FEATURE")
        return arcpy.Copy_management(_source,_target)[0]

    def create_temp_folder(self, _folder):
        """Create parent temp folder with current datetime, for uniqueness"""
        arcpy.AddMessage("Creating working folder")
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        new_dir = os.path.join(_folder,"temp_{}".format(dt))
        os.mkdir(new_dir)
        return new_dir

    def create_output_gdb(self, _name, _folder):
        _gdb_path = os.path.join(_folder, "{}.gdb".format(_name))
        if not os.path.exists(_gdb_path):
            arcpy.AddMessage("Creating output geodatabse: {}".format(_gdb_path))
            return arcpy.CreateFileGDB_management(_folder, _name)[0]
        else:
            arcpy.AddMessage("Output GDB Exists. Using {}".format(_gdb_path))
            return _gdb_path

class RasterProcessing(object):
    def __init__(self, scenario_folder):
        self.scenario_folder = scenario_folder
        self.scenario_name = self.scenario_folder.split("\\")[-1]
     
        self.gdb = self._identify_gdb(self.scenario_folder)
        self.gdb_feature_classes = self._get_feature_classes()
        self.clip_feature = self._identify_clip_feature()
        self.scenario_feature = self._identify_scenario_feature()

        self.output_raster = None

        arcpy.env.overwriteOutput = True

    def _get_feature_classes(self):
        arcpy.env.workspace = self.gdb
        return [_ for _ in arcpy.ListFeatureClasses()]

    def _identify_gdb(self, folder):
        """Find geodatabase in folder"""
        for _ in os.listdir(folder):
            if ".GDB" in _.upper():
                return os.path.join(folder, _)

    def _identify_clip_feature(self):
        """Get feature class matching CLIP_FEATURE"""
        for _ in self.gdb_feature_classes:
            if _ == "CLIP_FEATURE":
                return os.path.join(self.gdb, _)

    def _identify_scenario_feature(self):
        """Get the feature class matching scenario name"""
        for _ in self.gdb_feature_classes:
            if self.scenario_name in _:
                return os.path.join(self.gdb, _)

    def process_raster(self, scenario_name, processing_field=None, cell_size=None, suffix=None, target_gdb=None):
        """Generate ZOI Raster for scenario and fields
        Input is scenario name of the raster, field to process, cell size of raster and the suffix to apply

        Specifying the final geodatabase will help save a step in the processing.
        Optional. Multiprocessing will require independant saves and copying afterwards, but for single thread it saves a step.

        Scenario name is input since it can be easily changed for a time of day value in a loop.
        Time of day variable one will be another process all together i think.

        Args:
            scenario_name (string): name of scenario to apply to raster name
            processing_field (string, optional): Field to process IDW from. Defaults to None and uses "QUALITY".
            cell_size (string, optional): String of cell size. Defaults to None and uses "10".
            suffix (string, optional): Suffix to append to scenario name. Defaults to None.
            target_gdb (string, optional): Geodatabase to save final raster to. If None, then defaults to the processing folder gdb
        Returns:
            [type]: [description]
        """
        if arcpy.CheckExtension("3D") != "Available": raise("Need 3D Analyst")
        processing_field = "QUALITY" if processing_field is None else processing_field

        reclass_raster = True if processing_field == "QUALITY" else False
        cell_size = "10" if cell_size is None else cell_size

        target_gdb = self.gdb if target_gdb is None else target_gdb

        if suffix is not None:
            scenario_name += suffix

        output_raster = self._process_idw(
            self.gdb,
            target_gdb,
            scenario_name,
            self.scenario_feature,
            self.clip_feature,
            processing_field,
            cell_size,
            reclass_raster
        )
        
        self.output_raster = output_raster

        return output_raster

    def _process_idw(self, 
        working_gdb, 
        target_gdb,
        scenario_name, scenario_data, 
        clip_feature, 
        processing_field, cell_size, reclass):
        
        arcpy.env.overwriteOutput=True

        output_raster = os.path.join(target_gdb, scenario_name) # this is in output gdb, aka Raster.gdb
        local_raster = os.path.join(working_gdb, scenario_name) # this is in the temp folder

        first_raster = os.path.join("in_memory", "fr_{}".format(scenario_name))

        print(output_raster)

        if arcpy.Exists(output_raster):
            arcpy.Delete_management(output_raster)

        temp_rasters = []

        idw_result = arcpy.Idw_3d(scenario_data, processing_field, first_raster, cell_size)[0]
        temp_rasters.append(idw_result)

        if reclass:
            reclassRaster = arcpy.Reclassify_3d(idw_result,"VALUE","-999999 4.999999 0;5 2000000 1","in_memory\\reclass_{}".format(scenario_name))[0]
            temp_rasters.append(reclassRaster)

            idw_result = arcpy.Times_3d(idw_result, reclassRaster,"in_memory\\times_{}".format(scenario_name))[0]
            temp_rasters.append(idw_result)

        if clip_feature is not None:
            idw_result = arcpy.Clip_management(
                                            in_raster=idw_result,
                                            out_raster=local_raster,
                                            in_template_dataset=clip_feature,
                                            clipping_geometry=True
                                        )[0]
        else:
            idw_result = arcpy.CopyRaster_management(idw_result, local_raster)[0]
        
        idw_result = arcpy.CopyRaster_management(idw_result, output_raster)[0]

        arcpy.Delete_management("in_memory")

        return idw_result
