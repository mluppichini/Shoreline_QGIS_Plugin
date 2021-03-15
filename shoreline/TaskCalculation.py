from PyQt5.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QVariant, Qt, QObject, QTimer, QEvent
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QAction, QFileDialog, QProgressBar, QProgressDialog
from PyQt5.QtWidgets import (QApplication, QWidget, QToolTip, QPushButton, QMessageBox, QToolButton)
from qgis.gui import QgsMessageBar, QgisInterface, QgsMapCanvas
import os
from qgis.core import *
from qgis.analysis import QgsZonalStatistics
from qgis.utils import iface

from qgis.core import QgsVectorLayer, QgsDataSourceUri
from math import pi, sqrt
import processing
import matplotlib.pyplot as plt
import numpy as np

# from PIL import Image
import traceback
import collections
import time
# Initialize Qt resources from file resources.py
from .resources import *

import zipfile
import os.path
import json
import shutil
import sys
import subprocess
import csv
import datetime
import time
import ftplib
from ftplib import FTP
from pyproj import Proj, transform
import pandas as pd
import numpy as np
from osgeo import gdal, ogr
from osgeo.gdalconst import *
#from scipy.interpolate import interp1d



class ExecuteShorelineCalculation(QgsTask):
    """Here we subclass QgsTask"""

    def __init__(self, desc, var):

        QgsTask.__init__(self, desc)
        self.prog = var[0]
        self.dx = var[1]
        self.r2Limit = var[2]
        self.dem_lay = var[3]
        self.transects_lay = var[4]
        self.field_order = var[5]
        self.output = var[6]
        self.dlg = var[7]
        self.id_failed = []



    def run(self):
        try:
            df_tot = pd.DataFrame()
            self.dem_lay_path = self.dem_lay.dataProvider().dataSourceUri()
            dict_features = {}
            count = 0
            for transetto in self.transects_lay.getFeatures():
                dict_features[count] = [transetto[self.field_order], transetto]
                count += 1
            dfFeatures = pd.DataFrame.from_dict(dict_features, orient='index')
            dfFeatures.columns = [self.field_order, "Feature"]
            dfFeatures = dfFeatures.sort_values(self.field_order)

            ds = gdal.Open(self.dem_lay_path, GA_ReadOnly)
            geotransform = ds.GetGeoTransform()
            cellx = geotransform[1]  # /* w - e pixel resolution */
            celly = -geotransform[5]  # /* n-s pixel resolution */
            Xori = geotransform[0] + cellx / 2  # /* top left x */
            Yori = geotransform[3] - celly / 2  # /* top left y */
            band = ds.GetRasterBand(1)
            noData = band.GetNoDataValue()

            count_perc = 0
            for row in dfFeatures.itertuples():
                count_perc += 1
                percentuale = count_perc *100/len(dfFeatures)
                self.prog.setValue(percentuale)
                try:
                    transetto = row.Feature
                    ln_geom = transetto.geometry()
                    vertici = ln_geom.vertices()
                    primo_punto = vertici.next()
                    secondo_punto = vertici.next()
                    #equation = interp1d([primo_punto.x(), secondo_punto.x()], [primo_punto.y(), secondo_punto.y()])
                    coefs = np.polyfit([primo_punto.x(), secondo_punto.x()], [primo_punto.y(), secondo_punto.y()], 1, full=False)
                    polynomial = np.poly1d(coefs)
                    distanza_x = secondo_punto.x() - primo_punto.x()
                    n_int = int(distanza_x / self.dx)
                    xnew = np.linspace(primo_punto.x(), secondo_punto.x(), num=n_int, endpoint=True)
                    ynew = polynomial(xnew)
                    znew = []
                    dlnew = []

                    for i in range(len(xnew)):
                        x = xnew[i]
                        y = ynew[i]
                        value_grid = self.get_grid_value_int(x, y, cellx, celly, Xori, Yori, band)[0]
                        znew.append(float(value_grid))
                        if i == 0:
                            dl = 0
                        else:
                            dl = dl + ((x - xnew[i - 1]) ** 2 + (y - ynew[i - 1]) ** 2) ** 0.5
                        dlnew.append(float(dl))
                    znew_control = np.array(znew, dtype=int)
                    if noData in znew_control:
                        self.id_failed.append(str(transetto[self.field_order]))
                        continue
                    # znew = np.array(znew)
                    # znew = (znew-min(znew))/(max(znew)-min(znew))
                    r2_list = []
                    for i in range(len(dlnew)):
                        if (dlnew[-1] - dlnew[i]) >= 0.5:
                            interv_dl = np.array(dlnew[i:])
                            interv_z = np.array(znew[i:])
                            P2 = np.polyfit(interv_dl, interv_z, 1)
                            p = np.poly1d(P2)
                            yhat = p(interv_dl)
                            ybar = sum(interv_z) / len(interv_z)
                            SST = sum((interv_z - ybar) ** 2)
                            SSreg = sum((yhat - ybar) ** 2)
                            R2 = SSreg / SST
                        else:
                            R2 = np.nan
                        r2_list.append(R2)

                    df = pd.DataFrame()
                    df['x'] = xnew
                    df['y'] = ynew
                    df['dl'] = dlnew
                    df['z'] = znew
                    df['r2'] = r2_list
                    df_max = df[df['r2'] >= self.r2Limit]
                    if df_max.empty == False:
                        first_index = df_max.index[0]
                        df_max = df_max[df_max.index == first_index]
                        df_tot = pd.concat([df_tot, df_max])
                    else:
                        self.id_failed.append(str(transetto[self.field_order]))
                except:
                    self.id_failed.append(str(transetto[self.field_order]))


            feats = []
            for row in df_tot.itertuples():
                x = row.x
                y = row.y
                feat = QgsPoint(x, y)
                feats.append(feat)
            line = QgsGeometry.fromPolyline(feats)
            df = pd.DataFrame()
            df['Description'] = ["shoreline"]
            df['Geometry'] = [line]
            try:
                #save = self.saveDataFrameToShp(df, "LineString", self.output, sr=3003, column_geom="Geometry")
                fields = list(df.columns)
                # remove x and y columns

                fields.remove('Geometry')
                list_newAttr = [QgsField('Description', QVariant.String)]
                # df = df.fillna(NULL)
                print ("ok333")
                layer = QgsVectorLayer("LineString", "output", "memory")
                # crs = layer.crs()
                # crs.createFromId(3003)
                # layer.setCrs(crs)
                result = QgsVectorFileWriter.writeAsVectorFormat(layer, self.output , 'utf-8', self.transects_lay.crs(),
                                                                 driverName='ESRI Shapefile')

                layer = QgsVectorLayer(self.output , "output", "ogr")

                if not layer.isValid():
                    self.dlg.log_attivita.setTextColor(QColor("red"))
                    message = "Impossible to save shoreline file."
                    self.dlg.log_attivita.append(message)
                    self.dlg.log_attivita.setTextColor(QColor("black"))
                    return False
                layer.startEditing()
                pr = layer.dataProvider()  # need to create a data provider
                pr.addAttributes(list_newAttr)  # define/add field data type
                layer.commitChanges()
                ListFeatures = []
                colonne = layer.fields()
                for index, rows in df.iterrows():
                    print (rows)
                    rowList = []
                    for column in fields:
                        rowList.append(rows[column])
                    feature = QgsFeature()
                    feature.setGeometry(rows["Geometry"])
                    feature.setFields(colonne)
                    feature.setAttributes(rowList)
                    ListFeatures.append(feature)
                    print ("pippo")

                layer.startEditing()
                pr.addFeatures(ListFeatures)
                layer.commitChanges()
                ExecuteShorelineCalculation.output = layer
            except:
                self.dlg.log_attivita.setTextColor(QColor("red"))
                message = "Impossible to save shoreline file."
                self.dlg.log_attivita.append(message)
                self.dlg.log_attivita.setTextColor(QColor("black"))
                print(traceback.format_exc())
                return False

            return True

        except Exception:
            print (traceback.format_exc())
            return False

    def get_grid_value_int(self, x_pt, y_pt, dx, dy, X0, Y0,
                           band):  # x del punto, y del punto, cell size x, cell size y, coord origine X0 e Y0, banda del raster

        diff_x = x_pt - X0
        diff_y = Y0 - y_pt

        col_tot = diff_x / float(dx)
        row_tot = diff_y / float(dy)

        n_row_pt = int(diff_y / float(dy))
        n_col_pt = int(diff_x / float(dx))
        lx = (x_pt - (X0 + dx * n_col_pt)) / dx
        ly = ((Y0 - dy * n_row_pt) - y_pt) / dy
        val_cella_x0y0 = float(
            band.ReadAsArray(n_col_pt, n_row_pt, 1, 1)[0][0])  # restituisce il valore dando la banda e gli offset
        val_cella_y1 = float(band.ReadAsArray(n_col_pt, n_row_pt + 1, 1, 1)[0][0])
        val_cella_x1 = float(band.ReadAsArray(n_col_pt + 1, n_row_pt, 1, 1)[0][0])
        val_cella_x1y1 = float(band.ReadAsArray(n_col_pt + 1, n_row_pt + 1, 1, 1)[0][0])

        # Gestisco i no data
        noData = band.GetNoDataValue()
        if val_cella_x0y0 == noData: val_cella_x0y0 = val_cella_x1
        if val_cella_x1y1 == noData: val_cella_x1 = val_cella_x0y0
        if val_cella_y1 == noData: val_cella_y1 = val_cella_x1y1
        if val_cella_x1y1 == noData: val_cella_x1y1 = val_cella_y1

        int_x = val_cella_x0y0 + lx * (val_cella_x1 - val_cella_x0y0)
        int_x2 = val_cella_y1 + lx * (val_cella_x1y1 - val_cella_y1)

        if val_cella_x0y0 == noData:
            val_interpolato = int_x2
        elif val_cella_x1y1 == noData:
            val_interpolato = int_x
        else:
            val_interpolato = int_x + ly * (int_x2 - int_x)

        # print val_interpolato
        array = [val_interpolato, val_cella_x0y0, val_cella_y1, val_cella_x1, val_cella_x1y1, lx, ly]

        return array

    def saveDataFrameToShp(self, df, type_geom, shape, sr=3003, column_geom="Geometry"):
        try:
            print(df)
            print("pippo")
            return

            fields = list(df.columns)
            # remove x and y columns
            fields.remove(column_geom)
            list_newAttr = []
            # df = df.fillna(NULL)

            for column in fields:
                if df[column].dtype == np.float64 or df[column].dtype == np.float32:
                    field = QgsField(column, QVariant.Double, 'double', 10, 3)
                elif df[column].dtype == np.int64:
                    field = QgsField(column, QVariant.Int)
                elif df[column].dtype == np.object:
                    field = QgsField(column, QVariant.String)
                elif df[column].dtype == "datetime64[ns]":
                    field = QgsField(column, QVariant.DateTime)
                else:
                    field = ''
                list_newAttr.append(field)
            layer = QgsVectorLayer("%s?crs=epsg:3003" % type_geom, "output", "memory")
            crs = layer.crs()
            crs.createFromId(sr)
            layer.setCrs(crs)
            result = QgsVectorFileWriter.writeAsVectorFormat(layer, shape, 'utf-8', layer.crs(), driverName='ESRI Shapefile')
            layer = QgsVectorLayer(shape, "output", "ogr")
            if not layer.isValid():
                return 1
            layer.startEditing()
            pr = layer.dataProvider()  # need to create a data provider
            pr.addAttributes(list_newAttr)  # define/add field data type
            layer.commitChanges()
            ListFeatures = []
            colonne = layer.fields()
            for index, rows in df.iterrows():
                # Create list for the current row
                rowList = []
                for column in fields:
                    rowList.append(rows[column])
                feature = QgsFeature()
                feature.setGeometry(rows[column_geom])
                feature.setFields(colonne)
                feature.setAttributes(rowList)
                ListFeatures.append(feature)
            layer.startEditing()
            pr.addFeatures(ListFeatures)
            layer.commitChanges()
        except:
            return 1

        return 0



    def finished(self, result):
        if self.id_failed != []:
            self.dlg.log_attivita.setTextColor(QColor("red"))
            message = "Impossible to calculate the shoreline for some transects"
            self.dlg.log_attivita.append(message)
            message= "ids: %s " %",".join(self.id_failed)
            #sel.dlg.log_attivita.append(message)
            self.dlg.log_attivita.setTextColor(QColor("black"))
