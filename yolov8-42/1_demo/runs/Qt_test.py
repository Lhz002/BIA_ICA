import os
from PySide6.QtCore import QCoreApplication

os.environ["QT_PLUGIN_PATH"] = r"d:\anaconda\envs\cv\lib\site-packages\PySide6\plugins"
QCoreApplication.addLibraryPath(os.environ["QT_PLUGIN_PATH"])
