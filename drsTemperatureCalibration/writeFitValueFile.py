import drsTemperatureCalibration.drsFitTool as fitTool


def search_drs_files():
    fitTool.search_drs_files()


def save_drs_attributes():
    fitTool.save_drs_attributes()


def save_fit_values():
    fitTool.save_fit_values()


if __name__ == '__search_drs_files__':
    search_drs_files()

if __name__ == '__save_drs_attributes__':
    save_drs_attributes()

if __name__ == '__save_fit_values__':
    save_fit_values()
