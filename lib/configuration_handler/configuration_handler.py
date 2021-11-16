###########################################################################
# imports
###########################################################################
import os
from json import load as json_load
from dataclasses import dataclass


###########################################################################
# ConfigurationHandler Class
###########################################################################

@dataclass
class ConfigurationHandler:
    """a configuration data class that contains the initial data for each project"""

    __settings_path: str = "settings.json"
    __validation_type_config: dict = None

    ####################################
    # Private
    ####################################

    def __repr__(self) -> str:
        """Returns a string of the loaded configuration data structure.

        Returns:
            str: configuration details
         """

        config_str = [f"{param}  : {value}" for param, value in
                      zip(self.__json_config.keys(), self.__json_config.values())]
        return "\n".join(config_str)

    def __post_init__(self) -> None:
        """post initialization processes - parsing the json file and validating the created data structure"""
        self.__json_config = self.__parse_json_file()
        self.___validation_type_configuration()

    def __parse_json_file(self) -> dict:
        """Parsing the json file into a data structure.

        Returns:
            dict: configuration dictionary data structure
         """
        assert os.path.exists(self.__settings_path), f"setting file was not found: {self.__settings_path}"
        with open(self.__settings_path, 'r') as json_file:
            json_conf = json_load(json_file)
        return json_conf

    def ___validation_type_configuration(self) -> None:
        """Validates the type of variables of the created configuration data type according tot the expected outcome"""
        if self.__validation_type_config:
            for param in self.__validation_type_config.keys():
                if not isinstance(self.__json_config[param], self.__validation_type_config[param]):
                    raise ValueError(
                        f"the following parameter is not according to the expected type of "
                        f"{self.__validation_type_config[param]} => {param}: {self.__json_config[param]}")
        else:
            print("no input type validation was required")

    ####################################
    # Public
    ####################################

    def get_config_value(self, param: str) -> any:
        """Returns a specific configuration value according to the given parameter.

        Args:
            param(str): a configuration string parameter

        Returns:
            any/None: configuration value, if the input parameter does not exist an error will be raised
         """
        try:
            return self.__json_config[param]
        except KeyError:
            raise ValueError(f"the following parameter is not part of the defined configuration: {param} ")
