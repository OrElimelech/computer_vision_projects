###########################################################################
# Imports
###########################################################################
import os
from typing import Union
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass


###########################################################################
# XmlParser Class
###########################################################################
@dataclass
class XmlParser:
    """A data class that stores and retrieve the xml parsed data"""
    __xml_path: str

    def __post_init__(self):
        assert os.path.exists(self.__xml_path), f"given input path does not exist: {self.__xml_path}"
        self.__tree_root = self.__parse_xml()

    ####################################
    # Private
    ####################################

    def __parse_xml(self) -> ElementTree:
        """Creates a parsing xml object
        Returns:
            ElementTree: a root of the xml object
         """
        xml_tree = ElementTree.parse(self.__xml_path)
        return xml_tree.getroot()

    ####################################
    # Public
    ####################################

    def get_xml_value(self, attribute: str, value: Union[None, str] = None,
                      variable_type: Union[callable, None] = None) -> any:
        """Retrieves a value in the xml
        args:
            attribute(str): attribute inside the xml
            value(str/None): value related to the given attribute, if None attribute contains a value
            variable_type(callable/None): a function to transform the given value, if None will be returned as a string

        Returns:
            any: a root of the xml object
         """
        try:
            if value is None:
                value = [tag.text for tag in self.__tree_root.findall(attribute)][0]
            else:
                value = [tag.find(value).text for tag in self.__tree_root.findall(attribute)][0]
        except IndexError:
            print(f"given attribute is invalid ({attribute}) check xml: {self.__xml_path}")
            return None
        except AttributeError:
            print(f"given value is invalid ({value}) check xml: {self.__xml_path}")
            return None
        if variable_type:
            value = variable_type(value)
        return value

    @property
    def get_xml_tree_root(self) -> ElementTree:
        """Returns a parsing xml object root
        Returns:
            ElementTree: a root of the xml object
         """
        return self.__tree_root
