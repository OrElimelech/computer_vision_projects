###########################################################################
# imports
###########################################################################
import os
import re
from typing import Iterator, Union
from itertools import cycle
from lib.xml_parser.xml_parser import XmlParser


###########################################################################
# XmlCarPlateDataGenerator
###########################################################################
class XmlCarPlateDataGenerator:
    """An object that constructs a xml data generator
        Args:
            xml_data_root_path(str): path root to the xml files of the data set
            image_data_root_path(str): path root to the images files of the data set
            crop_percentage_index(None/float): if none or bigger than 1 the whole list of found xml will be iterated, e
                                              else will start cropping from the calculated index
            crop_from_start(bool): true - indicates that the cropping will be done from the start of the found xml list,
                                          else from the end
    """

    def __init__(self, xml_data_root_path: str, image_data_root_path: str,
                 crop_percentage_index: Union[None, float] = None, crop_from_start: bool = True):
        self.__xml_data_path = self.__validate_input_path(xml_data_root_path)
        self.__image_data_path = self.__validate_input_path(image_data_root_path)
        self.__xml_generator = self.__build_xml_data_generator()
        self.__crop_from_start_flag = crop_from_start
        self.__crop_percentage_index = crop_percentage_index

    def __iter__(self) -> Iterator:
        """iterator method"""
        return self.__xml_generator

    def __next__(self) -> dict:
        """Builds the next value of the generator
        Returns:
            dict: structure of the generator value -> {"x_min": <int>,"y_min": <int>,"y_max": <int>,"x_max": <int>,
                                                        "image_path": <str>}
        """
        return next(self.__xml_generator)

    ####################################
    # Public
    ####################################

    def get_xml_files_num(self) -> int:
        """calculates the number of used xml files in the generator
        Returns:
            int: number of used xml files in the generator
        """
        return len(self.__get_xml_paths())

    ####################################
    # Private
    ####################################

    def __build_xml_data_generator(self) -> Union[dict, str]:
        """Builds the generator structure
        Returns:
            Union[dict, str]: generator structure  -> first index : bounding box structure, second index : image path
        """
        for xml_path in cycle(self.__get_xml_paths()):
            xml_parser_obj = XmlParser(xml_path)
            yield self.__get_bounding_box_data(xml_parser_obj), self.__get_image_path(xml_parser_obj)

    @staticmethod
    def __validate_input_path(path: str) -> str:
        """Validates the given input file to see if it exists
        Args:
            path(str): path input for validation
        Returns:
            str: if the path is real same directory path is returned

        """
        assert os.path.exists(path), f"path does not exist: {path}"
        return path

    @staticmethod
    def __get_bounding_box_data(xml_parser_obj: XmlParser) -> dict:
        """Retrieves the bounding box data
        Args:
            xml_parser_obj(XmlParser): initialized xml parser object
        Returns:
            dict: bounding box data structure

        """
        return {
            "x_min": xml_parser_obj.get_xml_value("object/bndbox", "xmin", int),
            "y_min": xml_parser_obj.get_xml_value("object/bndbox", "ymin", int),
            "y_max": xml_parser_obj.get_xml_value("object/bndbox", "ymax", int),
            "x_max": xml_parser_obj.get_xml_value("object/bndbox", "xmax", int)
        }

    def __get_image_path(self, xml_parser_obj: XmlParser) -> str:
        """Retrieves the image path data
        Args:
            xml_parser_obj(XmlParser): initialized xml parser object
        Returns:
            str: image path data

        """
        image_file__name = xml_parser_obj.get_xml_value("filename")
        return self.__validate_input_path(os.path.join(self.__image_data_path, image_file__name))

    def __get_xml_paths(self) -> list:
        """Retrieves the xml files paths

        Returns:
            list: xml paths list

        """
        xml_files = [os.path.join(self.__xml_data_path, file_name) for file_name in
                     os.listdir(self.__xml_data_path) if re.search("xml", file_name)]
        assert xml_files, f"no xml files were found in : {self.__xml_data_path}"
        if self.__crop_percentage_index:
            if self.__crop_from_start_flag:
                xml_files = xml_files[:round(len(xml_files) * self.__crop_percentage_index)]
            else:
                xml_files = xml_files[round(len(xml_files) * self.__crop_percentage_index):]
        return xml_files
