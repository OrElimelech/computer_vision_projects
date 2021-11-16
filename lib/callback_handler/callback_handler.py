###########################################################################
# imports
###########################################################################
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, \
    LearningRateScheduler, ModelCheckpoint


###########################################################################
# CallbackHandler Class
###########################################################################
class CallbackHandler:
    """An object that creates callback instances
        Args:
            callback_config(dict): a dictionary that contains all the relevant configuration of the callbacks
                                   instances
        Example:
            callback_config = {"EarlyStopping": {"monitor": 'val_loss','min_delta': 0,'patience': 3},
                               "ReduceLROnPlateau": {'monitor': 'val_loss', 'factor'"': 0.5}}
    """

    def __init__(self, callback_config: dict):

        self.__callback_configs = callback_config
        self.__call_back_struct = self.__callback_struct()
        self.__call_back_output = list()

    ####################################
    # Public
    ####################################

    def build_callback_objects(self) -> None:
        """Creates and adds callback instances to the list of callback instances"""
        for callback_type, callback_conf in self.__callback_configs.items():
            self.__add_call_back_object(callback_type, callback_conf)

    def reset_callback_list(self) -> None:
        """Resets the list of callback instances"""
        self.__call_back_output = list()

    @property
    def get_callback_output(self) -> list:
        """Returns the callback instances list

        Returns:
           list: callback instances list
        """
        return self.__call_back_output

    ####################################
    # Private
    ####################################

    @staticmethod
    def __callback_struct() -> dict:
        """Returns the structure of callback objects

        Returns:
           dict: structure of callback objects
        """
        return {"ReduceLROnPlateau": ReduceLROnPlateau,
                "EarlyStopping": EarlyStopping,
                "ModelCheckpoint": ModelCheckpoint,
                "LearningRateScheduler": LearningRateScheduler
                }

    def __add_call_back_object(self, call_back_type: str, callback_config: dict):
        """Creates a callback instance and adds it to the list
            Args:
                call_back_type(str): callback instance name
                callback_config(dict): configuration structure for a callback instance
         """
        assert bool(
            call_back_type in self.__call_back_struct), f"invalid callback type: {call_back_type}, " \
                                                        f"available types are: " \
                                                        f"{list(self.__call_back_struct.keys())}"

        if callback_config:
            callback_obj = self.__call_back_struct[call_back_type](**callback_config)
        else:
            callback_obj = self.__call_back_struct[call_back_type]
        self.__call_back_output.append(callback_obj)
