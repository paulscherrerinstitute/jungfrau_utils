import json


def load_default_channel_list(filename="/sf/bernina/config/com/channel_lists/default_channel_list.json"):

        with open(filename, 'r') as input_file:
            channels_config = json.load(input_file)

        channels = channels_config["channels"]
        return channels
        
