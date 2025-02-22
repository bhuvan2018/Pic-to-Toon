input_file_uri = "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"

response = algo.pipe({
    "data_uri": input_file_uri,
    "data_type": 1,
    "datastore": None
}).result

print(response)  # Check if the response contains an output URL
