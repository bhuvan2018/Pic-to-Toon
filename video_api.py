import algorithmia
input_file_uri = "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
algo = algorithmia.client('YOUR_API_KEY').algo('algo://YOUR_ALGORITHM')
response = algo.pipe({
    "data_uri": input_file_uri,
    "data_type": 1,
    "datastore": None
}).result
print(response)