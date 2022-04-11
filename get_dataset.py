import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os

from transformers import RoFormerForQuestionAnswering 

class Dataset():
    def __init__(self):
        self.write_path = "./dataset/"
        self.metadata = []

    def download_metadata(self,editions):
        data = []
        for edition in editions:
            query_string = json.loads(requests.get(f'https://api.scryfall.com/sets/{edition}').text)["uri"]
            search_uri = json.loads(requests.get(query_string, params="fuzzy").text)["search_uri"]
            data.extend(json.loads(requests.get(search_uri, params="fuzzy").text)["data"])
        return data

    def save_artworks(self):
        for doc in tqdm(self.metadata,desc="Downloading artworks..."):
            try:
                img_data = requests.get(doc["image_uris"]["art_crop"]).content
            except:
                print(f"{doc['name']} does not have any image uris")
                continue
            artwork = Image.open(BytesIO(img_data))
            title = f"{doc['set']}_{doc['name']}".replace("//","-").lower().strip()
            if f"{title}.png" not in os.listdir(self.write_path):
                artwork.save(f"{self.write_path}/{title}.png")

    def download(self, editions):
        self.metadata = self.download_metadata(editions=editions)
        self.save_artworks()

if __name__ == "__main__":
    dataset = Dataset()
    dataset.download(editions=['chk','bok','sok','neo'])
