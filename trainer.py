import numpy as np
from PIL import Image
import os
from imgbeddings import imgbeddings
from tqdm import tqdm


class Trainer():
    def __init__(self):
        self.write_path = "./embeddings/"
        self.read_path = "./dataset/"

        self.embedder = imgbeddings()  # todo: add gpu etc

        self.images = []
        self.embeddings = []

    def load_images(self):
        return [Image.open(f"{self.read_path}{img}") for img in tqdm(os.listdir(self.read_path), desc="Loading images...")]

    def get_embeddings(self):
        embeddings = self.embedder.to_embeddings(self.images)
        self.embedder.pca_fit(embeddings, 126)
        embeddings = self.embedder.pca_transform(embeddings)
        return embeddings

    def save(self):
        np.save(f"{self.write_path}embeddings.npy", self.embeddings)

    def train(self):
        self.images = self.load_images()
        self.embeddings = self.get_embeddings()
        self.save()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
