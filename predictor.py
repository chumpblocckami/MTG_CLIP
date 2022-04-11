import faiss
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
from tqdm import tqdm
import os 
from imgbeddings import imgbeddings

class Predictor():
  def __init__(self):
    self.embeddings = np.load("embeddings/embeddings.npy")
    self.imgs = [Image.open(f"./dataset/{img}") for img in tqdm(os.listdir("./dataset/"),desc="Loading images...")]
    self.embedder = imgbeddings(pca="pca.npz")
    self.index = self.prepare_index()

  def prepare_index(self):
    index = faiss.index_factory(self.embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(normalize(self.embeddings))
    return index

  def predict(self,path):
    example = Image.open(path)
    q_embedding = self.embedder.to_embeddings(path)
    distances, indices = self.index.search(normalize(q_embedding), 10)

    imgs_to_display = 5
    fig, axs = plt.subplots(1,imgs_to_display+1,figsize=(10,8),dpi=64)
    axs[0].imshow(example)
    axs[0].set_title("Target")
    n = 1
    for idx, dist in zip(indices[0][:imgs_to_display],distances[0][:imgs_to_display]):
      axs[n].imshow(self.imgs[idx])
      title = "Similarity "+str(round(dist,2))
      axs[n].set_title(title)
      axs[n].axis("off")
      n = n+1
    plt.tight_layout()
    plt.savefig('results.png')


if __name__ == "__main__":
  predictor = Predictor()
  predictor.predict("./images/example3.jpg")